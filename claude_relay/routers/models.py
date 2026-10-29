"""GET /v1/models, the cached live model list behind it, and startup default-model resolution."""

import asyncio
import logging
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.security import HTTPAuthorizationCredentials

from claude_relay import constants
from claude_relay.auth import security, verify_api_key
from claude_relay.constants import (
    ANTHROPIC_MODELS_URL,
    ANTHROPIC_VERSION,
    CLAUDE_MODEL_ALIASES,
    CLAUDE_MODELS,
    DEFAULT_MODEL_FALLBACK,
    MODEL_LIST_CACHE_TTL_SECONDS,
    MODEL_LIST_ERROR_TTL_SECONDS,
    MODEL_LIST_REQUEST_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# Best-effort cache for Anthropic's live Models API.  The static constants remain
# the fallback so /v1/models keeps working for Claude CLI, Bedrock, Vertex, local
# development, and transient Anthropic API outages.
_model_list_cache: Dict[str, Any] = {"expires_at": 0.0, "models": None}
# Serializes cache refreshes so concurrent /v1/models requests at TTL expiry
# don't all stampede the upstream Anthropic API.
_model_list_lock = asyncio.Lock()


def _iso_to_unix(value: Any) -> Optional[int]:
    """Convert an Anthropic ISO-8601 'created_at' string to a unix timestamp."""
    if not isinstance(value, str):
        return None
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def _openai_model_from_anthropic(model_info: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an Anthropic ModelInfo object to OpenAI-compatible model metadata."""
    created = _iso_to_unix(model_info.get("created_at"))
    model: Dict[str, Any] = {
        "id": model_info["id"],
        "object": "model",
        "created": created if created is not None else int(datetime.now(timezone.utc).timestamp()),
        "owned_by": "anthropic",
    }

    # Preserve useful Anthropic metadata for clients that want it.  OpenAI clients
    # ignore unknown keys, and the existing id/object/owned_by shape is retained.
    for key in (
        "display_name",
        "created_at",
        "max_input_tokens",
        "max_tokens",
        "capabilities",
        "type",
    ):
        if key in model_info:
            model[key] = model_info[key]

    return model


def _fallback_model_payload() -> List[Dict[str, Any]]:
    now = int(datetime.now(timezone.utc).timestamp())
    return [
        {"id": model_id, "object": "model", "created": now, "owned_by": "anthropic"}
        for model_id in CLAUDE_MODELS
    ]


async def _fetch_anthropic_models() -> Optional[List[Dict[str, Any]]]:
    """Fetch all available models from Anthropic, returning None on fallback-worthy errors."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    headers = {
        "anthropic-version": ANTHROPIC_VERSION,
        "x-api-key": api_key,
    }
    beta_header = os.getenv("ANTHROPIC_BETA") or os.getenv("ANTHROPIC_BETA_HEADER")
    if beta_header:
        headers["anthropic-beta"] = beta_header

    params: Dict[str, Any] = {"limit": 1000}
    models: List[Dict[str, Any]] = []

    try:
        async with httpx.AsyncClient(timeout=MODEL_LIST_REQUEST_TIMEOUT_SECONDS) as client:
            while True:
                response = await client.get(ANTHROPIC_MODELS_URL, headers=headers, params=params)
                response.raise_for_status()
                payload = response.json()
                models.extend(
                    _openai_model_from_anthropic(model)
                    for model in payload.get("data", [])
                    if model.get("id")
                )

                if not payload.get("has_more") or not payload.get("last_id"):
                    break
                params["after_id"] = payload["last_id"]
    except Exception as exc:  # noqa: BLE001 - endpoint should degrade gracefully
        logger.warning("Failed to fetch Anthropic model list, using fallback: %s", exc)
        return None

    return models or None


async def get_available_models() -> List[Dict[str, Any]]:
    """Return live Anthropic models when possible, with cached static fallback."""
    if os.getenv("CLAUDE_MODELS_OVERRIDE", "").strip():
        return _fallback_model_payload()

    now = time.time()
    cached_models = _model_list_cache.get("models")
    if cached_models and now < float(_model_list_cache.get("expires_at", 0)):
        return cached_models

    async with _model_list_lock:
        # Recheck inside the lock so the first waiter populates the cache and
        # subsequent waiters return without re-fetching.
        now = time.time()
        cached_models = _model_list_cache.get("models")
        if cached_models and now < float(_model_list_cache.get("expires_at", 0)):
            return cached_models

        live_models = await _fetch_anthropic_models()
        if live_models:
            _model_list_cache.update(
                {"models": live_models, "expires_at": now + MODEL_LIST_CACHE_TTL_SECONDS}
            )
            return live_models

        fallback_models = _fallback_model_payload()
        # Use a short TTL on failure so transient outages don't suppress live
        # discovery for the full MODEL_LIST_CACHE_TTL_SECONDS window.
        _model_list_cache.update(
            {"models": fallback_models, "expires_at": now + MODEL_LIST_ERROR_TTL_SECONDS}
        )
        return fallback_models


def _pick_latest_sonnet(models: List[Dict[str, Any]]) -> Optional[str]:
    """Return the id of the newest Sonnet model in `models`, or None."""
    sonnets = [m for m in models if isinstance(m.get("id"), str) and "sonnet" in m["id"].lower()]
    if not sonnets:
        return None
    # Prefer Anthropic-provided created_at; fall back to the int `created` we set,
    # then to id-sort (date-suffixed ids sort correctly newest-last).
    sonnets.sort(
        key=lambda m: (
            _iso_to_unix(m.get("created_at")) or m.get("created") or 0,
            m["id"],
        )
    )
    return sonnets[-1]["id"]


async def resolve_default_model() -> Optional[str]:
    """Pick the latest Sonnet from /v1/models and store it as the default.

    Skipped when the operator pinned DEFAULT_MODEL via env var, or when no
    ANTHROPIC_API_KEY is configured (live discovery is the only auth-aware
    path; Bedrock, Vertex, and Claude CLI subscription users get the static
    DEFAULT_MODEL_FALLBACK).
    """
    if constants.DEFAULT_MODEL_ENV:
        return constants.DEFAULT_MODEL_ENV

    if not os.getenv("ANTHROPIC_API_KEY"):
        logger.info(
            "Live model discovery disabled (no ANTHROPIC_API_KEY); " "using fallback default %s",
            DEFAULT_MODEL_FALLBACK,
        )
        return None

    try:
        models = await get_available_models()
    except Exception as exc:  # noqa: BLE001 - startup should never abort on this
        logger.warning("Could not resolve default model from /v1/models: %s", exc)
        return None

    latest = _pick_latest_sonnet(models)
    if latest:
        constants.RESOLVED_DEFAULT_MODEL = latest
        logger.info("Resolved default model from Anthropic Models API: %s", latest)
        return latest

    logger.info(
        "No Sonnet model found in /v1/models response; using fallback %s",
        DEFAULT_MODEL_FALLBACK,
    )
    return None


@router.get("/v1/models")
async def list_models(
    request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """List available models, preferring Anthropic's live Models API when configured."""
    # Check FastAPI API key if configured
    await verify_api_key(request, credentials)

    models = await get_available_models()
    if not os.getenv("CLAUDE_MODELS_OVERRIDE", "").strip():
        # Added here, not in get_available_models, so "sonnet[1m]" never wins
        # _pick_latest_sonnet
        now = int(datetime.now(timezone.utc).timestamp())
        models = models + [
            {"id": alias, "object": "model", "created": now, "owned_by": "anthropic"}
            for alias in CLAUDE_MODEL_ALIASES
        ]
    return {"object": "list", "data": models}
