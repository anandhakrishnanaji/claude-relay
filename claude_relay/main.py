import os
import json
import asyncio
import logging
import time
import uuid
from typing import Optional, AsyncIterator, Dict, Any, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import Headers
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import PlainTextResponse
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.exceptions import RequestValidationError
import httpx
from dotenv import load_dotenv

from claude_relay.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    Choice,
    Message,
    Usage,
    StreamChoice,
    SessionListResponse,
    # Anthropic API compatible models
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicTextBlock,
    AnthropicUsage,
)
from claude_relay import usage as sdk_usage
from claude_relay.auth import (
    verify_api_key,
    security,
    validate_claude_code_auth,
    get_claude_code_auth_info,
)
from claude_relay.completion import CompletionError, Run, claude_cli, prepare
from claude_relay.responses_api import router as responses_router
from claude_relay.session_manager import session_manager
from claude_relay.rate_limiter import (
    limiter,
    rate_limit_exceeded_handler,
    rate_limit_endpoint,
)
from datetime import datetime, timezone

from claude_relay import __version__, constants
from claude_relay.constants import (
    ALLOWED_HOSTS,
    ANTHROPIC_MODELS_URL,
    ANTHROPIC_VERSION,
    CORS_ORIGINS,
    CLAUDE_MODELS,
    CLAUDE_MODEL_ALIASES,
    DEFAULT_MODEL_FALLBACK,
    MODEL_LIST_CACHE_TTL_SECONDS,
    MODEL_LIST_ERROR_TTL_SECONDS,
    MODEL_LIST_REQUEST_TIMEOUT_SECONDS,
)

# Load environment variables
load_dotenv()

# Configure logging based on debug mode
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() in ("true", "1", "yes", "on")
VERBOSE = os.getenv("VERBOSE", "false").lower() in ("true", "1", "yes", "on")

# Set logging level based on debug/verbose mode
log_level = logging.DEBUG if (DEBUG_MODE or VERBOSE) else logging.INFO
logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Verify Claude Code authentication and CLI on startup."""
    logger.info("Verifying Claude Code authentication and CLI...")

    # Validate authentication first
    auth_valid, auth_info = validate_claude_code_auth()

    if not auth_valid:
        logger.error("❌ Claude Code authentication failed!")
        for error in auth_info.get("errors", []):
            logger.error(f"  - {error}")
        logger.warning("Authentication setup guide:")
        logger.warning("  1. For Anthropic API: Set ANTHROPIC_API_KEY")
        logger.warning("  2. For Bedrock: Set CLAUDE_CODE_USE_BEDROCK=1 + AWS credentials")
        logger.warning("  3. For Vertex AI: Set CLAUDE_CODE_USE_VERTEX=1 + GCP credentials")
    else:
        logger.info(f"✅ Claude Code authentication validated: {auth_info['method']}")

    # `claude auth status`: checks the CLI runs and is logged in, without a model call.
    # Only the CLI login uses it; API key, Bedrock and Vertex auth would fail it.
    if auth_info.get("method") == "claude_cli":
        if await claude_cli.verify_cli():
            logger.info("✅ Claude Code CLI found and logged in")
        else:
            logger.warning(
                "⚠️  Claude Code CLI check failed; the server will start, but requests may fail"
            )

    # Log debug information if debug mode is enabled
    if DEBUG_MODE or VERBOSE:
        logger.debug("🔧 Debug mode enabled - Enhanced logging active")
        logger.debug("🔧 Environment variables:")
        logger.debug(f"   DEBUG_MODE: {DEBUG_MODE}")
        logger.debug(f"   VERBOSE: {VERBOSE}")
        logger.debug(f"   PORT: {os.getenv('PORT', '8000')}")
        logger.debug(f"   CORS_ORIGINS: {CORS_ORIGINS}")
        logger.debug(f"   ALLOWED_HOSTS: {','.join(ALLOWED_HOSTS)}")
        logger.debug(f"   ALLOW_TOOLS: {constants.ALLOW_TOOLS}")
        logger.debug(f"   MAX_TIMEOUT: {os.getenv('MAX_TIMEOUT', '600000')}")
        logger.debug(f"   CLAUDE_CWD: {os.getenv('CLAUDE_CWD', 'Not set')}")
        logger.debug("🔧 Available endpoints:")
        logger.debug("   POST /v1/chat/completions - Main chat endpoint")
        logger.debug("   GET  /v1/models - List available models")
        logger.debug("   GET  /v1/auth/status - Authentication status")
        logger.debug("   GET  /health - Health check")
        logger.debug(f"🔧 API Key protection: {'Enabled' if os.getenv('API_KEY') else 'Disabled'}")

    # Resolve the default model from the live Anthropic Models API so /v1/chat
    # uses the latest Sonnet without a code change. Best-effort: any failure
    # leaves the static fallback in place.
    try:
        await resolve_default_model()
    except Exception as e:
        logger.warning(f"Default model resolution skipped: {e}")

    # Start session cleanup task
    session_manager.start_cleanup_task()

    yield

    # Cleanup on shutdown
    logger.info("Shutting down session manager...")
    session_manager.shutdown()


# Create FastAPI app
app = FastAPI(
    title="claude-relay",
    description="Claude Code API server with OpenAI- and Anthropic-compatible endpoints",
    version=__version__,
    lifespan=lifespan,
)
app.include_router(responses_router)


class HostAllowListMiddleware(TrustedHostMiddleware):
    """TrustedHostMiddleware that also handles IPv6 literals such as "[::1]:8787".

    Starlette strips the port with split(":")[0], which turns any IPv6 Host into "[".
    """

    async def __call__(self, scope, receive, send):
        if self.allow_any or scope["type"] not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        host = Headers(scope=scope).get("host", "")
        if host.startswith("["):
            host = host[: host.find("]") + 1]  # "[::1]:8787" -> "[::1]"
        else:
            host = host.split(":")[0]

        for pattern in self.allowed_hosts:
            if host == pattern or (pattern.startswith("*") and host.endswith(pattern[1:])):
                await self.app(scope, receive, send)
                return
        await PlainTextResponse("Invalid host header", status_code=400)(scope, receive, send)


def add_security_middleware(app: FastAPI, cors_origins: List[str], allowed_hosts: List[str]):
    """Add CORS and Host allow-list middleware.

    The Host check runs first (outermost) and guards against DNS rebinding,
    which CORS alone can't stop.
    """
    # Credentials with a wildcard would make Starlette echo any Origin back
    allow_credentials = "*" not in cors_origins
    if not allow_credentials:
        logger.warning(
            "CORS_ORIGINS contains '*': any website can call this server from a browser; "
            "credentials disabled"
        )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    if "*" in allowed_hosts:
        logger.warning("ALLOWED_HOSTS contains '*': Host header check disabled")
    app.add_middleware(HostAllowListMiddleware, allowed_hosts=allowed_hosts)


add_security_middleware(app, CORS_ORIGINS, ALLOWED_HOSTS)

# Add rate limiting error handler
if limiter:
    app.state.limiter = limiter
    app.add_exception_handler(429, rate_limit_exceeded_handler)

# Security configuration
MAX_REQUEST_SIZE = int(os.getenv("MAX_REQUEST_SIZE", str(10 * 1024 * 1024)))  # 10MB default

# Add middleware
from starlette.middleware.base import BaseHTTPMiddleware


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to each request for audit trails."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class RequestSizeLimitMiddleware:
    """Reject request bodies over max_size bytes, with or without a Content-Length.

    Pure ASGI, so it counts the bytes of a chunked body as they arrive.
    """

    def __init__(self, app, max_size: int):
        self.app = app
        self.max_size = max_size

    @staticmethod
    def _error(scope, status: int, error_type: str, message: str) -> JSONResponse:
        if is_anthropic_path(scope["path"]):
            content = anthropic_error(status, message)
        else:
            content = {"error": {"message": message, "type": error_type, "code": status}}
        return JSONResponse(status_code=status, content=content)

    def _too_large(self, scope) -> JSONResponse:
        message = f"Request body too large. Maximum size is {self.max_size} bytes."
        return self._error(scope, 413, "request_too_large", message)

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        content_length = Headers(scope=scope).get("content-length")
        if content_length is not None:
            if not content_length.isdigit():
                error = self._error(
                    scope, 400, "invalid_request_error", "Invalid Content-Length header"
                )
                await error(scope, receive, send)
                return
            if int(content_length) > self.max_size:
                await self._too_large(scope)(scope, receive, send)
                return

        received = 0
        response_started = False
        rejected = False

        async def limited_receive():
            nonlocal received, rejected
            if rejected:
                return {"type": "http.disconnect"}
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_size:
                    rejected = True
                    if not response_started:
                        await self._too_large(scope)(scope, receive, send)
                    # The app sees a disconnect and stops; its own reply is dropped below
                    return {"type": "http.disconnect"}
            return message

        async def guarded_send(message):
            nonlocal response_started
            if rejected:
                return
            response_started = response_started or message["type"] == "http.response.start"
            await send(message)

        await self.app(scope, limited_receive, guarded_send)


# Add security middleware (order matters - first added = last executed)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(RequestSizeLimitMiddleware, max_size=MAX_REQUEST_SIZE)


SECRET_HEADERS = {"authorization", "x-api-key", "cookie", "proxy-authorization"}


def redact_headers(headers) -> Dict[str, str]:
    """Copy request headers with credential values masked, for logs and debug output."""
    return {k: "[REDACTED]" if k.lower() in SECRET_HEADERS else v for k, v in headers.items()}


class DebugLoggingMiddleware(BaseHTTPMiddleware):
    """ASGI-compliant middleware for logging request/response details when debug mode is enabled."""

    async def dispatch(self, request: Request, call_next):
        # Get request ID for correlation
        request_id = getattr(request.state, "request_id", "unknown")

        if not (DEBUG_MODE or VERBOSE):
            return await call_next(request)

        # Log request details
        start_time = asyncio.get_event_loop().time()

        # Log basic request info with request ID for correlation
        logger.debug(f"🔍 [{request_id}] Incoming request: {request.method} {request.url}")
        logger.debug(f"🔍 [{request_id}] Headers: {redact_headers(request.headers)}")

        # For POST requests, try to log body (but don't break if we can't)
        body_logged = False
        if request.method == "POST" and request.url.path.startswith("/v1/"):
            try:
                # Only attempt to read body if it's reasonable size and content-type
                content_length = request.headers.get("content-length")
                if content_length and int(content_length) < 100000:  # Less than 100KB
                    body = await request.body()
                    if body:
                        try:
                            import json as json_lib

                            parsed_body = json_lib.loads(body.decode())
                            logger.debug(
                                f"🔍 Request body: {json_lib.dumps(parsed_body, indent=2)}"
                            )
                            body_logged = True
                        except:
                            logger.debug(f"🔍 Request body (raw): {body.decode()[:500]}...")
                            body_logged = True
            except Exception as e:
                logger.debug(f"🔍 Could not read request body: {e}")

        if not body_logged and request.method == "POST":
            logger.debug("🔍 Request body: [not logged - streaming or large payload]")

        # Process the request
        try:
            response = await call_next(request)

            # Log response details
            end_time = asyncio.get_event_loop().time()
            duration = (end_time - start_time) * 1000  # Convert to milliseconds

            logger.debug(f"🔍 Response: {response.status_code} in {duration:.2f}ms")

            return response

        except Exception as e:
            end_time = asyncio.get_event_loop().time()
            duration = (end_time - start_time) * 1000

            logger.debug(f"🔍 Request failed after {duration:.2f}ms: {e}")
            raise


# Add the debug middleware
app.add_middleware(DebugLoggingMiddleware)


# Custom exception handler for 422 validation errors
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors with detailed debugging information."""

    # Log the validation error details
    logger.error(f"❌ Request validation failed for {request.method} {request.url}")
    # Leave out "input": it can hold a whole prompt, and this logs outside debug mode too
    logger.error(
        f"❌ Validation errors: {[{k: v for k, v in e.items() if k != 'input'} for e in exc.errors()]}"
    )

    if is_anthropic_path(request.url.path):
        message = "; ".join(
            f"{'.'.join(str(loc) for loc in error.get('loc', []))}: {error.get('msg')}"
            for error in exc.errors()
        )
        return JSONResponse(status_code=422, content=anthropic_error(422, message))

    # Create detailed error response
    error_details = []
    for error in exc.errors():
        location = " -> ".join(str(loc) for loc in error.get("loc", []))
        error_details.append(
            {
                "field": location,
                "message": error.get("msg", "Unknown validation error"),
                "type": error.get("type", "validation_error"),
                "input": error.get("input"),
            }
        )

    # If debug mode is enabled, include the raw request body
    debug_info = {}
    if DEBUG_MODE or VERBOSE:
        try:
            body = await request.body()
            if body:
                debug_info["raw_request_body"] = body.decode()
        except:
            debug_info["raw_request_body"] = "Could not read request body"

    error_response = {
        "error": {
            "message": "Request validation failed - the request body doesn't match the expected format",
            "type": "validation_error",
            "code": "invalid_request_error",
            "details": error_details,
            "help": {
                "common_issues": [
                    "Missing required fields (model, messages)",
                    "Invalid field types (e.g. messages should be an array)",
                    "Invalid role values (must be 'system', 'user', or 'assistant')",
                    "Invalid parameter ranges (e.g. temperature must be 0-2)",
                ],
                "debug_tip": "Set DEBUG_MODE=true or VERBOSE=true environment variable for more detailed logging",
            },
        }
    }

    # Add debug info if available
    if debug_info:
        error_response["error"]["debug"] = debug_info

    return JSONResponse(status_code=422, content=error_response)


def _openai_error(status: int, message: Any) -> Dict[str, Any]:
    return {"error": {"message": message, "type": "api_error", "code": str(status)}}


# CompletionError kinds with their own OpenAI error type; the rest are api_error
_OPENAI_ERROR_TYPES = {
    "claude_cli_not_authenticated": "authentication_error",
    "model_not_found": "invalid_request_error",
    "tools_disabled": "permission_error",
}

ANTHROPIC_ERROR_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    413: "request_too_large",
    422: "invalid_request_error",
    429: "rate_limit_error",
    529: "overloaded_error",
}


def openai_error(exc: CompletionError) -> Dict[str, Any]:
    if exc.kind not in _OPENAI_ERROR_TYPES:
        return _openai_error(exc.status, exc.message)
    error = {"message": exc.message, "type": _OPENAI_ERROR_TYPES[exc.kind], "code": exc.kind}
    if exc.kind == "model_not_found":
        error["param"] = "model"
    return {"error": error}


def anthropic_error(status: int, message: Any) -> Dict[str, Any]:
    error_type = ANTHROPIC_ERROR_TYPES.get(status, "api_error")
    return {"type": "error", "error": {"type": error_type, "message": message}}


def is_anthropic_path(path: str) -> bool:
    return path.startswith("/v1/messages")


@app.exception_handler(CompletionError)
async def completion_error_handler(request: Request, exc: CompletionError):
    """Render a failed request in the calling endpoint's native error format."""
    if is_anthropic_path(request.url.path):
        content = anthropic_error(exc.status, exc.message)
    else:
        content = openai_error(exc)
    return JSONResponse(status_code=exc.status, content=content)


def sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def start_stream(events: AsyncIterator[str]) -> StreamingResponse:
    """Stream SSE events, but wait for the first one so an unknown model is a plain 400."""
    first = await events.__anext__()  # a CompletionError here becomes a normal HTTP error

    async def all_events():
        yield first
        async for event in events:
            yield event

    return StreamingResponse(
        all_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


async def chat_stream(run: Run, request_id: str, include_usage: bool) -> AsyncIterator[str]:
    """OpenAI chat.completion.chunk events for one run."""

    def chunk(delta: Dict[str, Any], finish_reason=None, usage=None) -> str:
        choice = StreamChoice(index=0, delta=delta, finish_reason=finish_reason)
        event = ChatCompletionStreamResponse(
            id=request_id, model=run.model, choices=[choice], usage=usage
        )
        return f"data: {event.model_dump_json()}\n\n"

    role_sent = False
    try:
        async for item in run.stream():
            if not role_sent:
                yield chunk({"role": "assistant", "content": ""})
                role_sent = True
            if isinstance(item, str):
                yield chunk({"content": item})
            else:
                completion = item
    except CompletionError as e:
        if e.kind == "model_not_found" and not role_sent:
            raise
        # The 200 status line is already sent, so report it as a final error event
        yield f"data: {json.dumps(openai_error(e))}\n\n"
        return

    usage = Usage(**completion.openai_usage()) if include_usage else None
    yield chunk({}, sdk_usage.openai_finish_reason(completion.chunks), usage)
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
@rate_limit_endpoint("chat")
async def chat_completions(
    request_body: ChatCompletionRequest,
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """OpenAI-compatible chat completions endpoint."""
    await verify_api_key(request, credentials)

    run = prepare(
        model=request_body.model,
        messages=request_body.messages,
        enable_tools=bool(request_body.enable_tools),
        headers=request.headers,
        effort=request_body.to_claude_options().get("effort"),
        system_suffix=request_body.get_sampling_instructions(),
        session_id=request_body.session_id,
    )
    request_id = f"chatcmpl-{os.urandom(8).hex()}"

    if request_body.stream:
        include_usage = bool(
            request_body.stream_options and request_body.stream_options.include_usage
        )
        return await start_stream(chat_stream(run, request_id, include_usage))

    completion = await run.complete()
    return ChatCompletionResponse(
        id=request_id,
        model=completion.model,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=completion.text),
                finish_reason=sdk_usage.openai_finish_reason(completion.chunks),
            )
        ],
        usage=Usage(**completion.openai_usage()),
    )


async def messages_stream(run: Run) -> AsyncIterator[str]:
    """Anthropic Messages SSE events for one run (shapes from anthropic.types)."""
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    started = False

    def start_events() -> List[str]:
        message = AnthropicMessagesResponse(
            id=message_id,
            model=run.model,
            content=[],
            stop_reason=None,
            usage=AnthropicUsage(input_tokens=0, output_tokens=0),
        )
        return [
            sse("message_start", {"type": "message_start", "message": message.model_dump()}),
            sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
        ]

    try:
        async for item in run.stream():
            if not started:
                for event in start_events():
                    yield event
                started = True
            if isinstance(item, str):
                delta = {"type": "text_delta", "text": item}
                yield sse(
                    "content_block_delta",
                    {"type": "content_block_delta", "index": 0, "delta": delta},
                )
            else:
                completion = item
    except CompletionError as e:
        if e.kind == "model_not_found" and not started:
            raise
        yield sse("error", anthropic_error(e.status, e.message))
        return

    stop_reason = sdk_usage.anthropic_stop_reason(completion.chunks)
    yield sse("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": completion.anthropic_usage(),
        },
    )
    yield sse("message_stop", {"type": "message_stop"})


@app.post("/v1/messages")
@rate_limit_endpoint("chat")
async def anthropic_messages(
    request_body: AnthropicMessagesRequest,
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Anthropic Messages API compatible endpoint."""
    await verify_api_key(request, credentials)

    system = request_body.system_text()
    messages = [Message(role="system", content=system)] if system else []
    run = prepare(
        model=request_body.model,
        messages=messages + request_body.to_openai_messages(),
        enable_tools=bool(request_body.enable_tools),
        thinking=(
            request_body.thinking.model_dump(exclude_none=True) if request_body.thinking else None
        ),
        effort=request_body.output_config.effort if request_body.output_config else None,
    )

    if request_body.stream:
        return await start_stream(messages_stream(run))

    completion = await run.complete()
    return AnthropicMessagesResponse(
        model=completion.model,
        content=[AnthropicTextBlock(text=completion.text)],
        stop_reason=sdk_usage.anthropic_stop_reason(completion.chunks),
        usage=AnthropicUsage(**completion.anthropic_usage()),
    )


@app.get("/v1/models")
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


@app.get("/health")
@rate_limit_endpoint("health")
async def health_check(request: Request):
    """Health check endpoint."""
    return {"status": "healthy", "service": "claude-code-openai-wrapper"}


@app.get("/version")
@rate_limit_endpoint("health")
async def version_info(request: Request):
    """Version information endpoint."""
    return {
        "version": __version__,
        "service": "claude-code-openai-wrapper",
        "api_version": "v1",
    }


@app.get("/", response_class=HTMLResponse)
async def root():
    """Landing page with API documentation."""
    auth_info = get_claude_code_auth_info()
    auth_method = auth_info.get("method", "unknown")
    auth_valid = auth_info.get("status", {}).get("valid", False)
    status_color = "#22c55e" if auth_valid else "#ef4444"
    status_text = "Connected" if auth_valid else "Not Connected"

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en" data-theme="dark">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <meta name="color-scheme" content="light dark">
        <title>claude-relay: Claude Code API server</title>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.min.css">
        <style>
            :root {{
                --pico-font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                --accent-color: #16a34a;
            }}
            /* Light mode colors */
            [data-theme="light"] {{
                --card-bg: #ffffff;
                --subtle-bg: #f1f5f9;
                --border-color: #e2e8f0;
                --page-bg: #f8fafc;
            }}
            /* Dark mode colors */
            [data-theme="dark"] {{
                --card-bg: #1e293b;
                --subtle-bg: #334155;
                --border-color: #475569;
                --page-bg: #0f172a;
            }}
            /* Page background */
            body {{ background: var(--page-bg); }}
            /* GLOBAL FIX: Remove Pico's default code styling everywhere */
            code:not(pre code) {{
                background: transparent !important;
                padding: 0 !important;
                border-radius: 0 !important;
                color: inherit !important;
            }}
            /* Only style code green where we explicitly want it */
            .green-code {{ color: var(--accent-color) !important; }}
            /* Constrain page width - wider for modern screens */
            .container {{
                max-width: 1100px;
                margin: 0 auto;
                padding: 1.5rem 2rem;
            }}
            /* Override Pico article styling */
            article {{
                background: var(--card-bg);
                border: 1px solid var(--border-color);
                border-radius: 0.75rem;
                margin-bottom: 1rem;
                padding: 1rem 1.25rem;
            }}
            article header {{
                padding: 0;
                margin-bottom: 0.75rem;
                background: transparent;
                border: none;
            }}
            /* Section headers with icons - matches status-flex layout */
            .section-header {{
                display: flex;
                align-items: center;
                gap: 0.5rem;
                margin-bottom: 0.75rem;
            }}
            .section-icon {{
                width: 1rem;
                height: 1rem;
                color: var(--accent-color);
                flex-shrink: 0;
            }}
            /* Status indicator */
            .status-dot {{
                width: 0.75rem;
                height: 0.75rem;
                border-radius: 50%;
                display: inline-block;
                animation: pulse 2s infinite;
            }}
            @keyframes pulse {{
                0%, 100% {{ opacity: 1; }}
                50% {{ opacity: 0.5; }}
            }}
            /* Method badges */
            .badge {{
                display: inline-block;
                padding: 0.25rem 0.5rem;
                font-size: 0.7rem;
                font-weight: 700;
                border-radius: 0.25rem;
                text-transform: uppercase;
            }}
            .badge-post {{ background: rgba(34, 197, 94, 0.15); color: #16a34a; }}
            .badge-get {{ background: rgba(59, 130, 246, 0.15); color: #2563eb; }}
            /* Header layout */
            .header-flex {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 1rem;
                margin-bottom: 1rem;
            }}
            .header-left {{
                display: flex;
                align-items: center;
                gap: 1rem;
                flex-shrink: 0;
            }}
            .header-right {{
                display: flex;
                align-items: center;
                gap: 0.75rem;
                flex-shrink: 0;
            }}
            .icon-btn {{
                padding: 0.5rem;
                border-radius: 0.5rem;
                background: var(--subtle-bg);
                border: 1px solid var(--border-color);
                cursor: pointer;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                color: inherit;
            }}
            .icon-btn:hover {{ opacity: 0.8; }}
            .icon-btn svg {{ width: 1.25rem; height: 1.25rem; }}
            .version-badge {{
                padding: 0.25rem 0.75rem;
                background: var(--subtle-bg);
                border: 1px solid var(--border-color);
                border-radius: 0.5rem;
                font-family: monospace;
                font-size: 0.875rem;
            }}
            /* Logo container */
            .logo-container {{
                background: linear-gradient(135deg, #22c55e 0%, #0ea5e9 100%);
                padding: 2px;
                border-radius: 0.75rem;
            }}
            .logo-inner {{
                background: var(--card-bg);
                border-radius: calc(0.75rem - 2px);
                padding: 0.75rem;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            .logo-inner svg {{ width: 2rem; height: 2rem; color: #22c55e; }}
            /* Endpoint list */
            .endpoint-item {{
                display: flex;
                align-items: center;
                gap: 0.75rem;
                padding: 0.5rem 0;
                border-bottom: 1px solid var(--pico-muted-border-color);
            }}
            .endpoint-item:last-child {{ border-bottom: none; }}
            .endpoint-item code {{ flex: 1; }}
            .endpoint-desc {{ color: var(--pico-muted-color); font-size: 0.85rem; }}
            /* Details accordion styling */
            details {{
                border: 1px solid var(--border-color);
                border-radius: 0.5rem;
                margin-bottom: 0.4rem;
                background: var(--subtle-bg);
            }}
            details summary {{
                padding: 0.5rem 0.75rem;
                display: flex;
                align-items: center;
                gap: 0.75rem;
                cursor: pointer;
                list-style: none;
            }}
            details summary::-webkit-details-marker {{ display: none; }}
            details summary::after {{
                content: "";
                margin-left: auto;
                width: 0.5rem;
                height: 0.5rem;
                border-right: 2px solid currentColor;
                border-bottom: 2px solid currentColor;
                transform: rotate(-45deg);
                transition: transform 0.2s;
            }}
            details[open] summary::after {{ transform: rotate(45deg); }}
            details .content {{ padding: 0 1rem 1rem; }}
            details .content pre {{
                margin: 0;
                font-size: 0.875rem;
                overflow-x: auto;
            }}
            /* Config grid */
            .config-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                gap: 0.75rem;
            }}
            .config-item {{
                padding: 0.75rem;
                background: var(--subtle-bg);
                border: 1px solid var(--border-color);
                border-radius: 0.5rem;
            }}
            .config-item code {{ font-weight: 600; }}
            .config-item p {{ margin: 0.25rem 0 0; font-size: 0.875rem; color: var(--pico-muted-color); }}
            /* Footer */
            footer nav {{
                display: flex;
                justify-content: center;
                gap: 2rem;
            }}
            footer a {{
                display: flex;
                align-items: center;
                gap: 0.5rem;
            }}
            footer svg {{ width: 1rem; height: 1rem; }}
            /* Quick start */
            .quickstart-wrapper {{ position: relative; }}
            .copy-btn {{
                position: absolute;
                top: 0.5rem;
                right: 0.5rem;
                padding: 0.5rem;
                background: var(--subtle-bg);
                border: 1px solid var(--border-color);
                border-radius: 0.5rem;
                cursor: pointer;
                z-index: 1;
                color: inherit;
            }}
            .copy-btn:hover {{ opacity: 0.8; }}
            .copy-btn svg {{ width: 1rem; height: 1rem; }}
            .hidden {{ display: none !important; }}
            /* Shiki code styling */
            .shiki {{ padding: 1rem; border-radius: 0.5rem; overflow-x: auto; }}
            .shiki code {{ white-space: pre-wrap; word-break: break-word; }}
            /* Status card layout */
            .status-flex {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                flex-wrap: wrap;
                gap: 1rem;
            }}
            .status-left {{
                display: flex;
                align-items: center;
                gap: 0.75rem;
            }}
            .auth-badge {{
                padding: 0.25rem 0.75rem;
                background: var(--subtle-bg);
                border: 1px solid var(--border-color);
                border-radius: 1rem;
                font-size: 0.875rem;
            }}
        </style>
        <script type="module">
            import {{ codeToHtml }} from 'https://esm.sh/shiki@3.0.0';

            const lightTheme = 'github-light';
            const darkTheme = 'github-dark';

            function isDark() {{
                return document.documentElement.getAttribute('data-theme') === 'dark';
            }}

            async function highlightJson(json, targetId) {{
                const code = typeof json === 'string' ? json : JSON.stringify(json, null, 2);
                const theme = isDark() ? darkTheme : lightTheme;
                try {{
                    const html = await codeToHtml(code, {{ lang: 'json', theme }});
                    document.getElementById(targetId).innerHTML = html;
                }} catch (e) {{
                    document.getElementById(targetId).innerHTML = '<pre style="color:red;">Error: ' + e.message + '</pre>';
                }}
            }}

            // Lazy load data when details opens
            document.querySelectorAll('details[data-endpoint]').forEach(details => {{
                details.addEventListener('toggle', async () => {{
                    if (details.open) {{
                        const id = details.id;
                        const endpoint = details.dataset.endpoint;
                        const dataContainer = document.getElementById('data-' + id);
                        const loader = document.getElementById('loader-' + id);
                        if (dataContainer.innerHTML === '' || dataContainer.dataset.theme !== (isDark() ? 'dark' : 'light')) {{
                            loader.classList.remove('hidden');
                            try {{
                                const response = await fetch(endpoint);
                                const json = await response.json();
                                await highlightJson(json, 'data-' + id);
                                dataContainer.dataset.theme = isDark() ? 'dark' : 'light';
                            }} catch (e) {{
                                dataContainer.innerHTML = '<span style="color:red;">Error: ' + e.message + '</span>';
                            }}
                            loader.classList.add('hidden');
                        }}
                    }}
                }});
            }});

            // Re-highlight on theme change
            window.addEventListener('themeChanged', async () => {{
                await highlightQuickstart();
                document.querySelectorAll('details[open][data-endpoint]').forEach(async details => {{
                    const id = details.id;
                    const endpoint = details.dataset.endpoint;
                    const dataContainer = document.getElementById('data-' + id);
                    if (dataContainer && dataContainer.innerHTML) {{
                        const response = await fetch(endpoint);
                        const json = await response.json();
                        await highlightJson(json, 'data-' + id);
                        dataContainer.dataset.theme = isDark() ? 'dark' : 'light';
                    }}
                }});
            }});

            const quickstartCode = `curl -X POST http://localhost:8000/v1/chat/completions \\\\
  -H "Content-Type: application/json" \\\\
  -d '{{"model": "claude-sonnet-4-5-20250929", "messages": [{{"role": "user", "content": "Hello!"}}]}}'`;

            async function highlightQuickstart() {{
                const theme = isDark() ? darkTheme : lightTheme;
                try {{
                    const html = await codeToHtml(quickstartCode, {{ lang: 'bash', theme }});
                    document.getElementById('quickstart-code').innerHTML = html;
                }} catch (e) {{
                    document.getElementById('quickstart-code').innerHTML = '<pre>' + quickstartCode + '</pre>';
                }}
            }}

            window.highlightQuickstart = highlightQuickstart;
            highlightQuickstart();
        </script>
        <script>
            const quickstartText = 'curl -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d \\'{{"model": "claude-sonnet-4-5-20250929", "messages": [{{"role": "user", "content": "Hello!"}}]}}\\'';

            function copyQuickstart() {{
                if (navigator.clipboard && navigator.clipboard.writeText) {{
                    navigator.clipboard.writeText(quickstartText).then(showCopySuccess).catch(fallbackCopy);
                }} else {{
                    fallbackCopy();
                }}
            }}

            function fallbackCopy() {{
                const textarea = document.createElement('textarea');
                textarea.value = quickstartText;
                textarea.style.position = 'fixed';
                textarea.style.opacity = '0';
                document.body.appendChild(textarea);
                textarea.select();
                try {{ document.execCommand('copy'); showCopySuccess(); }} catch (e) {{ console.error('Copy failed:', e); }}
                document.body.removeChild(textarea);
            }}

            function showCopySuccess() {{
                const copyIcon = document.getElementById('copy-icon');
                const checkIcon = document.getElementById('check-icon');
                copyIcon.classList.add('hidden');
                checkIcon.classList.remove('hidden');
                setTimeout(() => {{
                    copyIcon.classList.remove('hidden');
                    checkIcon.classList.add('hidden');
                }}, 2000);
            }}

            function toggleTheme() {{
                const html = document.documentElement;
                const current = html.getAttribute('data-theme');
                const next = current === 'dark' ? 'light' : 'dark';
                html.setAttribute('data-theme', next);
                localStorage.setItem('theme', next);
                updateThemeIcon(next === 'dark');
                window.dispatchEvent(new Event('themeChanged'));
            }}

            function updateThemeIcon(isDark) {{
                document.getElementById('sun-icon').classList.toggle('hidden', isDark);
                document.getElementById('moon-icon').classList.toggle('hidden', !isDark);
            }}

            document.addEventListener('DOMContentLoaded', () => {{
                const saved = localStorage.getItem('theme');
                if (saved) {{
                    document.documentElement.setAttribute('data-theme', saved);
                    updateThemeIcon(saved === 'dark');
                }} else {{
                    updateThemeIcon(true);
                }}
            }});
        </script>
    </head>
    <body>
        <main class="container">
            <!-- Header -->
            <header class="header-flex">
                <div class="header-left">
                    <div class="logo-container">
                        <div class="logo-inner">
                            <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"/>
                            </svg>
                        </div>
                    </div>
                    <div>
                        <h1 style="margin:0;">claude-relay</h1>
                        <p style="margin:0;color:var(--pico-muted-color);">OpenAI-compatible API for Claude</p>
                    </div>
                </div>
                <div class="header-right">
                    <span class="version-badge">v{__version__}</span>
                    <button onclick="toggleTheme()" class="icon-btn" title="Toggle theme">
                        <svg id="sun-icon" class="hidden" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z"/>
                        </svg>
                        <svg id="moon-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z"/>
                        </svg>
                    </button>
                    <a href="https://github.com/anandhakrishnanaji/claude-relay" target="_blank" rel="noopener noreferrer" class="icon-btn" title="View on GitHub">
                        <svg fill="currentColor" viewBox="0 0 24 24">
                            <path fill-rule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clip-rule="evenodd"/>
                        </svg>
                    </a>
                </div>
            </header>

            <!-- Status Card -->
            <article>
                <div class="status-flex">
                    <div class="status-left">
                        <span class="status-dot" style="background-color: {status_color};"></span>
                        <strong>{status_text}</strong>
                    </div>
                    <span class="auth-badge">Auth: <code class="green-code">{auth_method}</code></span>
                </div>
            </article>

            <!-- Quick Start -->
            <article>
                <div class="section-header">
                    <svg class="section-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"/></svg>
                    <strong>Quick Start</strong>
                </div>
                <div class="quickstart-wrapper">
                    <button onclick="copyQuickstart()" class="copy-btn" title="Copy to clipboard">
                        <svg id="copy-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"/>
                        </svg>
                        <svg id="check-icon" class="hidden" fill="none" stroke="currentColor" viewBox="0 0 24 24" style="color:#22c55e;">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"/>
                        </svg>
                    </button>
                    <div id="quickstart-code"></div>
                </div>
            </article>

            <!-- API Endpoints -->
            <article>
                <div class="section-header">
                    <svg class="section-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"/></svg>
                    <strong>API Endpoints</strong>
                </div>

                <!-- Static POST endpoints -->
                <div class="endpoint-item">
                    <span class="badge badge-post">POST</span>
                    <code>/v1/chat/completions</code>
                    <span class="endpoint-desc">OpenAI-compatible chat</span>
                </div>
                <div class="endpoint-item">
                    <span class="badge badge-post">POST</span>
                    <code>/v1/messages</code>
                    <span class="endpoint-desc">Anthropic-compatible</span>
                </div>

                <!-- Expandable GET endpoints -->
                <details id="models" data-endpoint="/v1/models" name="endpoints">
                    <summary>
                        <span class="badge badge-get">GET</span>
                        <code>/v1/models</code>
                        <span class="endpoint-desc">List models</span>
                    </summary>
                    <div class="content">
                        <small id="loader-models" class="hidden">Loading...</small>
                        <div id="data-models"></div>
                    </div>
                </details>

                <details id="auth" data-endpoint="/v1/auth/status" name="endpoints">
                    <summary>
                        <span class="badge badge-get">GET</span>
                        <code>/v1/auth/status</code>
                        <span class="endpoint-desc">Auth status</span>
                    </summary>
                    <div class="content">
                        <small id="loader-auth" class="hidden">Loading...</small>
                        <div id="data-auth"></div>
                    </div>
                </details>

                <details id="sessions" data-endpoint="/v1/sessions" name="endpoints">
                    <summary>
                        <span class="badge badge-get">GET</span>
                        <code>/v1/sessions</code>
                        <span class="endpoint-desc">Active sessions</span>
                    </summary>
                    <div class="content">
                        <small id="loader-sessions" class="hidden">Loading...</small>
                        <div id="data-sessions"></div>
                    </div>
                </details>

                <details id="health" data-endpoint="/health" name="endpoints">
                    <summary>
                        <span class="badge badge-get">GET</span>
                        <code>/health</code>
                        <span class="endpoint-desc">Health check</span>
                    </summary>
                    <div class="content">
                        <small id="loader-health" class="hidden">Loading...</small>
                        <div id="data-health"></div>
                    </div>
                </details>

                <details id="version" data-endpoint="/version" name="endpoints">
                    <summary>
                        <span class="badge badge-get">GET</span>
                        <code>/version</code>
                        <span class="endpoint-desc">API version</span>
                    </summary>
                    <div class="content">
                        <small id="loader-version" class="hidden">Loading...</small>
                        <div id="data-version"></div>
                    </div>
                </details>
            </article>

            <!-- Configuration -->
            <article>
                <div class="section-header">
                    <svg class="section-icon" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/></svg>
                    <strong>Configuration</strong>
                </div>
                <p>Set <code>CLAUDE_AUTH_METHOD</code> to choose authentication:</p>
                <div class="config-grid">
                    <div class="config-item">
                        <code class="green-code">cli</code>
                        <p>Claude CLI auth</p>
                    </div>
                    <div class="config-item">
                        <code class="green-code">api_key</code>
                        <p>ANTHROPIC_API_KEY</p>
                    </div>
                    <div class="config-item">
                        <code class="green-code">bedrock</code>
                        <p>AWS Bedrock</p>
                    </div>
                    <div class="config-item">
                        <code class="green-code">vertex</code>
                        <p>Google Vertex AI</p>
                    </div>
                </div>
            </article>

            <!-- Footer -->
            <footer>
                <nav>
                    <a href="/docs">
                        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/>
                        </svg>
                        API Docs
                    </a>
                    <a href="/redoc">
                        <svg fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"/>
                        </svg>
                        ReDoc
                    </a>
                </nav>
            </footer>
        </main>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/v1/auth/status")
@rate_limit_endpoint("auth")
async def get_auth_status(
    request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """Get Claude Code authentication status."""
    await verify_api_key(request, credentials)
    from claude_relay.auth import auth_manager, last_auth_failure

    auth_info = get_claude_code_auth_info()
    active_api_key = auth_manager.get_api_key()

    return {
        "claude_code_auth": auth_info,
        "last_auth_failure": last_auth_failure,
        "server_info": {
            "api_key_required": bool(active_api_key),
            "api_key_source": "environment" if os.getenv("API_KEY") else "none",
            "version": __version__,
        },
    }


@app.get("/v1/sessions/stats")
async def get_session_stats(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Get session manager statistics."""
    await verify_api_key(request, credentials)
    stats = session_manager.get_stats()
    return {
        "session_stats": stats,
        "cleanup_interval_minutes": session_manager.cleanup_interval_minutes,
        "default_ttl_hours": session_manager.default_ttl_hours,
    }


@app.get("/v1/sessions")
async def list_sessions(
    request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """List all active sessions."""
    await verify_api_key(request, credentials)
    sessions = session_manager.list_sessions()
    return SessionListResponse(sessions=sessions, total=len(sessions))


@app.get("/v1/sessions/{session_id}")
async def get_session(
    session_id: str,
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Get information about a specific session."""
    await verify_api_key(request, credentials)
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return session.to_session_info()


@app.delete("/v1/sessions/{session_id}")
async def delete_session(
    session_id: str,
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Delete a specific session."""
    await verify_api_key(request, credentials)
    deleted = session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"message": f"Session {session_id} deleted successfully"}


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Format HTTP exceptions as OpenAI- or Anthropic-style errors."""
    if is_anthropic_path(request.url.path):
        content = anthropic_error(exc.status_code, exc.detail)
    else:
        content = _openai_error(exc.status_code, exc.detail)
    return JSONResponse(status_code=exc.status_code, content=content)


if __name__ == "__main__":
    from claude_relay.cli import main

    main()
