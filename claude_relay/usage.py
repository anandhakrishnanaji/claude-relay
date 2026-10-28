"""Read real token usage and stop reasons from Claude Agent SDK messages.

claude_cli.run_completion turns SDK message objects into dicts, so a
ResultMessage arrives as a dict with ``subtype``, ``is_error``, ``usage``,
``stop_reason`` and friends, and an AssistantMessage as a dict with a
``content`` list (and its own ``usage``/``stop_reason``).
"""

from typing import Any, Dict, List, Optional

# Anthropic stop reasons the response model accepts; anything else becomes end_turn.
ANTHROPIC_STOP_REASONS = {
    "end_turn",
    "max_tokens",
    "stop_sequence",
    "tool_use",
    "pause_turn",
    "refusal",
}

# tool_use maps to "stop" because no tool_calls are returned to OpenAI clients.
_OPENAI_FINISH_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "tool_use": "stop",
    "pause_turn": "stop",
    "max_tokens": "length",
    "refusal": "content_filter",
}


def find_result_message(chunks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the last ResultMessage-like dict, or None."""
    for chunk in reversed(chunks):
        if chunk.get("type") == "result" or ("subtype" in chunk and "is_error" in chunk):
            return chunk
    return None


def _last_assistant(chunks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for chunk in reversed(chunks):
        if isinstance(chunk.get("content"), list) or chunk.get("type") == "assistant":
            return chunk
    return None


def extract_usage(chunks: List[Dict[str, Any]]) -> Optional[Dict[str, int]]:
    """Token counts from the ResultMessage, or None if the SDK reported none."""
    result = find_result_message(chunks)
    usage = result.get("usage") if result else None
    if not isinstance(usage, dict) or "input_tokens" not in usage:
        return None
    return {
        key: int(usage.get(key) or 0)
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        )
    }


def extract_stop_reason(chunks: List[Dict[str, Any]]) -> Optional[str]:
    """The SDK's native stop reason (ResultMessage first, then the last AssistantMessage)."""
    result = find_result_message(chunks)
    if result and result.get("stop_reason"):
        return result["stop_reason"]
    assistant = _last_assistant(chunks)
    if assistant:
        return assistant.get("stop_reason") or (assistant.get("message") or {}).get("stop_reason")
    return None


def extract_model(chunks: List[Dict[str, Any]], default: str) -> str:
    """The model that actually ran, or ``default`` if the SDK didn't say.

    Prefers AssistantMessage.model, then the ResultMessage model_usage entry with
    the most output, then the init SystemMessage (which echoes unknown names).
    """
    for chunk in reversed(chunks):
        model = chunk.get("model")
        if isinstance(chunk.get("content"), list) and model and model != "<synthetic>":
            return model
    for chunk in reversed(chunks):
        model_usage = chunk.get("model_usage")
        if isinstance(model_usage, dict) and model_usage:
            return max(model_usage, key=lambda m: (model_usage[m] or {}).get("outputTokens", 0))
    for chunk in chunks:
        data = chunk.get("data")
        if chunk.get("subtype") == "init" and isinstance(data, dict) and data.get("model"):
            return data["model"]
    return default


def openai_finish_reason(chunks: List[Dict[str, Any]]) -> str:
    return _OPENAI_FINISH_REASONS.get(extract_stop_reason(chunks), "stop")


def anthropic_stop_reason(chunks: List[Dict[str, Any]]) -> str:
    stop_reason = extract_stop_reason(chunks)
    return stop_reason if stop_reason in ANTHROPIC_STOP_REASONS else "end_turn"


def openai_usage_counts(
    chunks: List[Dict[str, Any]], prompt_estimate: int, completion_estimate: int
) -> Dict[str, int]:
    """OpenAI prompt/completion/total tokens; falls back to the given estimates."""
    usage = extract_usage(chunks)
    if usage is None:
        prompt_tokens, completion_tokens = prompt_estimate, completion_estimate
    else:
        prompt_tokens = (
            usage["input_tokens"]
            + usage["cache_read_input_tokens"]
            + usage["cache_creation_input_tokens"]
        )
        completion_tokens = usage["output_tokens"]
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


# Login failure texts the bundled Claude Code CLI reports (its own UI strings,
# plus the API's 401 error type), each mapped to a short failure kind.
_AUTH_FAILURE_MARKERS = (
    ("Not logged in", "not_logged_in"),
    ("Login expired", "login_expired"),
    ("OAuth session expired", "login_expired"),
    ("OAuth token revoked", "token_revoked"),
    ("Invalid API key", "invalid_api_key"),
    ("Invalid auth token", "invalid_auth_token"),
    ("Please run /login", "not_logged_in"),
    ("authentication_error", "authentication_error"),
)


def auth_failure_kind(chunks: List[Dict[str, Any]], message: str) -> Optional[str]:
    """The kind of Claude Code login failure behind an error message, or None."""
    for marker, kind in _AUTH_FAILURE_MARKERS:
        if marker in message:
            return kind
    for chunk in chunks:
        # The SDK sets error="authentication_failed" on the CLI's synthetic AssistantMessage
        if chunk.get("error") == "authentication_failed":
            return "authentication_failed"
        if chunk.get("api_error_status") == 401:
            return "authentication_error"
    return None


def result_error(chunks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Describe an error ResultMessage as {"status": int, "message": str}, else None.

    A Claude Code login failure gets status 401 and an "auth_failure" kind, and an
    unknown model gets "code": "model_not_found" and status 400.

    error_max_turns is not treated as an error here: callers keep returning
    whatever partial assistant text was produced, as before.
    """
    result = find_result_message(chunks)
    if not result or not result.get("is_error") or result.get("subtype") == "error_max_turns":
        return None
    message = (
        result.get("error_message")
        or "; ".join(result.get("errors") or [])
        or result.get("result")
        or f"Claude Code returned an error ({result.get('subtype')})"
    )
    message = str(message)
    kind = auth_failure_kind(chunks, message)
    if kind:
        return {"status": 401, "message": message, "auth_failure": kind}
    # The CLI flags an unknown or inaccessible --model on its synthetic AssistantMessage
    if any(chunk.get("error") == "model_not_found" for chunk in chunks):
        return {"status": 400, "message": message, "code": "model_not_found"}
    status = 429 if result.get("api_error_status") == 429 else 502
    return {"status": status, "message": message}
