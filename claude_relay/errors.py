"""Error bodies in OpenAI or Anthropic format, and the app's exception handlers.

Paths under /v1/messages get Anthropic-style errors; everything else gets OpenAI-style.
"""

import logging
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from claude_relay import constants
from claude_relay.completion import CompletionError

logger = logging.getLogger(__name__)


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
    if constants.DEBUG_MODE or constants.VERBOSE:
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


async def completion_error_handler(request: Request, exc: CompletionError):
    """Render a failed request in the calling endpoint's native error format."""
    if is_anthropic_path(request.url.path):
        content = anthropic_error(exc.status, exc.message)
    else:
        content = openai_error(exc)
    return JSONResponse(status_code=exc.status, content=content)


async def http_exception_handler(request: Request, exc: HTTPException):
    """Format HTTP exceptions as OpenAI- or Anthropic-style errors."""
    if is_anthropic_path(request.url.path):
        content = anthropic_error(exc.status_code, exc.detail)
    else:
        content = _openai_error(exc.status_code, exc.detail)
    return JSONResponse(status_code=exc.status_code, content=content)


def add_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(CompletionError, completion_error_handler)
    app.add_exception_handler(HTTPException, http_exception_handler)
