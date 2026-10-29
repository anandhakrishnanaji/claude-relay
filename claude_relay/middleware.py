"""ASGI middleware: Host allow-list and CORS, request IDs, body size limit, debug logging."""

import asyncio
import logging
import uuid
from typing import Dict, List

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.responses import PlainTextResponse

from claude_relay import constants
from claude_relay.errors import anthropic_error, is_anthropic_path

logger = logging.getLogger(__name__)


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


SECRET_HEADERS = {"authorization", "x-api-key", "cookie", "proxy-authorization"}


def redact_headers(headers) -> Dict[str, str]:
    """Copy request headers with credential values masked, for logs and debug output."""
    return {k: "[REDACTED]" if k.lower() in SECRET_HEADERS else v for k, v in headers.items()}


class DebugLoggingMiddleware(BaseHTTPMiddleware):
    """ASGI-compliant middleware for logging request/response details when debug mode is enabled."""

    async def dispatch(self, request: Request, call_next):
        # Get request ID for correlation
        request_id = getattr(request.state, "request_id", "unknown")

        if not (constants.DEBUG_MODE or constants.VERBOSE):
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
