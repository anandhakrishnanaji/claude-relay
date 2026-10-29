import os
from typing import Optional
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
from fastapi.responses import JSONResponse


def get_rate_limit_key(request: Request) -> str:
    """Get the rate limiting key (IP address) from the request."""
    return get_remote_address(request)


def create_rate_limiter() -> Optional[Limiter]:
    """Create and configure the rate limiter based on environment variables."""
    rate_limit_enabled = os.getenv("RATE_LIMIT_ENABLED", "true").lower() in (
        "true",
        "1",
        "yes",
        "on",
    )

    if not rate_limit_enabled:
        return None

    # Create limiter with IP-based identification
    limiter = Limiter(
        key_func=get_rate_limit_key, default_limits=[]  # We'll apply limits per endpoint
    )

    return limiter


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Custom rate limit exceeded handler that returns JSON error response."""
    # Calculate retry after based on rate limit window (default 60 seconds)
    retry_after = 60
    message = f"Rate limit exceeded. Try again in {retry_after} seconds."
    if request.url.path.startswith("/v1/messages"):
        content = {"type": "error", "error": {"type": "rate_limit_error", "message": message}}
    else:
        content = {
            "error": {
                "message": message,
                "type": "rate_limit_exceeded",
                "code": "too_many_requests",
                "retry_after": retry_after,
            }
        }
    return JSONResponse(status_code=429, content=content, headers={"Retry-After": str(retry_after)})


# endpoint -> (environment variable, default requests per minute)
_RATE_LIMITS = {
    "chat": ("RATE_LIMIT_CHAT_PER_MINUTE", 10),
    "auth": ("RATE_LIMIT_AUTH_PER_MINUTE", 10),
    "health": ("RATE_LIMIT_HEALTH_PER_MINUTE", 30),
}


def get_rate_limit_for_endpoint(endpoint: str) -> str:
    """Rate limit string for an endpoint group, from its environment variable."""
    env_var, default = _RATE_LIMITS[endpoint]
    return f"{int(os.getenv(env_var, default))}/minute"


def rate_limit_endpoint(endpoint: str):
    """Decorator factory for applying rate limits to endpoints."""

    def decorator(func):
        if limiter:
            return limiter.limit(get_rate_limit_for_endpoint(endpoint))(func)
        return func

    return decorator


# Create the global limiter instance
limiter = create_rate_limiter()
