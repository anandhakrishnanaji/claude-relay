"""Health, version, auth status and the landing page."""

import os
from importlib.resources import files
from string import Template
from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials

from claude_relay import __version__
from claude_relay.auth import get_claude_code_auth_info, security, verify_api_key
from claude_relay.rate_limiter import rate_limit_endpoint

router = APIRouter()

# $version, $status_color, $status_text and $auth_method are filled per request
LANDING_PAGE = Template(files("claude_relay").joinpath("static/index.html").read_text("utf-8"))


@router.get("/health")
@rate_limit_endpoint("health")
async def health_check(request: Request):
    """Health check endpoint."""
    return {"status": "healthy", "service": "claude-code-openai-wrapper"}


@router.get("/version")
@rate_limit_endpoint("health")
async def version_info(request: Request):
    """Version information endpoint."""
    return {
        "version": __version__,
        "service": "claude-code-openai-wrapper",
        "api_version": "v1",
    }


@router.get("/", response_class=HTMLResponse)
async def root():
    """Landing page with API documentation."""
    auth_info = get_claude_code_auth_info()
    auth_method = auth_info.get("method", "unknown")
    auth_valid = auth_info.get("status", {}).get("valid", False)
    status_color = "#22c55e" if auth_valid else "#ef4444"
    status_text = "Connected" if auth_valid else "Not Connected"

    html_content = LANDING_PAGE.substitute(
        version=__version__,
        status_color=status_color,
        status_text=status_text,
        auth_method=auth_method,
    )
    return HTMLResponse(content=html_content)


@router.get("/v1/auth/status")
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
