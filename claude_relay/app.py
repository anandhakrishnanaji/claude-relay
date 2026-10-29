"""The FastAPI app: startup checks, routers, error handlers and middleware."""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from dotenv import load_dotenv

from claude_relay.auth import validate_claude_code_auth
from claude_relay.completion import claude_cli
from claude_relay.errors import add_error_handlers
from claude_relay.middleware import (
    DebugLoggingMiddleware,
    RequestIDMiddleware,
    RequestSizeLimitMiddleware,
    add_security_middleware,
)
from claude_relay.routers.chat import router as chat_router
from claude_relay.routers.messages import router as messages_router
from claude_relay.routers.models import resolve_default_model
from claude_relay.routers.models import router as models_router
from claude_relay.routers.responses import router as responses_router
from claude_relay.routers.sessions import router as sessions_router
from claude_relay.routers.system import router as system_router
from claude_relay.session_manager import session_manager
from claude_relay.rate_limiter import limiter, rate_limit_exceeded_handler

from claude_relay import __version__, constants
from claude_relay.constants import (
    ALLOWED_HOSTS,
    CORS_ORIGINS,
    DEBUG_MODE,
    MAX_REQUEST_SIZE,
    VERBOSE,
)

# Load environment variables
load_dotenv()

# Set logging level based on debug/verbose mode
log_level = logging.DEBUG if (DEBUG_MODE or VERBOSE) else logging.INFO
logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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
app.include_router(chat_router)
app.include_router(messages_router)
app.include_router(models_router)
app.include_router(system_router)
app.include_router(sessions_router)
add_error_handlers(app)


add_security_middleware(app, CORS_ORIGINS, ALLOWED_HOSTS)

# Add rate limiting error handler
if limiter:
    app.state.limiter = limiter
    app.add_exception_handler(429, rate_limit_exceeded_handler)

# Add security middleware (order matters - first added = last executed)
app.add_middleware(RequestIDMiddleware)
app.add_middleware(RequestSizeLimitMiddleware, max_size=MAX_REQUEST_SIZE)
app.add_middleware(DebugLoggingMiddleware)
