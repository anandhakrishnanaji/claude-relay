"""
Constants and configuration for Claude Code OpenAI Wrapper.

Single source of truth for tool names, models, and other configuration values.

Usage Examples:
    # Check if a model is supported
    from claude_relay.constants import CLAUDE_MODELS
    if model_name in CLAUDE_MODELS:
        # proceed with request

    # Get default allowed tools
    from claude_relay.constants import DEFAULT_ALLOWED_TOOLS
    options = {"allowed_tools": DEFAULT_ALLOWED_TOOLS}

    # Use rate limits in FastAPI
    from claude_relay.constants import RATE_LIMIT_CHAT
    @limiter.limit(f"{RATE_LIMIT_CHAT}/minute")
    async def chat_endpoint(): ...

Note:
    - Tool configurations are managed by ToolManager (see tool_manager.py)
    - Model validation rejects only names that are clearly not Claude models
    - Rate limits can be overridden via environment variables
"""

import json
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv

# Load .env before reading any env-derived constant below; this module is
# imported before main.py gets to its own load_dotenv().
load_dotenv()

# Claude Agent SDK Tool Names
# These are the built-in tools available in the Claude Agent SDK
# See: https://docs.anthropic.com/en/docs/claude-code/sdk
CLAUDE_TOOLS = [
    "Task",  # Launch agents for complex tasks
    "Bash",  # Execute bash commands
    "Glob",  # File pattern matching
    "Grep",  # Search file contents
    "Read",  # Read files
    "Edit",  # Edit files
    "Write",  # Write files
    "NotebookEdit",  # Edit Jupyter notebooks
    "WebFetch",  # Fetch web content
    "TodoWrite",  # Manage todo lists
    "WebSearch",  # Search the web
    "BashOutput",  # Get bash output
    "KillShell",  # Kill bash shells
    "Skill",  # Execute skills
    "SlashCommand",  # Execute slash commands
]

# Default tools to allow when tools are enabled
# Subset of CLAUDE_TOOLS that are safe and commonly used
DEFAULT_ALLOWED_TOOLS = [
    "Read",
    "Glob",
    "Grep",
    "Bash",
    "Write",
    "Edit",
]

# Tools to disallow by default (potentially dangerous or slow)
DEFAULT_DISALLOWED_TOOLS = [
    "Task",  # Can spawn sub-agents
    "WebFetch",  # External network access
    "WebSearch",  # External network access
]

# Claude Models
# Static fallback models exposed by /v1/models and accepted by validation when
# the live Anthropic Models API is unavailable or not configured.
# NOTE: Claude Agent SDK only supports Claude 4+ models, not Claude 3.x.
#
# Operators can override the advertised model list without rebuilding the image:
#   CLAUDE_MODELS_OVERRIDE=claude-sonnet-4-6,claude-opus-4-6
# First-party IDs from the model table in the bundled Claude Code CLI (2.1.310).
DEFAULT_CLAUDE_MODELS = [
    # Claude 5.x
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5-5",
    "claude-sonnet-5",
    "claude-haiku-5-5",
    "claude-fable-5-1",
    "claude-fable-5",
    # Claude 4.x
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-opus-4-5-20251101",
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
    "claude-opus-4-1-20250805",
]

# Aliases the CLI resolves itself to its current model of each family
# (e.g. sonnet -> latest Sonnet). Also listed by /v1/models.
CLAUDE_MODEL_ALIASES = [
    "sonnet",
    "opus",
    "haiku",
    "fable",
    "best",
    "opusplan",
    "sonnet[1m]",
    "opus[1m]",
    "fable[1m]",
]

_models_override = os.getenv("CLAUDE_MODELS_OVERRIDE", "").strip()
CLAUDE_MODELS = (
    [model.strip() for model in _models_override.split(",") if model.strip()]
    if _models_override
    else DEFAULT_CLAUDE_MODELS
)

# Default model (recommended for most use cases)
# DEFAULT_MODEL_ENV is the explicit operator override; when unset, the wrapper
# resolves the latest Sonnet from Anthropic's live Models API at startup and
# stores it in RESOLVED_DEFAULT_MODEL. DEFAULT_MODEL_FALLBACK is used until/if
# that resolution succeeds; it is the alias the CLI maps to its latest Sonnet.
DEFAULT_MODEL_ENV: Optional[str] = os.getenv("DEFAULT_MODEL")
DEFAULT_MODEL_FALLBACK = "sonnet"
DEFAULT_MODEL = DEFAULT_MODEL_ENV or DEFAULT_MODEL_FALLBACK
RESOLVED_DEFAULT_MODEL: Optional[str] = None


def parse_model_aliases(value: Optional[str]) -> Dict[str, str]:
    """Parse MODEL_ALIASES ("gpt-4o=sonnet,gpt-4o-mini=haiku") into a dict."""
    pairs = (item.split("=", 1) for item in (value or "").split(",") if "=" in item)
    return {
        name.strip(): target.strip() for name, target in pairs if name.strip() and target.strip()
    }


# Request model names to rewrite before calling the SDK, e.g. gpt-4o=sonnet,
# for clients that hard-code OpenAI model names.
MODEL_ALIASES = parse_model_aliases(os.getenv("MODEL_ALIASES"))

# Anthropic Models API configuration for dynamically refreshing /v1/models
ANTHROPIC_MODELS_URL = os.getenv("ANTHROPIC_MODELS_URL", "https://api.anthropic.com/v1/models")
ANTHROPIC_VERSION = os.getenv("ANTHROPIC_VERSION", "2023-06-01")
MODEL_LIST_CACHE_TTL_SECONDS = int(os.getenv("MODEL_LIST_CACHE_TTL_SECONDS", "3600"))
# Shorter TTL applied when the live fetch fails so a transient blip doesn't
# suppress live discovery for a full hour.
MODEL_LIST_ERROR_TTL_SECONDS = int(os.getenv("MODEL_LIST_ERROR_TTL_SECONDS", "60"))
MODEL_LIST_REQUEST_TIMEOUT_SECONDS = float(os.getenv("MODEL_LIST_REQUEST_TIMEOUT_SECONDS", "5"))

# Filesystem settings the CLI loads per request (comma-separated: user, project, local).
# Empty by default so the host's ~/.claude/CLAUDE.md, settings, skills and plugins
# don't leak into API requests.
CLAUDE_SETTING_SOURCES = [
    s.strip() for s in os.getenv("CLAUDE_SETTING_SOURCES", "").split(",") if s.strip()
]


def parse_cors_origins(value: Optional[str]) -> List[str]:
    """Parse CORS_ORIGINS (a JSON list). Unset or empty means no origins."""
    return json.loads(value) if value and value.strip() else []


def parse_allowed_hosts(value: Optional[str]) -> List[str]:
    """Parse ALLOWED_HOSTS (comma-separated). Unset or empty means the local defaults."""
    hosts = [h.strip() for h in (value or "").split(",") if h.strip()]
    return hosts or DEFAULT_ALLOWED_HOSTS


# Browser origins allowed to call the API cross-origin (JSON list). Empty by
# default: without an API key, any web page the user visits could otherwise
# call the local server and read the reply.
CORS_ORIGINS = parse_cors_origins(os.getenv("CORS_ORIGINS"))

# Host header allow-list (comma-separated, wildcards like *.ts.net). Blocks DNS
# rebinding, which CORS can't. ALLOWED_HOSTS=* disables the check.
DEFAULT_ALLOWED_HOSTS = ["localhost", "127.0.0.1", "::1", "[::1]", "*.localhost"]
ALLOWED_HOSTS = parse_allowed_hosts(os.getenv("ALLOWED_HOSTS"))

# Honour "enable_tools": true in requests. Off by default: tools run Bash, Write
# and Edit with permissions bypassed, so a request is enough to run code here.
ALLOW_TOOLS = os.getenv("ALLOW_TOOLS", "false").lower() in ("true", "1", "yes", "on")

# System Prompt Types
SYSTEM_PROMPT_TYPE_TEXT = "text"
SYSTEM_PROMPT_TYPE_PRESET = "preset"

# System Prompt Presets
SYSTEM_PROMPT_PRESET_CLAUDE_CODE = "claude_code"

# API Configuration
DEFAULT_MAX_TURNS = 10
DEFAULT_TIMEOUT_MS = 600000  # 10 minutes
DEFAULT_PORT = 8000

# Session Management
SESSION_CLEANUP_INTERVAL_MINUTES = 5
SESSION_MAX_AGE_MINUTES = 60

# Rate Limiting (requests per minute)
RATE_LIMIT_DEFAULT = 60
RATE_LIMIT_CHAT = 30
RATE_LIMIT_MODELS = 100
RATE_LIMIT_HEALTH = 200
