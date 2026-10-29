"""
Parameter validation and mapping utilities for OpenAI to Claude Code SDK conversion.
"""

import logging
from typing import Dict, Any, Optional
from claude_relay import constants
from claude_relay.constants import CLAUDE_MODEL_ALIASES, CLAUDE_MODELS

logger = logging.getLogger(__name__)


class ParameterValidator:
    """Validates and maps OpenAI Chat Completions parameters to Claude Code SDK options."""

    # Use models from constants (single source of truth)
    SUPPORTED_MODELS = set(CLAUDE_MODELS)

    @classmethod
    def validate_model(cls, model: str) -> bool:
        """False only for names that are clearly not Claude models.

        Unlisted Claude names are allowed so new models work before the static
        list catches up; the CLI rejects real typos (reported as a 400).
        """
        if (
            model in cls.SUPPORTED_MODELS
            or model.lower() in CLAUDE_MODEL_ALIASES
            # also Bedrock/Vertex IDs such as us.anthropic.claude-... or claude-...@date
            or "claude" in model.lower()
        ):
            return True
        logger.warning(f"Rejected model '{model}': not a Claude model or alias")
        return False

    @classmethod
    def resolve_model(cls, model: str) -> Optional[str]:
        """Apply MODEL_ALIASES; return the model to run, or None to reject it."""
        if model in constants.MODEL_ALIASES:
            return constants.MODEL_ALIASES[model]
        return model if cls.validate_model(model) else None

    @staticmethod
    def thinking_from_budget(budget: Optional[int]) -> Optional[Dict[str, Any]]:
        """X-Claude-Max-Thinking-Tokens as an SDK thinking config; 0 turns thinking off."""
        if budget is None:
            return None
        if budget <= 0:
            return {"type": "disabled"}
        return {"type": "enabled", "budget_tokens": budget}

    @classmethod
    def extract_claude_headers(cls, headers: Dict[str, str]) -> Dict[str, Any]:
        """
        Extract Claude-Code-specific parameters from custom HTTP headers.

        This allows clients to pass SDK-specific options via headers:
        - X-Claude-Max-Turns: 5
        - X-Claude-Disallowed-Tools: Bash,Write
        - X-Claude-Max-Thinking-Tokens: 4096 (0 disables thinking)
        - X-Claude-Effort: low|medium|high|xhigh|max
        - X-Claude-Fallback-Model: haiku
        """
        claude_options = {}

        # Extract max_turns
        if "x-claude-max-turns" in headers:
            try:
                claude_options["max_turns"] = int(headers["x-claude-max-turns"])
            except ValueError:
                logger.warning(
                    f"Invalid X-Claude-Max-Turns header: {headers['x-claude-max-turns']}"
                )

        # Extract disallowed tools (only narrows the enable_tools set; tools off overrides it)
        if "x-claude-disallowed-tools" in headers:
            tools = [tool.strip() for tool in headers["x-claude-disallowed-tools"].split(",")]
            if tools:
                claude_options["disallowed_tools"] = tools

        # Extract max thinking tokens
        if "x-claude-max-thinking-tokens" in headers:
            try:
                claude_options["max_thinking_tokens"] = int(headers["x-claude-max-thinking-tokens"])
            except ValueError:
                logger.warning(
                    f"Invalid X-Claude-Max-Thinking-Tokens header: {headers['x-claude-max-thinking-tokens']}"
                )

        if "x-claude-effort" in headers:
            effort = headers["x-claude-effort"].strip().lower()
            if effort in constants.EFFORT_LEVELS:
                claude_options["effort"] = effort
            else:
                logger.warning(f"Invalid X-Claude-Effort header: {headers['x-claude-effort']}")

        if "x-claude-fallback-model" in headers:
            fallback = cls.resolve_model(headers["x-claude-fallback-model"].strip())
            if fallback:
                claude_options["fallback_model"] = fallback

        return claude_options
