"""
Parameter validation and mapping utilities for OpenAI to Claude Code SDK conversion.
"""

import logging
from typing import Dict, Any, List, Optional
from claude_relay.models import ChatCompletionRequest
from claude_relay import constants
from claude_relay.constants import CLAUDE_MODEL_ALIASES, CLAUDE_MODELS

logger = logging.getLogger(__name__)


class ParameterValidator:
    """Validates and maps OpenAI Chat Completions parameters to Claude Code SDK options."""

    # Use models from constants (single source of truth)
    SUPPORTED_MODELS = set(CLAUDE_MODELS)

    # Valid permission modes for Claude Code SDK
    VALID_PERMISSION_MODES = {"default", "acceptEdits", "bypassPermissions", "plan"}

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
    def validate_permission_mode(cls, permission_mode: str) -> bool:
        """Validate permission mode parameter."""
        if permission_mode not in cls.VALID_PERMISSION_MODES:
            logger.error(
                f"Invalid permission_mode '{permission_mode}'. Valid options: {cls.VALID_PERMISSION_MODES}"
            )
            return False
        return True

    @classmethod
    def validate_tools(cls, tools: List[str]) -> bool:
        """Validate tool names (basic validation for non-empty strings)."""
        if not all(isinstance(tool, str) and tool.strip() for tool in tools):
            logger.error("All tool names must be non-empty strings")
            return False
        return True

    @classmethod
    def create_enhanced_options(
        cls,
        request: ChatCompletionRequest,
        max_turns: Optional[int] = None,
        allowed_tools: Optional[List[str]] = None,
        disallowed_tools: Optional[List[str]] = None,
        permission_mode: Optional[str] = None,
        max_thinking_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Create enhanced Claude Code SDK options with additional parameters.

        This allows API users to pass Claude-Code-specific parameters that don't
        exist in the OpenAI API through custom headers or environment variables.
        """
        # Start with basic options from request
        options = request.to_claude_options()

        # Add Claude Code SDK specific options
        if max_turns is not None:
            if max_turns < 1 or max_turns > 100:
                logger.warning(f"max_turns={max_turns} is outside recommended range (1-100)")
            options["max_turns"] = max_turns

        if allowed_tools:
            if cls.validate_tools(allowed_tools):
                options["allowed_tools"] = allowed_tools

        if disallowed_tools:
            if cls.validate_tools(disallowed_tools):
                options["disallowed_tools"] = disallowed_tools

        if permission_mode:
            if cls.validate_permission_mode(permission_mode):
                options["permission_mode"] = permission_mode

        if max_thinking_tokens is not None:
            if max_thinking_tokens < 0 or max_thinking_tokens > 50000:
                logger.warning(
                    f"max_thinking_tokens={max_thinking_tokens} is outside recommended range (0-50000)"
                )
            options["max_thinking_tokens"] = max_thinking_tokens

        return options

    @classmethod
    def extract_claude_headers(cls, headers: Dict[str, str]) -> Dict[str, Any]:
        """
        Extract Claude-Code-specific parameters from custom HTTP headers.

        This allows clients to pass SDK-specific options via headers:
        - X-Claude-Max-Turns: 5
        - X-Claude-Allowed-Tools: tool1,tool2,tool3
        - X-Claude-Permission-Mode: acceptEdits
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

        # Extract allowed tools
        if "x-claude-allowed-tools" in headers:
            tools = [tool.strip() for tool in headers["x-claude-allowed-tools"].split(",")]
            if tools:
                claude_options["allowed_tools"] = tools

        # Extract disallowed tools
        if "x-claude-disallowed-tools" in headers:
            tools = [tool.strip() for tool in headers["x-claude-disallowed-tools"].split(",")]
            if tools:
                claude_options["disallowed_tools"] = tools

        # Extract permission mode
        if "x-claude-permission-mode" in headers:
            claude_options["permission_mode"] = headers["x-claude-permission-mode"]

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


class CompatibilityReporter:
    """Reports on OpenAI API compatibility and suggests alternatives."""

    @classmethod
    def generate_compatibility_report(cls, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Generate a detailed compatibility report for the request."""
        report = {
            "supported_parameters": [],
            "unsupported_parameters": [],
            "warnings": [],
            "suggestions": [],
        }

        # Check supported parameters
        if request.model:
            report["supported_parameters"].append("model")
        if request.messages:
            report["supported_parameters"].append("messages")
        if request.stream is not None:
            report["supported_parameters"].append("stream")
        if request.user:
            report["supported_parameters"].append("user (for logging)")

        # Check unsupported parameters with suggestions
        if request.temperature != 1.0:
            report["unsupported_parameters"].append("temperature")
            report["suggestions"].append(
                "Claude Code SDK does not support temperature control. Consider using different models for varied response styles (e.g., claude-haiku-4-5 for more focused responses)."
            )

        if request.top_p != 1.0:
            report["unsupported_parameters"].append("top_p")
            report["suggestions"].append(
                "Claude Code SDK does not support top_p. This parameter will be ignored."
            )

        if request.max_tokens:
            report["unsupported_parameters"].append("max_tokens")
            report["suggestions"].append(
                "Use max_turns parameter instead to limit conversation length, or use max_thinking_tokens to limit internal reasoning."
            )

        if request.n > 1:
            report["unsupported_parameters"].append("n")
            report["suggestions"].append(
                "Claude Code SDK only supports single responses (n=1). For multiple variations, make separate API calls."
            )

        if request.stop:
            report["unsupported_parameters"].append("stop")
            report["suggestions"].append(
                "Stop sequences are not supported. Consider post-processing responses or using max_turns to limit output."
            )

        if request.presence_penalty != 0 or request.frequency_penalty != 0:
            report["unsupported_parameters"].extend(["presence_penalty", "frequency_penalty"])
            report["suggestions"].append(
                "Penalty parameters are not supported. Consider using different system prompts to encourage varied responses."
            )

        if request.logit_bias:
            report["unsupported_parameters"].append("logit_bias")
            report["suggestions"].append(
                "Logit bias is not supported. Consider using system prompts to guide response style."
            )

        return report
