#!/usr/bin/env python3
"""
Unit tests for claude_relay/parameter_validator.py

Tests the ParameterValidator class.
These are pure unit tests that don't require a running server.
"""

from unittest.mock import patch

from claude_relay.parameter_validator import ParameterValidator


class TestParameterValidatorValidateModel:
    """Test ParameterValidator.validate_model()"""

    def test_valid_model_returns_true(self):
        """Known supported model returns True."""
        result = ParameterValidator.validate_model("claude-sonnet-4-5-20250929")
        assert result is True

    def test_unlisted_claude_model_returns_true(self):
        """Unlisted claude-* names pass so new models work; the CLI rejects typos."""
        assert ParameterValidator.validate_model("claude-sonnet-9") is True
        assert ParameterValidator.validate_model("us.anthropic.claude-opus-4-8") is True

    def test_aliases_return_true(self):
        for alias in ("sonnet", "Opus", "haiku", "opusplan", "sonnet[1m]"):
            assert ParameterValidator.validate_model(alias) is True

    def test_non_claude_model_returns_false_with_warning(self):
        """Names that are clearly not Claude are rejected with a warning."""
        with patch("claude_relay.parameter_validator.logger") as mock_logger:
            result = ParameterValidator.validate_model("gpt-4o")
            assert result is False
            mock_logger.warning.assert_called_once()
            assert "gpt-4o" in str(mock_logger.warning.call_args)

    def test_resolve_model_applies_aliases(self):
        with patch("claude_relay.constants.MODEL_ALIASES", {"gpt-4o": "haiku"}):
            assert ParameterValidator.resolve_model("gpt-4o") == "haiku"
            assert ParameterValidator.resolve_model("claude-opus-5") == "claude-opus-5"
            assert ParameterValidator.resolve_model("gpt-4o-mini") is None

    def test_all_known_models_valid(self):
        """All models in SUPPORTED_MODELS are valid."""
        for model in ParameterValidator.SUPPORTED_MODELS:
            assert ParameterValidator.validate_model(model) is True


class TestParameterValidatorExtractClaudeHeaders:
    """Test ParameterValidator.extract_claude_headers()"""

    def test_empty_headers_returns_empty_dict(self):
        """Empty headers dict returns empty options dict."""
        result = ParameterValidator.extract_claude_headers({})
        assert result == {}

    def test_extracts_max_turns(self):
        """X-Claude-Max-Turns header is extracted correctly."""
        headers = {"x-claude-max-turns": "10"}
        result = ParameterValidator.extract_claude_headers(headers)
        assert result.get("max_turns") == 10

    def test_invalid_max_turns_logs_warning(self):
        """Invalid X-Claude-Max-Turns logs warning and is ignored."""
        headers = {"x-claude-max-turns": "not-a-number"}
        with patch("claude_relay.parameter_validator.logger") as mock_logger:
            result = ParameterValidator.extract_claude_headers(headers)
            assert "max_turns" not in result
            mock_logger.warning.assert_called_once()

    def test_ignores_removed_tool_headers(self):
        """X-Claude-Allowed-Tools and X-Claude-Permission-Mode are no longer read."""
        headers = {"x-claude-allowed-tools": "Bash", "x-claude-permission-mode": "plan"}
        assert ParameterValidator.extract_claude_headers(headers) == {}

    def test_extracts_disallowed_tools(self):
        """X-Claude-Disallowed-Tools header is extracted with whitespace stripped."""
        headers = {"x-claude-disallowed-tools": " Edit , Delete "}
        result = ParameterValidator.extract_claude_headers(headers)
        assert result.get("disallowed_tools") == ["Edit", "Delete"]

    def test_extracts_max_thinking_tokens(self):
        """X-Claude-Max-Thinking-Tokens header is extracted correctly."""
        headers = {"x-claude-max-thinking-tokens": "5000"}
        result = ParameterValidator.extract_claude_headers(headers)
        assert result.get("max_thinking_tokens") == 5000

    def test_invalid_max_thinking_tokens_logs_warning(self):
        """Invalid X-Claude-Max-Thinking-Tokens logs warning."""
        headers = {"x-claude-max-thinking-tokens": "invalid"}
        with patch("claude_relay.parameter_validator.logger") as mock_logger:
            result = ParameterValidator.extract_claude_headers(headers)
            assert "max_thinking_tokens" not in result
            mock_logger.warning.assert_called_once()

    def test_extracts_multiple_headers(self):
        """Multiple Claude headers are all extracted."""
        headers = {
            "x-claude-max-turns": "5",
            "x-claude-disallowed-tools": "Write",
            "x-claude-max-thinking-tokens": "3000",
        }
        result = ParameterValidator.extract_claude_headers(headers)
        assert result.get("max_turns") == 5
        assert result.get("disallowed_tools") == ["Write"]
        assert result.get("max_thinking_tokens") == 3000
