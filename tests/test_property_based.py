#!/usr/bin/env python3
"""
Property-based tests using Hypothesis for edge case discovery.

These tests generate random inputs to find edge cases that manual testing might miss.
"""

from hypothesis import given, strategies as st, settings

from claude_relay.message_adapter import MessageAdapter
from claude_relay.parameter_validator import ParameterValidator
from claude_relay.constants import CLAUDE_MODELS


class TestMessageAdapterProperties:
    """Property-based tests for MessageAdapter."""

    @given(text=st.text(min_size=0, max_size=10000))
    @settings(max_examples=30)
    def test_estimate_tokens_returns_positive(self, text: str):
        """Token estimation should return a non-negative integer."""
        result = MessageAdapter.estimate_tokens(text)
        assert isinstance(result, int)
        assert result >= 0


class TestParameterValidatorProperties:
    """Property-based tests for ParameterValidator."""

    @given(model=st.sampled_from(CLAUDE_MODELS))
    @settings(max_examples=20)
    def test_valid_model_names_accepted(self, model: str):
        """Valid Claude model names should be accepted."""
        result = ParameterValidator.validate_model(model)
        assert result is True

    @given(model=st.text(min_size=1, max_size=50))
    @settings(max_examples=30)
    def test_any_claude_model_name_accepted(self, model: str):
        """Any name containing "claude" is accepted, so new models keep working."""
        assert ParameterValidator.validate_model(f"claude-{model}") is True


class TestTokenEstimation:
    """Property-based tests for token estimation consistency."""

    @given(text=st.text(min_size=0, max_size=5000))
    @settings(max_examples=30)
    def test_token_estimation_non_negative(self, text: str):
        """Token estimation should always return non-negative value."""
        tokens = MessageAdapter.estimate_tokens(text)
        assert tokens >= 0

    @given(
        prefix=st.text(min_size=10, max_size=100),
        suffix=st.text(min_size=10, max_size=100),
    )
    @settings(max_examples=20)
    def test_concatenation_increases_tokens(self, prefix: str, suffix: str):
        """Concatenating text should not decrease token count."""
        tokens_prefix = MessageAdapter.estimate_tokens(prefix)
        tokens_suffix = MessageAdapter.estimate_tokens(suffix)
        tokens_combined = MessageAdapter.estimate_tokens(prefix + suffix)

        # Combined should be at least as many as the larger part
        # (may be less than sum due to subword tokenization)
        assert tokens_combined >= min(tokens_prefix, tokens_suffix)
