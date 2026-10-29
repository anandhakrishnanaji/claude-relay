"""
Pytest configuration and fixtures for claude-relay tests.
"""

import sys

import pytest


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Clear rate-limit counters so in-process endpoint tests don't hit 429s.

    Uses the app's limiter rather than claude_relay.rate_limiter.limiter, which
    test_rate_limiter_unit replaces by reloading the module.
    """
    app_module = sys.modules.get("claude_relay.app")
    limiter = getattr(app_module.app.state, "limiter", None) if app_module else None
    if limiter:
        limiter.reset()
    yield
