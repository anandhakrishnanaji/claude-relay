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
    main = sys.modules.get("claude_relay.main")
    limiter = getattr(main.app.state, "limiter", None) if main else None
    if limiter:
        limiter.reset()
    yield
