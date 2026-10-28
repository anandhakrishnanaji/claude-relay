"""
Pytest configuration and fixtures for claude-relay tests.
"""

import sys

import pytest
import requests


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    """Clear rate-limit counters so in-process endpoint tests don't hit 429s.

    Uses the app's limiter rather than src.rate_limiter.limiter, which
    test_rate_limiter_unit replaces by reloading the module.
    """
    main = sys.modules.get("src.main")
    limiter = getattr(main.app.state, "limiter", None) if main else None
    if limiter:
        limiter.reset()
    yield


# Check if server is running for integration tests
def is_server_running(base_url: str = "http://localhost:8000") -> bool:
    """Check if the test server is running."""
    try:
        response = requests.get(f"{base_url}/health", timeout=2)
        return response.status_code == 200
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return False


# Marker for tests that require a running server
requires_server = pytest.mark.skipif(
    not is_server_running(),
    reason="Server not running at localhost:8000. Start with: poetry run python main.py",
)
