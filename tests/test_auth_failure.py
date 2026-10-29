#!/usr/bin/env python3
"""
Unit tests: a logged-out or expired Claude Code CLI gives a clear 401.

The mocked chunks match what claude-agent-sdk 0.2.174 yields with no login:
a synthetic AssistantMessage (error="authentication_failed"), an is_error
ResultMessage, then the ResultError that run_completion turns into an
error_during_execution chunk.
"""

import asyncio
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import claude_relay.auth as auth
import claude_relay.errors as errors
import claude_relay.main as main
from claude_relay import usage as sdk_usage
from claude_relay.claude_cli import ClaudeCodeCLI
from claude_relay.completion import claude_login_error
from tests.test_real_usage import make_chunks, post, stream_events

MODEL = "claude-haiku-4-5-20251001"
CLI_TEXT = "Not logged in · Please run /login"

NOT_LOGGED_IN = [
    {
        "content": [{"type": "text", "text": CLI_TEXT}],
        "model": "<synthetic>",
        "error": "authentication_failed",
    },
    {"subtype": "success", "is_error": True, "result": CLI_TEXT, "api_error_status": None},
    {
        "type": "result",
        "subtype": "error_during_execution",
        "is_error": True,
        "error_message": f"Claude Code returned an error result: {CLI_TEXT} (exit code: 1)",
    },
]


@pytest.fixture(autouse=True)
def clear_last_failure():
    auth.record_auth_failure(None)
    yield
    auth.record_auth_failure(None)


def chat(chunks, **extra):
    payload = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], **extra}
    return post("/v1/chat/completions", payload, chunks)


def auth_status():
    return TestClient(main.app, base_url="http://localhost").get("/v1/auth/status").json()


class TestResultError:
    def test_not_logged_in(self):
        assert sdk_usage.result_error(NOT_LOGGED_IN)["status"] == 401
        assert sdk_usage.result_error(NOT_LOGGED_IN)["auth_failure"] == "not_logged_in"

    @pytest.mark.parametrize(
        "text,kind",
        [
            ("Login expired · Please run /login", "login_expired"),
            ("OAuth token revoked · Please run /login", "token_revoked"),
            ("Invalid API key · Fix external API key", "invalid_api_key"),
            (
                'API Error: 401 {"type":"error","error":{"type":"authentication_error"}}',
                "authentication_error",
            ),
        ],
    )
    def test_cli_exception_markers(self, text, kind):
        chunks = [{"subtype": "error_during_execution", "is_error": True, "error_message": text}]
        assert sdk_usage.result_error(chunks) == {
            "status": 401,
            "message": text,
            "auth_failure": kind,
        }

    def test_structured_signals(self):
        flagged = [{"error": "authentication_failed"}, {"subtype": "success", "is_error": True}]
        assert sdk_usage.result_error(flagged)["auth_failure"] == "authentication_failed"
        status_401 = [{"subtype": "success", "is_error": True, "api_error_status": 401}]
        assert sdk_usage.result_error(status_401)["status"] == 401

    def test_other_errors_unchanged(self):
        chunks = [{"subtype": "success", "is_error": True, "result": "boom"}]
        assert sdk_usage.result_error(chunks) == {"status": 502, "message": "boom"}

    def test_login_text_in_a_successful_answer_is_not_an_error(self):
        assert sdk_usage.result_error(make_chunks(text=CLI_TEXT)) is None


class TestEndpoints:
    def test_openai_401(self):
        response = chat(NOT_LOGGED_IN)
        assert response.status_code == 401
        error = response.json()["error"]
        assert error["type"] == "authentication_error"
        assert error["code"] == "claude_cli_not_authenticated"
        assert "claude auth login" in error["message"]

    def test_anthropic_401(self):
        payload = {
            "model": MODEL,
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
        }
        response = post("/v1/messages", payload, NOT_LOGGED_IN)
        assert response.status_code == 401
        body = response.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "authentication_error"
        assert "claude auth login" in body["error"]["message"]

    def test_streaming_ends_with_error_event(self):
        response = chat(NOT_LOGGED_IN, stream=True)
        assert response.status_code == 200
        events = stream_events(response)
        assert events == [errors.openai_error(claude_login_error("not_logged_in"))]
        assert CLI_TEXT not in response.text

    def test_last_failure_recorded_then_cleared(self):
        assert auth_status()["last_auth_failure"] is None
        chat(NOT_LOGGED_IN)
        failure = auth_status()["last_auth_failure"]
        assert failure["kind"] == "not_logged_in"
        assert isinstance(failure["timestamp"], int)
        assert chat(make_chunks()).status_code == 200
        assert auth_status()["last_auth_failure"] is None


class TestOAuthTokenPassThrough:
    def test_token_only_counts_as_cli_auth(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("CLAUDE_AUTH_METHOD", raising=False)
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-test")
        for method in (None, "cli"):
            if method:
                monkeypatch.setenv("CLAUDE_AUTH_METHOD", method)
            manager = auth.ClaudeCodeAuthManager()
            assert manager.auth_method == "claude_cli"
            assert manager.auth_status["valid"] is True
            assert manager.auth_status["config"]["oauth_token_present"] is True

    def test_env_reaches_cli_subprocess(self, monkeypatch, tmp_path):
        """Run the real SDK transport up to the process spawn and inspect its env."""
        from claude_agent_sdk._internal.transport import subprocess_cli

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("CLAUDE_AUTH_METHOD", raising=False)
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat01-test")
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://proxy.example:8080")
        monkeypatch.setenv("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", "1")
        captured = {}

        async def fake_open_process(cmd, **kwargs):
            captured.update(kwargs["env"])
            raise RuntimeError("spawn intercepted")

        monkeypatch.setattr(subprocess_cli.anyio, "open_process", fake_open_process)
        with patch.object(auth, "auth_manager", auth.ClaudeCodeAuthManager()):
            cli = ClaudeCodeCLI(cwd=str(tmp_path))

        async def run():
            return [c async for c in cli.run_completion(prompt="hi", stream=False)]

        chunks = asyncio.run(run())
        assert chunks[-1]["is_error"] is True
        assert captured["CLAUDE_CODE_OAUTH_TOKEN"] == "sk-ant-oat01-test"
        assert captured["ANTHROPIC_BASE_URL"] == "http://proxy.example:8080"
