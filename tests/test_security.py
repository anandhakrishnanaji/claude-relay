#!/usr/bin/env python3
"""
Tests for the browser-facing defences: CORS, the Host allow-list and the
server-side ALLOW_TOOLS gate on enable_tools.

The CORS and Host tests build a small app with add_security_middleware() so
they don't depend on CORS_ORIGINS/ALLOWED_HOSTS in the environment or a .env.
"""

from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import claude_relay.main as main
from claude_relay.constants import (
    CLAUDE_TOOLS,
    DEFAULT_ALLOWED_HOSTS,
    DEFAULT_ALLOWED_TOOLS,
    parse_allowed_hosts,
    parse_cors_origins,
)

EVIL_ORIGIN = "https://evil.example"


def make_app(cors_origins=None, allowed_hosts=None):
    """Build a minimal app with the same security middleware as main.app."""
    app = FastAPI()

    @app.get("/ping")
    async def ping():
        return {"ok": True}

    @app.post("/ping")
    async def ping_post():
        return {"ok": True}

    main.add_security_middleware(
        app,
        parse_cors_origins(cors_origins),
        parse_allowed_hosts(allowed_hosts),
    )
    return app


def preflight(client, origin=EVIL_ORIGIN):
    return client.options(
        "/ping",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )


class TestCORS:
    def test_default_allows_no_origins(self):
        assert parse_cors_origins(None) == []
        assert parse_cors_origins("") == []

    def test_evil_origin_preflight_rejected_by_default(self):
        client = TestClient(make_app(), base_url="http://localhost")
        response = preflight(client)
        assert response.status_code == 400
        assert "access-control-allow-origin" not in response.headers

    def test_evil_origin_simple_request_gets_no_allow_origin(self):
        client = TestClient(make_app(), base_url="http://localhost")
        response = client.post("/ping", headers={"Origin": EVIL_ORIGIN})
        assert response.status_code == 200
        assert "access-control-allow-origin" not in response.headers

    def test_configured_origin_allowed_with_credentials(self):
        app = make_app(cors_origins='["https://app.example"]')
        client = TestClient(app, base_url="http://localhost")
        response = preflight(client, origin="https://app.example")
        assert response.headers["access-control-allow-origin"] == "https://app.example"
        assert response.headers["access-control-allow-credentials"] == "true"
        assert "access-control-allow-origin" not in preflight(client).headers

    def test_wildcard_never_sends_credentials(self):
        client = TestClient(make_app(cors_origins='["*"]'), base_url="http://localhost")
        response = preflight(client)
        assert response.headers["access-control-allow-origin"] == "*"
        assert "access-control-allow-credentials" not in response.headers


class TestTrustedHost:
    def test_default_hosts(self):
        assert parse_allowed_hosts(None) == DEFAULT_ALLOWED_HOSTS
        assert parse_allowed_hosts(" a.example , *.ts.net ,") == ["a.example", "*.ts.net"]

    @pytest.mark.parametrize(
        "host",
        [
            "localhost",
            "localhost:8787",
            "claude.localhost",
            "claude.localhost:8787",
            "127.0.0.1:8000",
            "[::1]:8787",
            "[::1]",
        ],
    )
    def test_local_hosts_pass(self, host):
        client = TestClient(make_app())
        response = client.get("/ping", headers={"Host": host})
        assert response.status_code == 200

    @pytest.mark.parametrize(
        "host", ["evil.example", "evil.example:8787", "localhost.evil.example", "[::2]:8787"]
    )
    def test_bad_host_rejected(self, host):
        client = TestClient(make_app())
        response = client.get("/ping", headers={"Host": host})
        assert response.status_code == 400

    def test_custom_wildcard_host(self):
        client = TestClient(make_app(allowed_hosts="localhost,*.ts.net"))
        assert client.get("/ping", headers={"Host": "claude.tail1234.ts.net"}).status_code == 200
        assert client.get("/ping", headers={"Host": "claude.localhost"}).status_code == 400

    def test_star_disables_check(self):
        client = TestClient(make_app(allowed_hosts="*"))
        assert client.get("/ping", headers={"Host": "anything.example"}).status_code == 200

    def test_main_app_rejects_bad_host(self):
        client = TestClient(main.app)
        response = client.get("/health", headers={"Host": "evil.example"})
        assert response.status_code == 400


class TestToolGate:
    """enable_tools is refused unless the server sets ALLOW_TOOLS."""

    CHAT = {
        "model": "claude-haiku-4-5-20251001",
        "messages": [{"role": "user", "content": "hi"}],
        "enable_tools": True,
    }
    MESSAGES = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "hi"}],
        "enable_tools": True,
    }

    def _post(self, path, payload, allow_tools, headers=None):
        captured = {}

        async def fake_run_completion(**kwargs):
            captured.update(kwargs)
            yield {"type": "assistant", "content": [{"type": "text", "text": "ok"}]}

        with (
            patch.object(main, "validate_claude_code_auth", return_value=(True, {})),
            patch.object(main.claude_cli, "run_completion", fake_run_completion),
            patch.object(main.claude_cli, "parse_claude_message", return_value="ok"),
            patch.object(main.constants, "ALLOW_TOOLS", allow_tools),
        ):
            client = TestClient(main.app, base_url="http://localhost")
            response = client.post(path, json=payload, headers=headers)
        return response, captured

    def test_default_is_off(self):
        with patch.dict("os.environ", {}, clear=True), patch("dotenv.load_dotenv"):
            import importlib
            import claude_relay.constants as constants

            try:
                assert importlib.reload(constants).ALLOW_TOOLS is False
            finally:
                importlib.reload(constants)

    def test_chat_completions_refused(self):
        response, opts = self._post("/v1/chat/completions", self.CHAT, allow_tools=False)
        assert response.status_code == 403
        error = response.json()["error"]
        assert error["type"] == "permission_error"
        assert error["code"] == "tools_disabled"
        assert "ALLOW_TOOLS" in error["message"]
        assert opts == {}

    def test_chat_completions_streaming_refused(self):
        payload = {**self.CHAT, "stream": True}
        response, opts = self._post("/v1/chat/completions", payload, allow_tools=False)
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "tools_disabled"
        assert opts == {}

    def test_messages_refused(self):
        response, opts = self._post("/v1/messages", self.MESSAGES, allow_tools=False)
        assert response.status_code == 403
        body = response.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "permission_error"
        assert "ALLOW_TOOLS" in body["error"]["message"]
        assert opts == {}

    def test_no_tools_unaffected(self):
        payload = {k: v for k, v in self.CHAT.items() if k != "enable_tools"}
        response, opts = self._post("/v1/chat/completions", payload, allow_tools=False)
        assert response.status_code == 200
        assert opts["disallowed_tools"] == CLAUDE_TOOLS

    @pytest.mark.parametrize("stream", [False, True])
    def test_headers_cannot_reenable_tools(self, stream):
        payload = {k: v for k, v in self.CHAT.items() if k != "enable_tools"}
        headers = {
            "X-Claude-Allowed-Tools": "Bash,Write",
            "X-Claude-Permission-Mode": "bypassPermissions",
        }
        response, opts = self._post(
            "/v1/chat/completions", {**payload, "stream": stream}, False, headers
        )
        assert response.status_code == 200
        assert opts["disallowed_tools"] == CLAUDE_TOOLS
        assert opts.get("allowed_tools") is None
        assert opts.get("permission_mode") is None

    def test_chat_completions_allowed(self):
        response, opts = self._post("/v1/chat/completions", self.CHAT, allow_tools=True)
        assert response.status_code == 200
        assert opts["allowed_tools"] == DEFAULT_ALLOWED_TOOLS
        assert opts["permission_mode"] == "bypassPermissions"

    def test_chat_completions_streaming_allowed(self):
        payload = {**self.CHAT, "stream": True}
        response, opts = self._post("/v1/chat/completions", payload, allow_tools=True)
        assert response.status_code == 200
        assert "data: [DONE]" in response.text
        assert opts["permission_mode"] == "bypassPermissions"

    def test_messages_allowed(self):
        response, opts = self._post("/v1/messages", self.MESSAGES, allow_tools=True)
        assert response.status_code == 200
        assert opts["allowed_tools"] == DEFAULT_ALLOWED_TOOLS
        assert opts["permission_mode"] == "bypassPermissions"


class TestSessionEndpointsAuth:
    """The /v1/sessions endpoints need the API key when one is configured."""

    @pytest.mark.parametrize(
        "method,path",
        [
            ("get", "/v1/sessions"),
            ("get", "/v1/sessions/stats"),
            ("get", "/v1/sessions/abc"),
            ("delete", "/v1/sessions/abc"),
        ],
    )
    def test_requires_api_key(self, method, path):
        import claude_relay.auth as auth

        main.session_manager.get_or_create_session("abc")
        try:
            with patch.object(auth.auth_manager, "env_api_key", "secret-key"):
                client = TestClient(main.app, base_url="http://localhost")
                assert getattr(client, method)(path).status_code == 401
                ok = getattr(client, method)(path, headers={"Authorization": "Bearer secret-key"})
                assert ok.status_code == 200
        finally:
            main.session_manager.delete_session("abc")


class TestDiagnosticEndpoints:
    """/v1/auth/status needs the API key."""

    @pytest.mark.parametrize("method,path", [("get", "/v1/auth/status")])
    def test_requires_api_key(self, method, path):
        import claude_relay.auth as auth

        with patch.object(auth.auth_manager, "env_api_key", "secret-key"):
            client = TestClient(main.app, base_url="http://localhost")
            assert getattr(client, method)(path).status_code == 401
            ok = getattr(client, method)(path, headers={"Authorization": "Bearer secret-key"})
            assert ok.status_code == 200


class TestLogRedaction:
    def test_debug_log_masks_authorization(self, caplog):
        with patch.object(main, "DEBUG_MODE", True), caplog.at_level("DEBUG"):
            TestClient(main.app, base_url="http://localhost").get(
                "/health", headers={"Authorization": "Bearer tok-123"}
            )
        assert any("Headers:" in r.getMessage() for r in caplog.records)
        assert "tok-123" not in caplog.text

    def test_validation_error_log_omits_input(self, caplog):
        body = {"model": "m", "messages": "my-secret-prompt"}
        with caplog.at_level("ERROR"):
            response = TestClient(main.app, base_url="http://localhost").post(
                "/v1/chat/completions", json=body
            )
        assert response.status_code == 422
        assert "Validation errors" in caplog.text
        assert "my-secret-prompt" not in caplog.text


class TestRequestSizeLimit:
    """MAX_REQUEST_SIZE holds for chunked bodies too, and a bad Content-Length is a 400."""

    def _client(self):
        return TestClient(main.app, base_url="http://localhost")

    def test_chunked_body_over_limit(self):
        def body():
            for _ in range(11):
                yield b"x" * (1024 * 1024)

        response = self._client().post(
            "/v1/chat/completions", content=body(), headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 413
        assert response.json()["error"]["type"] == "request_too_large"

    def test_content_length_over_limit(self):
        response = self._client().post(
            "/v1/chat/completions", content=b"x" * (main.MAX_REQUEST_SIZE + 1)
        )
        assert response.status_code == 413

    def test_bad_content_length(self):
        response = self._client().post(
            "/v1/chat/completions", content=b"{}", headers={"Content-Length": "abc"}
        )
        assert response.status_code == 400
        assert response.json()["error"]["message"] == "Invalid Content-Length header"

    def test_normal_body_passes(self):
        from tests.test_real_usage import chat, make_chunks

        response = chat(make_chunks(text="ok"))
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == "ok"
