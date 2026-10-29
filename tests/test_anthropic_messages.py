#!/usr/bin/env python3
"""
Tests for the Anthropic Messages API compatible endpoint (/v1/messages).

This endpoint provides compatibility with the native Anthropic SDK,
enabling tools like VC to use this wrapper via custom base URL configuration.
"""

from contextlib import ExitStack
from unittest.mock import patch

import pytest


class TestAnthropicMessagesModels:
    """Test Anthropic API model classes."""

    def test_anthropic_text_block(self):
        """Test AnthropicTextBlock model."""
        from claude_relay.models import AnthropicTextBlock

        block = AnthropicTextBlock(text="Hello world")
        assert block.type == "text"
        assert block.text == "Hello world"

    def test_anthropic_message(self):
        """Test AnthropicMessage model."""
        from claude_relay.models import AnthropicMessage

        # String content
        msg = AnthropicMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

        # List content
        from claude_relay.models import AnthropicTextBlock

        msg2 = AnthropicMessage(role="assistant", content=[AnthropicTextBlock(text="Hi there")])
        assert msg2.role == "assistant"
        assert len(msg2.content) == 1

    def test_anthropic_messages_request(self):
        """Test AnthropicMessagesRequest model."""
        from claude_relay.models import AnthropicMessagesRequest, AnthropicMessage

        request = AnthropicMessagesRequest(
            model="claude-sonnet-4-5-20250929",
            messages=[AnthropicMessage(role="user", content="Hello")],
            max_tokens=100,
            system="You are helpful",
        )

        assert request.model == "claude-sonnet-4-5-20250929"
        assert len(request.messages) == 1
        assert request.max_tokens == 100
        assert request.system == "You are helpful"

    def test_anthropic_messages_request_to_openai(self):
        """Test conversion from Anthropic to OpenAI message format."""
        from claude_relay.models import AnthropicMessagesRequest, AnthropicMessage

        request = AnthropicMessagesRequest(
            model="claude-sonnet-4-5-20250929",
            messages=[
                AnthropicMessage(role="user", content="Hello"),
                AnthropicMessage(role="assistant", content="Hi there"),
                AnthropicMessage(role="user", content="How are you?"),
            ],
        )

        openai_messages = request.to_openai_messages()
        assert len(openai_messages) == 3
        assert openai_messages[0].role == "user"
        assert openai_messages[0].content == "Hello"
        assert openai_messages[1].role == "assistant"
        assert openai_messages[2].content == "How are you?"

    def test_anthropic_messages_response(self):
        """Test AnthropicMessagesResponse model."""
        from claude_relay.models import (
            AnthropicMessagesResponse,
            AnthropicTextBlock,
            AnthropicUsage,
        )

        response = AnthropicMessagesResponse(
            model="claude-sonnet-4-5-20250929",
            content=[AnthropicTextBlock(text="Hello!")],
            usage=AnthropicUsage(input_tokens=10, output_tokens=5),
        )

        assert response.type == "message"
        assert response.role == "assistant"
        assert response.model == "claude-sonnet-4-5-20250929"
        assert len(response.content) == 1
        assert response.content[0].text == "Hello!"
        assert response.stop_reason == "end_turn"
        assert response.usage.input_tokens == 10
        assert response.usage.output_tokens == 5


class TestAnthropicMessagesToolGating:
    """Unit tests: /v1/messages passes the right options to Claude Code."""

    def _post(self, payload, allow_tools=False):
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        import claude_relay.completion as completion
        import claude_relay.main as main

        captured = {}

        async def fake_run_completion(**kwargs):
            captured.update(kwargs)
            yield {"type": "assistant", "content": [{"type": "text", "text": "ok"}]}

        with (
            patch.object(completion, "validate_claude_code_auth", return_value=(True, {})),
            patch.object(completion.claude_cli, "run_completion", fake_run_completion),
            patch.object(completion.claude_cli, "parse_claude_message", return_value="ok"),
            patch.object(main.constants, "ALLOW_TOOLS", allow_tools),
        ):
            # Host must pass the ALLOWED_HOSTS check (TestClient defaults to "testserver")
            client = TestClient(main.app, base_url="http://localhost")
            response = client.post("/v1/messages", json=payload)
        return response, captured

    def test_tools_disabled_by_default(self):
        from claude_relay.constants import CLAUDE_TOOLS

        response, opts = self._post(
            {"model": "claude-haiku-4-5-20251001", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert response.status_code == 200
        assert opts["max_turns"] == 1
        assert opts["disallowed_tools"] == CLAUDE_TOOLS
        assert opts["permission_mode"] is None
        assert opts["allowed_tools"] is None

    def test_tools_enabled_on_request(self):
        from claude_relay.constants import DEFAULT_ALLOWED_TOOLS

        response, opts = self._post(
            {
                "model": "claude-haiku-4-5-20251001",
                "enable_tools": True,
                "messages": [{"role": "user", "content": "hi"}],
            },
            allow_tools=True,
        )
        assert response.status_code == 200
        assert opts["allowed_tools"] == DEFAULT_ALLOWED_TOOLS
        assert opts["permission_mode"] == "bypassPermissions"

    def test_system_as_text_blocks(self):
        response, opts = self._post(
            {
                "model": "claude-haiku-4-5-20251001",
                "system": [
                    {"type": "text", "text": "Rule one."},
                    {"type": "text", "text": "Rule two.", "cache_control": {"type": "ephemeral"}},
                ],
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        assert response.status_code == 200
        assert opts["system_prompt"] == "Rule one.\n\nRule two."

    def test_tool_use_tool_result_and_image_blocks(self):
        """Tool-using clients' content blocks are accepted and rendered as text."""
        response, opts = self._post(
            {
                "model": "claude-haiku-4-5-20251001",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "What's in this file?"},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": "AA",
                                },
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "Reading it."},
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "read_file",
                                "input": {"path": "a.txt"},
                            },
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": "toolu_1",
                                "content": [{"type": "text", "text": "hello world"}],
                            }
                        ],
                    },
                ],
            }
        )
        assert response.status_code == 200
        prompt = opts["prompt"]
        assert "What's in this file?" in prompt
        assert 'Assistant: Reading it.\nTool call: read_file({"path": "a.txt"})' in prompt
        assert "Tool result (toolu_1): hello world" in prompt
        assert "base64" not in prompt


MODEL = "claude-haiku-4-5-20251001"


class TestAnthropicMessagesEndpoint:
    """/v1/messages prompt, errors and streaming, checked with the real anthropic SDK."""

    def _http(self, chunks, captured=None):
        """Patched Claude Code run plus a TestClient; returns (patches, client)."""
        from fastapi.testclient import TestClient

        import claude_relay.completion as completion
        import claude_relay.main as main

        async def fake_run_completion(**kwargs):
            if captured is not None:
                captured.update(kwargs)
            for chunk in chunks:
                yield chunk

        stack = ExitStack()
        stack.enter_context(
            patch.object(completion, "validate_claude_code_auth", return_value=(True, {}))
        )
        stack.enter_context(
            patch.object(completion.claude_cli, "run_completion", fake_run_completion)
        )
        return stack, TestClient(main.app, base_url="http://localhost")

    def _sdk(self, chunks, captured=None):
        """The real anthropic client talking to the app; returns (patches, client)."""
        # Not a dev dependency yet; run with `--with anthropic`
        anthropic = pytest.importorskip("anthropic")
        httpx2 = pytest.importorskip("httpx2")
        stack, http = self._http(chunks, captured)

        def forward(request):
            # The SDK uses httpx2, so hand each request to the (httpx) TestClient
            response = http.request(
                request.method, str(request.url), headers=request.headers, content=request.read()
            )
            return httpx2.Response(
                response.status_code, headers=response.headers, content=response.content
            )

        client = anthropic.Anthropic(
            api_key="x",
            base_url="http://localhost",
            http_client=httpx2.Client(transport=httpx2.MockTransport(forward)),
            max_retries=0,
        )
        return stack, client

    def _chunks(self, text="Hello there", is_error=False, **extra):
        return [
            {"content": [{"type": "text", "text": text}], "model": MODEL, **extra},
            {
                "subtype": "success",
                "is_error": is_error,
                "result": text,
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 7, "output_tokens": 2},
            },
        ]

    def test_prompt_uses_chat_builder(self):
        captured = {}
        stack, client = self._sdk(self._chunks(), captured)
        with stack:
            client.messages.create(
                model=MODEL,
                max_tokens=10,
                messages=[
                    {"role": "user", "content": "one"},
                    {"role": "assistant", "content": "two"},
                    {"role": "user", "content": "three"},
                ],
            )
        assert captured["prompt"] == "Human: one\n\nAssistant: two\n\nHuman: three"

    def test_stream_with_sdk(self):
        stack, client = self._sdk(self._chunks())
        with (
            stack,
            client.messages.stream(
                model=MODEL, max_tokens=10, messages=[{"role": "user", "content": "hi"}]
            ) as stream,
        ):
            events = [event.type for event in stream]
            message = stream.get_final_message()
        assert events[0] == "message_start"
        assert "content_block_delta" in events and events[-1] == "message_stop"
        assert message.content[0].text == "Hello there"
        assert message.model == MODEL
        assert message.stop_reason == "end_turn"
        assert message.usage.input_tokens == 7 and message.usage.output_tokens == 2

    def test_stream_error_event(self):
        chunks = self._chunks("API Error: 429", is_error=True, error="rate_limit")
        chunks[1]["api_error_status"] = 429
        stack, client = self._sdk(chunks)
        with stack, pytest.raises(Exception) as raised:
            with client.messages.stream(
                model=MODEL, max_tokens=10, messages=[{"role": "user", "content": "hi"}]
            ) as stream:
                stream.get_final_message()
        assert type(raised.value).__name__ == "APIStatusError"
        assert raised.value.body["error"]["type"] == "rate_limit_error"

    def test_unknown_model_stream_is_400(self):
        stack, client = self._sdk(self._chunks())
        with stack, pytest.raises(Exception) as raised:
            with client.messages.stream(
                model="gpt-4o", max_tokens=10, messages=[{"role": "user", "content": "hi"}]
            ) as stream:
                stream.get_final_message()
        assert type(raised.value).__name__ == "BadRequestError"
        assert raised.value.body["type"] == "error"

    @pytest.mark.parametrize(
        "payload,status,error_type",
        [
            ({"messages": "nope"}, 422, "invalid_request_error"),
            ({"model": MODEL, "messages": [], "enable_tools": True}, 403, "permission_error"),
        ],
    )
    def test_errors_in_anthropic_format(self, payload, status, error_type):
        stack, http = self._http(self._chunks())
        with stack:
            response = http.post("/v1/messages", json=payload)
        assert response.status_code == status
        assert response.json()["type"] == "error"
        assert response.json()["error"]["type"] == error_type

    def test_auth_not_configured_is_anthropic_503(self):
        import claude_relay.completion as completion

        stack, http = self._http(self._chunks())
        not_configured = (False, {"errors": ["no key"]})
        with (
            stack,
            patch.object(completion, "validate_claude_code_auth", return_value=not_configured),
        ):
            response = http.post(
                "/v1/messages",
                json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
            )
        assert response.status_code == 503
        body = response.json()
        assert body["type"] == "error" and body["error"]["type"] == "api_error"
        assert "no key" in body["error"]["message"]
