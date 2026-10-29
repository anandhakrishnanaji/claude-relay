#!/usr/bin/env python3
"""
Tests for the Anthropic Messages API compatible endpoint (/v1/messages).

This endpoint provides compatibility with the native Anthropic SDK,
enabling tools like VC to use this wrapper via custom base URL configuration.
"""


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
