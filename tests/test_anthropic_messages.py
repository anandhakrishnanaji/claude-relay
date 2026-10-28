#!/usr/bin/env python3
"""
Tests for the Anthropic Messages API compatible endpoint (/v1/messages).

This endpoint provides compatibility with the native Anthropic SDK,
enabling tools like VC to use this wrapper via custom base URL configuration.
"""

import pytest
import requests

from tests.conftest import requires_server

BASE_URL = "http://localhost:8000"


class TestAnthropicMessagesModels:
    """Test Anthropic API model classes."""

    def test_anthropic_text_block(self):
        """Test AnthropicTextBlock model."""
        from src.models import AnthropicTextBlock

        block = AnthropicTextBlock(text="Hello world")
        assert block.type == "text"
        assert block.text == "Hello world"

    def test_anthropic_message(self):
        """Test AnthropicMessage model."""
        from src.models import AnthropicMessage

        # String content
        msg = AnthropicMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

        # List content
        from src.models import AnthropicTextBlock

        msg2 = AnthropicMessage(role="assistant", content=[AnthropicTextBlock(text="Hi there")])
        assert msg2.role == "assistant"
        assert len(msg2.content) == 1

    def test_anthropic_messages_request(self):
        """Test AnthropicMessagesRequest model."""
        from src.models import AnthropicMessagesRequest, AnthropicMessage

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
        from src.models import AnthropicMessagesRequest, AnthropicMessage

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
        from src.models import (
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

    def _post(self, payload):
        from unittest.mock import patch
        from fastapi.testclient import TestClient
        import src.main as main

        captured = {}

        async def fake_run_completion(**kwargs):
            captured.update(kwargs)
            yield {"type": "assistant", "content": [{"type": "text", "text": "ok"}]}

        with (
            patch.object(main, "validate_claude_code_auth", return_value=(True, {})),
            patch.object(main.claude_cli, "run_completion", fake_run_completion),
            patch.object(main.claude_cli, "parse_claude_message", return_value="ok"),
        ):
            response = TestClient(main.app).post("/v1/messages", json=payload)
        return response, captured

    def test_tools_disabled_by_default(self):
        from src.constants import CLAUDE_TOOLS

        response, opts = self._post(
            {"model": "claude-haiku-4-5-20251001", "messages": [{"role": "user", "content": "hi"}]}
        )
        assert response.status_code == 200
        assert opts["max_turns"] == 1
        assert opts["disallowed_tools"] == CLAUDE_TOOLS
        assert "permission_mode" not in opts
        assert "allowed_tools" not in opts

    def test_tools_enabled_on_request(self):
        from src.constants import DEFAULT_ALLOWED_TOOLS

        response, opts = self._post(
            {
                "model": "claude-haiku-4-5-20251001",
                "enable_tools": True,
                "messages": [{"role": "user", "content": "hi"}],
            }
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


class TestAnthropicMessagesEndpoint:
    """Integration tests for /v1/messages endpoint."""

    @requires_server
    def test_basic_message(self):
        """Test basic message request."""
        response = requests.post(
            f"{BASE_URL}/v1/messages",
            json={
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "Say 'test' and nothing else"}],
            },
        )

        assert response.status_code == 200
        result = response.json()

        # Verify Anthropic response format
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert "content" in result
        assert len(result["content"]) > 0
        assert result["content"][0]["type"] == "text"
        assert "usage" in result
        assert "input_tokens" in result["usage"]
        assert "output_tokens" in result["usage"]

    @requires_server
    def test_message_with_system_prompt(self):
        """Test message with system prompt."""
        response = requests.post(
            f"{BASE_URL}/v1/messages",
            json={
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 50,
                "system": "You always respond with exactly one word.",
                "messages": [{"role": "user", "content": "Say hello"}],
            },
        )

        assert response.status_code == 200
        result = response.json()
        assert result["type"] == "message"
        assert len(result["content"]) > 0

    @requires_server
    def test_multi_turn_conversation(self):
        """Test multi-turn conversation."""
        response = requests.post(
            f"{BASE_URL}/v1/messages",
            json={
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 100,
                "messages": [
                    {"role": "user", "content": "My name is Alice."},
                    {"role": "assistant", "content": "Hello Alice!"},
                    {"role": "user", "content": "What's my name?"},
                ],
            },
        )

        assert response.status_code == 200
        result = response.json()
        assert result["type"] == "message"
        # The response should reference Alice
        response_text = result["content"][0]["text"].lower()
        assert "alice" in response_text

    @requires_server
    def test_invalid_request_missing_messages(self):
        """Test error handling for missing messages."""
        response = requests.post(
            f"{BASE_URL}/v1/messages",
            json={
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 50,
                # Missing 'messages' field
            },
        )

        assert response.status_code == 422  # Validation error

    @requires_server
    def test_response_format_matches_anthropic_sdk(self):
        """Test that response format matches what Anthropic SDK expects."""
        response = requests.post(
            f"{BASE_URL}/v1/messages",
            json={
                "model": "claude-sonnet-4-5-20250929",
                "max_tokens": 50,
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )

        assert response.status_code == 200
        result = response.json()

        # Required fields for Anthropic SDK compatibility
        assert "id" in result
        assert result["id"].startswith("msg_")
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert isinstance(result["content"], list)
        assert result["stop_reason"] in ["end_turn", "max_tokens", "stop_sequence"]
        assert "usage" in result
        assert "input_tokens" in result["usage"]
        assert "output_tokens" in result["usage"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
