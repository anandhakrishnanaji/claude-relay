#!/usr/bin/env python3
"""In-process test: agentic-client chat payloads are accepted."""


def test_agentic_client_payload_accepted():
    """An OpenClaw-style payload (tool role, null content with tool_calls, image part,
    developer role, extra fields) gets 200 from /v1/chat/completions, not 422."""
    from unittest.mock import patch
    from fastapi.testclient import TestClient
    import claude_relay.main as main

    captured = {}

    async def fake_run_completion(**kwargs):
        captured.update(kwargs)
        yield {"type": "assistant", "content": [{"type": "text", "text": "It is sunny."}]}

    payload = {
        "model": "claude-haiku-4-5-20251001",
        "parallel_tool_calls": True,
        "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
        "messages": [
            {"role": "developer", "content": "You are OpenClaw."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's the weather here?"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny, 21C"},
        ],
    }

    with (
        patch.object(main, "validate_claude_code_auth", return_value=(True, {})),
        patch.object(main.claude_cli, "run_completion", fake_run_completion),
        patch.object(main.claude_cli, "parse_claude_message", return_value="It is sunny."),
    ):
        response = TestClient(main.app, base_url="http://localhost").post(
            "/v1/chat/completions", json=payload
        )

    assert response.status_code == 200, response.text
    message = response.json()["choices"][0]["message"]
    assert message == {"role": "assistant", "content": "It is sunny.", "name": None}

    prompt = captured["prompt"]
    assert "Human: What's the weather here?" in prompt
    assert 'Assistant: Tool call: get_weather({"city": "Paris"})' in prompt
    assert "Human: Tool result (call_1): Sunny, 21C" in prompt
    assert "base64" not in prompt
    assert captured["system_prompt"].startswith("You are OpenClaw.")
