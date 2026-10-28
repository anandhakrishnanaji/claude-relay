#!/usr/bin/env python3
"""
Quick endpoint test for Claude Code OpenAI wrapper.
Run this while the server is running on localhost:8000
"""

import pytest
import requests

from tests.conftest import requires_server
import json

BASE_URL = "http://localhost:8000"


@requires_server
def test_health():
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"  Status: {response.status_code}")
        print(f"  Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"  Error: {e}")
        return False


@requires_server
def test_auth_status():
    print("\nTesting /v1/auth/status endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/v1/auth/status")
        print(f"  Status: {response.status_code}")
        print(f"  Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"  Error: {e}")
        return False


@requires_server
def test_models():
    print("\nTesting /v1/models endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/v1/models")
        print(f"  Status: {response.status_code}")
        models = response.json()
        print(f"  Found {len(models.get('data', []))} models")
        for model in models.get("data", [])[:3]:  # Show first 3
            print(f"    - {model.get('id')}")
        return response.status_code == 200
    except Exception as e:
        print(f"  Error: {e}")
        return False


@requires_server
def test_chat_completion():
    print("\nTesting /v1/chat/completions endpoint...")
    try:
        payload = {
            "model": "claude-haiku-4-5",  # Use fastest model
            "messages": [
                {
                    "role": "user",
                    "content": "Say 'Hello, SDK integration working!' and nothing else.",
                }
            ],
            "max_tokens": 50,
        }

        response = requests.post(
            f"{BASE_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
        )

        print(f"  Status: {response.status_code}")

        if response.status_code == 200:
            result = response.json()
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"  Response: {content}")
            print(f"  Usage: {result.get('usage', {})}")
            return True
        else:
            print(f"  Error: {response.text}")
            return False

    except Exception as e:
        print(f"  Error: {e}")
        return False


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


def main():
    print("Claude Code OpenAI Wrapper - Endpoint Tests")
    print("=" * 50)

    tests = [
        ("Health Check", test_health),
        ("Auth Status", test_auth_status),
        ("Models List", test_models),
        ("Chat Completion", test_chat_completion),
    ]

    passed = 0
    total = len(tests)

    for name, test_func in tests:
        if test_func():
            print(f"✓ {name} passed")
            passed += 1
        else:
            print(f"✗ {name} failed")

    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! SDK integration is working correctly.")
    else:
        print("❌ Some tests failed. Check server logs for details.")


if __name__ == "__main__":
    main()
