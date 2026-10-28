#!/usr/bin/env python3
"""
Unit tests: responses report the SDK's real usage and stop reason.

run_completion is mocked to yield dicts shaped like the converted SDK
AssistantMessage and ResultMessage.
"""

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

import claude_relay.main as main
from claude_relay import usage as sdk_usage

MODEL = "claude-haiku-4-5-20251001"

REAL_USAGE = {
    "input_tokens": 12,
    "output_tokens": 34,
    "cache_read_input_tokens": 100,
    "cache_creation_input_tokens": 5,
}


def make_chunks(text="Hello there", usage=REAL_USAGE, stop_reason="end_turn", **result_fields):
    assistant = {"content": [{"type": "text", "text": text}], "model": MODEL}
    result = {
        "subtype": "success",
        "is_error": False,
        "num_turns": 1,
        "session_id": "s",
        "result": text,
        "usage": usage,
        "stop_reason": stop_reason,
        "total_cost_usd": 0.001,
    }
    result.update(result_fields)
    return [assistant, result]


def post(path, payload, chunks):
    async def fake_run_completion(**kwargs):
        for chunk in chunks:
            yield chunk

    with (
        patch.object(main, "validate_claude_code_auth", return_value=(True, {})),
        patch.object(main.claude_cli, "run_completion", fake_run_completion),
    ):
        return TestClient(main.app, base_url="http://localhost").post(path, json=payload)


def chat(chunks, **extra):
    payload = {"model": MODEL, "messages": [{"role": "user", "content": "hi"}], **extra}
    return post("/v1/chat/completions", payload, chunks)


def stream_events(response):
    events = []
    for line in response.text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            events.append(json.loads(line[len("data: ") :]))
    return events


class TestChatCompletionsUsage:
    def test_real_usage_non_streaming(self):
        response = chat(make_chunks())
        assert response.status_code == 200
        body = response.json()
        assert body["usage"] == {
            "prompt_tokens": 117,
            "completion_tokens": 34,
            "total_tokens": 151,
        }
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["choices"][0]["message"]["content"] == "Hello there"

    def test_max_tokens_maps_to_length(self):
        response = chat(make_chunks(stop_reason="max_tokens"))
        assert response.status_code == 200
        assert response.json()["choices"][0]["finish_reason"] == "length"

    def test_fallback_estimate_without_usage(self):
        response = chat(make_chunks(text="x" * 40, usage=None))
        assert response.status_code == 200
        usage = response.json()["usage"]
        # MessageAdapter.estimate_tokens: ~4 characters per token
        assert usage["completion_tokens"] == 10
        assert usage["total_tokens"] == usage["prompt_tokens"] + 10

    def test_streaming_include_usage(self):
        response = chat(
            make_chunks(stop_reason="max_tokens"),
            stream=True,
            stream_options={"include_usage": True},
        )
        assert response.status_code == 200
        final = stream_events(response)[-1]
        assert final["choices"][0]["finish_reason"] == "length"
        assert final["usage"] == {
            "prompt_tokens": 117,
            "completion_tokens": 34,
            "total_tokens": 151,
        }

    def test_streaming_without_include_usage(self):
        response = chat(make_chunks(), stream=True)
        final = stream_events(response)[-1]
        assert final["choices"][0]["finish_reason"] == "stop"
        assert final.get("usage") is None

    def test_error_result_returns_error_status(self):
        chunks = make_chunks(
            text="API Error: 529 overloaded", is_error=True, api_error_status=529
        )
        response = chat(chunks)
        assert response.status_code == 502
        assert "529" in response.text


class TestAnthropicMessagesUsage:
    def _post(self, chunks):
        payload = {
            "model": MODEL,
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}],
        }
        return post("/v1/messages", payload, chunks)

    def test_real_usage_and_stop_reason(self):
        response = self._post(make_chunks(stop_reason="max_tokens"))
        assert response.status_code == 200
        body = response.json()
        assert body["stop_reason"] == "max_tokens"
        assert body["usage"] == REAL_USAGE

    def test_default_stop_reason_and_fallback_usage(self):
        response = self._post(make_chunks(text="x" * 40, usage=None, stop_reason=None))
        assert response.status_code == 200
        body = response.json()
        assert body["stop_reason"] == "end_turn"
        assert body["usage"]["output_tokens"] == 10

    def test_rate_limited_result(self):
        chunks = make_chunks(text="API Error: 429", is_error=True, api_error_status=429)
        assert self._post(chunks).status_code == 429


class TestUsageHelpers:
    @pytest.mark.parametrize(
        "stop_reason,expected",
        [
            ("end_turn", "stop"),
            ("stop_sequence", "stop"),
            ("max_tokens", "length"),
            ("tool_use", "stop"),
            ("refusal", "content_filter"),
            (None, "stop"),
            ("something_new", "stop"),
        ],
    )
    def test_openai_finish_reason(self, stop_reason, expected):
        assert sdk_usage.openai_finish_reason(make_chunks(stop_reason=stop_reason)) == expected

    def test_stop_reason_falls_back_to_assistant_message(self):
        chunks = make_chunks(stop_reason=None)
        chunks[0]["stop_reason"] = "max_tokens"
        assert sdk_usage.openai_finish_reason(chunks) == "length"

    def test_max_turns_is_not_an_error(self):
        chunks = make_chunks(is_error=True, subtype="error_max_turns")
        assert sdk_usage.result_error(chunks) is None

    def test_cli_exception_chunk_is_an_error(self):
        chunks = [
            {
                "type": "result",
                "subtype": "error_during_execution",
                "is_error": True,
                "error_message": "boom",
            }
        ]
        assert sdk_usage.result_error(chunks) == {"status": 502, "message": "boom"}
        assert sdk_usage.extract_usage(chunks) is None
