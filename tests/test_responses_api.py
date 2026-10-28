"""Tests for POST /v1/responses with the Claude Agent SDK mocked."""

from contextlib import contextmanager
from unittest.mock import patch

import openai
import pytest
from fastapi.testclient import TestClient

import claude_relay.main as main
from claude_relay.constants import CLAUDE_TOOLS

MODEL = "claude-haiku-4-5-20251001"


def _result(text="Hello there", stop_reason="end_turn", **extra):
    return {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": text,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": 10,
            "output_tokens": 3,
            "cache_read_input_tokens": 4,
            "cache_creation_input_tokens": 2,
        },
        **extra,
    }


def _assistant(text):
    return {"type": "assistant", "content": [{"type": "text", "text": text}]}


@contextmanager
def _client(chunks):
    """TestClient whose Claude Code run yields `chunks`; records the run's kwargs."""
    captured = {}

    async def fake_run_completion(**kwargs):
        captured.update(kwargs)
        for chunk in chunks:
            yield chunk

    with (
        patch.object(main, "validate_claude_code_auth", return_value=(True, {})),
        patch.object(main.claude_cli, "run_completion", fake_run_completion),
    ):
        yield TestClient(main.app, base_url="http://localhost"), captured


def test_string_input():
    with _client([_assistant("Hello there"), _result()]) as (client, opts):
        r = client.post("/v1/responses", json={"model": MODEL, "input": "Say hi"})
    assert r.status_code == 200
    body = r.json()
    assert body["id"].startswith("resp_")
    assert body["object"] == "response"
    assert body["status"] == "completed"
    assert body["model"] == MODEL
    assert body["output"][0]["type"] == "message"
    assert body["output"][0]["content"] == [
        {"type": "output_text", "text": "Hello there", "annotations": []}
    ]
    assert body["output_text"] == "Hello there"
    assert body["usage"]["input_tokens"] == 16
    assert body["usage"]["output_tokens"] == 3
    assert body["usage"]["total_tokens"] == 19
    assert body["usage"]["input_tokens_details"]["cached_tokens"] == 4
    assert opts["prompt"] == "Human: Say hi"
    assert opts["system_prompt"] is None
    assert opts["max_turns"] == 1
    assert opts["disallowed_tools"] == CLAUDE_TOOLS
    assert "allowed_tools" not in opts and "permission_mode" not in opts


def test_message_list_input_and_instructions():
    payload = {
        "model": MODEL,
        "instructions": "Be terse.",
        "input": [
            {"role": "developer", "content": "Answer in English."},
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "<a>"}]},
            {"role": "assistant", "content": [{"type": "output_text", "text": "Yes?"}]},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Go on"},
                    {"type": "input_image", "image_url": "http://x/y.png"},
                ],
            },
        ],
    }
    with _client([_assistant("ok"), _result("ok")]) as (client, opts):
        r = client.post("/v1/responses", json=payload)
    assert r.status_code == 200
    assert opts["system_prompt"] == "Be terse.\n\nAnswer in English."
    # XML-looking input is passed through untouched; the image part is dropped
    assert opts["prompt"] == "Human: <a>\n\nAssistant: Yes?\n\nHuman: Go on"


@pytest.mark.parametrize(
    "extra, param",
    [
        ({"tools": [{"type": "function", "name": "f", "parameters": {}}]}, "tools"),
        ({"tool_choice": "required"}, "tool_choice"),
        ({"previous_response_id": "resp_123"}, "previous_response_id"),
    ],
)
def test_unsupported_parameters_give_400(extra, param):
    with _client([_result()]) as (client, opts):
        r = client.post("/v1/responses", json={"model": MODEL, "input": "hi", **extra})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == param
    assert r.json()["error"]["type"] == "invalid_request_error"
    assert opts == {}  # Claude Code never ran


def test_empty_tools_and_auto_tool_choice_are_fine():
    with _client([_assistant("ok"), _result("ok")]) as (client, _):
        r = client.post(
            "/v1/responses",
            json={"model": MODEL, "input": "hi", "tools": [], "tool_choice": "auto"},
        )
    assert r.status_code == 200


def test_max_tokens_gives_incomplete():
    with _client([_assistant("Hel"), _result("Hel", stop_reason="max_tokens")]) as (client, _):
        r = client.post("/v1/responses", json={"model": MODEL, "input": "hi"})
    body = r.json()
    assert body["status"] == "incomplete"
    assert body["incomplete_details"] == {"reason": "max_output_tokens"}
    assert body["output"][0]["status"] == "incomplete"


def test_error_result_gives_error_status():
    error = {
        "type": "result",
        "subtype": "error_during_execution",
        "is_error": True,
        "error_message": "boom",
    }
    with _client([error]) as (client, _):
        r = client.post("/v1/responses", json={"model": MODEL, "input": "hi"})
        stream = client.post("/v1/responses", json={"model": MODEL, "input": "hi", "stream": True})
    assert r.status_code == 502
    assert r.json()["error"]["message"] == "boom"
    assert "response.failed" in stream.text
    assert "response.completed" not in stream.text
    assert "output_text.delta" not in stream.text


def _sdk_client(client):
    return openai.OpenAI(base_url="http://localhost/v1", api_key="unused", http_client=client)


def test_openai_sdk_non_streaming():
    with _client([_assistant("Hello there"), _result()]) as (client, _):
        response = _sdk_client(client).responses.create(model=MODEL, input="Say hi")
    assert response.output_text == "Hello there"
    assert response.usage.input_tokens_details.cached_tokens == 4


def test_openai_sdk_parses_stream():
    chunks = [_assistant("Hello there"), _assistant("!"), _result("Hello there\n!")]
    with _client(chunks) as (client, _):
        events = list(_sdk_client(client).responses.create(model=MODEL, input="hi", stream=True))
        with _sdk_client(client).responses.stream(model=MODEL, input="hi") as stream:
            final = stream.get_final_response()

    assert [e.type for e in events] == [
        "response.created",
        "response.in_progress",
        "response.output_item.added",
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]
    assert [e.sequence_number for e in events] == list(range(len(events)))
    assert [e.delta for e in events if e.type == "response.output_text.delta"] == [
        "Hello there",
        "!",
    ]
    assert events[-1].response.output_text == "Hello there!"
    assert events[-1].response.usage.total_tokens == 19
    assert final.output_text == "Hello there!"


def test_stream_max_tokens_ends_with_incomplete():
    chunks = [_assistant("Hel"), _result("Hel", stop_reason="max_tokens")]
    with _client(chunks) as (client, _):
        events = list(_sdk_client(client).responses.create(model=MODEL, input="hi", stream=True))
    assert events[-1].type == "response.incomplete"
    assert events[-1].response.incomplete_details.reason == "max_output_tokens"
