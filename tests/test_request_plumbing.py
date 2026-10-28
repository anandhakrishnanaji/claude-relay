#!/usr/bin/env python3
"""
Unit tests: what the endpoints hand to claude_cli.run_completion.

run_completion is mocked and records the keyword arguments it was called with.
"""

import json
from unittest.mock import patch

from fastapi.testclient import TestClient

import claude_relay.main as main
from claude_relay import usage as sdk_usage

MODEL = "claude-haiku-4-5-20251001"

TOOL_PROMPT = (
    "Use tools like this:\n<read_file>\n<path>src/main.py</path>\n</read_file>\n"
    "Think first in <thinking>...</thinking> tags."
)
USER_TEXT = "<thinking>plan</thinking> Now <read_file><path>a.txt</path></read_file>"


def make_chunks(text="ok", model=MODEL):
    return [
        {"subtype": "init", "data": {"model": model}},
        {"content": [{"type": "text", "text": text}], "model": model},
        {"subtype": "success", "is_error": False, "result": text, "stop_reason": "end_turn"},
    ]


def post(path, payload, chunks=None, headers=None):
    """POST and return (response, list of run_completion kwargs)."""
    calls = []

    async def fake_run_completion(**kwargs):
        calls.append(kwargs)
        for chunk in make_chunks() if chunks is None else chunks:
            yield chunk

    with (
        patch.object(main, "validate_claude_code_auth", return_value=(True, {})),
        patch.object(main.claude_cli, "run_completion", fake_run_completion),
    ):
        client = TestClient(main.app, base_url="http://localhost")
        response = client.post(path, json=payload, headers=headers or {})
    return response, calls


class TestPromptPassthrough:
    """Incoming prompts reach the SDK unchanged; filter_content is output-only."""

    def chat_payload(self, **extra):
        return {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": TOOL_PROMPT},
                {"role": "user", "content": USER_TEXT},
            ],
            **extra,
        }

    def test_chat_non_streaming(self):
        response, calls = post("/v1/chat/completions", self.chat_payload())
        assert response.status_code == 200
        assert calls[0]["system_prompt"] == TOOL_PROMPT
        assert USER_TEXT in calls[0]["prompt"]

    def test_chat_streaming(self):
        response, calls = post("/v1/chat/completions", self.chat_payload(stream=True))
        assert response.status_code == 200
        assert calls[0]["system_prompt"] == TOOL_PROMPT
        assert USER_TEXT in calls[0]["prompt"]

    def test_anthropic_messages(self):
        payload = {
            "model": MODEL,
            "system": TOOL_PROMPT,
            "messages": [{"role": "user", "content": USER_TEXT}],
        }
        response, calls = post("/v1/messages", payload)
        assert response.status_code == 200
        assert calls[0]["system_prompt"] == TOOL_PROMPT
        assert USER_TEXT in calls[0]["prompt"]

    def test_assistant_history_kept_raw(self):
        # A Cline-style earlier tool call stays in the history being replayed
        payload = self.chat_payload()
        payload["messages"] += [
            {"role": "assistant", "content": "<read_file><path>a.txt</path></read_file>"},
            {"role": "user", "content": "contents of a.txt"},
        ]
        response, calls = post("/v1/chat/completions", payload)
        assert response.status_code == 200
        assert "Assistant: <read_file><path>a.txt</path></read_file>" in calls[0]["prompt"]

    def test_output_still_filtered(self):
        chunks = make_chunks(text="<thinking>secret</thinking>Answer")
        response, _ = post("/v1/chat/completions", self.chat_payload(), chunks)
        assert response.json()["choices"][0]["message"]["content"] == "Answer"


REAL_MODEL = "claude-haiku-5-5"

# What the CLI sends for an unknown --model (captured from CLI 2.1.310)
UNKNOWN_MODEL_TEXT = (
    "There's an issue with the selected model (claude-nope). "
    "It may not exist or you may not have access to it."
)
UNKNOWN_MODEL_CHUNKS = [
    {"subtype": "init", "data": {"model": "claude-nope"}},
    {
        "content": [{"type": "text", "text": UNKNOWN_MODEL_TEXT}],
        "model": "<synthetic>",
        "error": "model_not_found",
    },
    {"subtype": "success", "is_error": True, "result": UNKNOWN_MODEL_TEXT, "api_error_status": 404},
    {
        "type": "result",
        "subtype": "error_during_execution",
        "is_error": True,
        "error_message": "Claude Code returned an error result: " + UNKNOWN_MODEL_TEXT,
    },
]


def chat_payload(model, **extra):
    return {"model": model, "messages": [{"role": "user", "content": "hi"}], **extra}


def anthropic_payload(model):
    return {"model": model, "messages": [{"role": "user", "content": "hi"}]}


def stream_events(response):
    return [
        json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")
    ]


class TestReportedModel:
    """Responses name the model that actually ran, not the requested alias."""

    def test_chat_non_streaming(self):
        chunks = make_chunks(model=REAL_MODEL)
        response, calls = post("/v1/chat/completions", chat_payload("haiku"), chunks)
        assert calls[0]["model"] == "haiku"
        assert response.json()["model"] == REAL_MODEL

    def test_chat_streaming(self):
        chunks = make_chunks(model=REAL_MODEL)
        response, _ = post("/v1/chat/completions", chat_payload("haiku", stream=True), chunks)
        events = stream_events(response)
        assert events and {e["model"] for e in events} == {REAL_MODEL}

    def test_anthropic_messages(self):
        chunks = make_chunks(model=REAL_MODEL)
        response, _ = post("/v1/messages", anthropic_payload("haiku"), chunks)
        assert response.json()["model"] == REAL_MODEL

    def test_falls_back_to_requested_name(self):
        chunks = [{"subtype": "success", "is_error": False, "result": "ok"}]
        response, _ = post("/v1/chat/completions", chat_payload("sonnet"), chunks)
        assert response.json()["model"] == "sonnet"

    def test_extract_model_order(self):
        assert sdk_usage.extract_model([], "x") == "x"
        assert sdk_usage.extract_model([{"subtype": "init", "data": {"model": "a"}}], "x") == "a"
        usage = {"model_usage": {"small": {"outputTokens": 1}, "big": {"outputTokens": 9}}}
        assert sdk_usage.extract_model([usage], "x") == "big"
        assistant = {"content": [], "model": "real"}
        assert sdk_usage.extract_model([assistant, usage], "x") == "real"
        assert sdk_usage.extract_model([{"content": [], "model": "<synthetic>"}], "x") == "x"


class TestUnknownModels:
    def test_non_claude_name_rejected_up_front(self):
        for extra in ({}, {"stream": True}):
            response, calls = post("/v1/chat/completions", chat_payload("gpt-4o", **extra))
            assert response.status_code == 400
            assert response.json()["error"]["code"] == "model_not_found"
            assert response.json()["error"]["type"] == "invalid_request_error"
            assert calls == []

    def test_non_claude_name_rejected_on_messages(self):
        response, calls = post("/v1/messages", anthropic_payload("gpt-4o"))
        assert response.status_code == 400
        assert response.json()["type"] == "error"
        assert response.json()["error"]["type"] == "invalid_request_error"
        assert calls == []

    def test_model_aliases_env(self):
        with patch("claude_relay.constants.MODEL_ALIASES", {"gpt-4o": "haiku"}):
            response, calls = post("/v1/chat/completions", chat_payload("gpt-4o"))
            assert response.status_code == 200
            assert calls[0]["model"] == "haiku"
            response, calls = post("/v1/messages", anthropic_payload("gpt-4o"))
            assert response.status_code == 200
            assert calls[0]["model"] == "haiku"

    def test_cli_unknown_model_is_400(self):
        for extra in ({}, {"stream": True}):
            payload = chat_payload("claude-nope", **extra)
            response, _ = post("/v1/chat/completions", payload, UNKNOWN_MODEL_CHUNKS)
            assert response.status_code == 400
            error = response.json()["error"]
            assert error["code"] == "model_not_found"
            assert "issue with the selected model" in error["message"]

    def test_cli_unknown_model_is_400_on_messages(self):
        response, _ = post("/v1/messages", anthropic_payload("claude-nope"), UNKNOWN_MODEL_CHUNKS)
        assert response.status_code == 400
        assert response.json()["error"]["type"] == "invalid_request_error"


def test_models_list_includes_aliases():
    response = TestClient(main.app, base_url="http://localhost").get("/v1/models")
    ids = {model["id"] for model in response.json()["data"]}
    assert {"sonnet", "opus", "haiku", "claude-opus-5-5"} <= ids
