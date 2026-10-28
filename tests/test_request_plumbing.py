#!/usr/bin/env python3
"""
Unit tests: what the endpoints hand to claude_cli.run_completion.

run_completion is mocked and records the keyword arguments it was called with.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

import claude_relay.main as main

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
