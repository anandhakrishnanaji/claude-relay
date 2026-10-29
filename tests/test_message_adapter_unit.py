#!/usr/bin/env python3
"""
Unit tests for claude_relay/message_adapter.py

Tests the MessageAdapter class for message format conversion.
These are pure unit tests that don't require a running server.
"""

from claude_relay.message_adapter import MessageAdapter
from claude_relay.models import Message


class TestMessagesToPrompt:
    """Test MessageAdapter.messages_to_prompt()"""

    def test_single_user_message(self):
        """Single user message converts correctly."""
        messages = [Message(role="user", content="Hello")]
        prompt, system = MessageAdapter.messages_to_prompt(messages)

        assert "Human: Hello" in prompt
        assert system is None

    def test_user_and_assistant_conversation(self):
        """User and assistant messages form conversation."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
            Message(role="user", content="How are you?"),
        ]
        prompt, system = MessageAdapter.messages_to_prompt(messages)

        assert "Human: Hello" in prompt
        assert "Assistant: Hi there!" in prompt
        assert "Human: How are you?" in prompt

    def test_system_message_extracted(self):
        """System message is extracted as system_prompt."""
        messages = [
            Message(role="system", content="You are a helpful assistant."),
            Message(role="user", content="Hello"),
        ]
        prompt, system = MessageAdapter.messages_to_prompt(messages)

        assert system == "You are a helpful assistant."
        assert "Human: Hello" in prompt

    def test_multiple_system_messages_are_joined(self):
        """Multiple system messages are all kept, in order."""
        messages = [
            Message(role="system", content="First system message"),
            Message(role="user", content="Hello"),
            Message(role="system", content="Second system message"),
        ]
        prompt, system = MessageAdapter.messages_to_prompt(messages)

        assert system == "First system message\n\nSecond system message"

    def test_developer_message_becomes_system_prompt(self):
        """The developer role is treated as system."""
        messages = [
            Message(role="system", content="Be brief."),
            Message(role="developer", content="Answer in French."),
            Message(role="user", content="Hello"),
        ]
        prompt, system = MessageAdapter.messages_to_prompt(messages)

        assert system == "Be brief.\n\nAnswer in French."
        assert "Answer in French" not in prompt

    def test_tool_call_round_trip(self):
        """Assistant tool calls and tool results appear in the prompt."""
        messages = [
            Message(role="user", content="Weather in Paris?"),
            Message(
                role="assistant",
                content=None,
                tool_calls=[
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            ),
            Message(role="tool", tool_call_id="call_1", content="Sunny, 21C"),
        ]
        prompt, system = MessageAdapter.messages_to_prompt(messages)

        assert 'Assistant: Tool call: get_weather({"city": "Paris"})' in prompt
        assert "Human: Tool result (call_1): Sunny, 21C" in prompt
        # A trailing tool result is the human side's turn, so no "Please continue"
        assert "Please continue" not in prompt

    def test_unvalidated_none_content_does_not_crash(self):
        """Messages built without validation and None content still render."""
        messages = [
            Message.model_construct(role="developer", content=None),
            Message.model_construct(role="user", content=None),
        ]
        prompt, system = MessageAdapter.messages_to_prompt(messages)

        assert prompt == "Human: "
        assert system is None

    def test_last_message_not_user_adds_continue(self):
        """If last message isn't from user, adds 'Please continue'."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ]
        prompt, system = MessageAdapter.messages_to_prompt(messages)

        assert "Please continue" in prompt

    def test_last_message_is_user_no_continue(self):
        """If last message is from user, no 'Please continue' added."""
        messages = [
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!"),
            Message(role="user", content="What's up?"),
        ]
        prompt, system = MessageAdapter.messages_to_prompt(messages)

        assert "Please continue" not in prompt

    def test_empty_messages_list(self):
        """Empty messages list returns empty prompt."""
        messages = []
        prompt, system = MessageAdapter.messages_to_prompt(messages)

        assert prompt == ""
        assert system is None


class TestEstimateTokens:
    """Test MessageAdapter.estimate_tokens()"""

    def test_short_text(self):
        """Short text token estimation."""
        # 12 chars / 4 = 3 tokens
        result = MessageAdapter.estimate_tokens("Hello World!")
        assert result == 3

    def test_empty_text(self):
        """Empty text returns 0 tokens."""
        result = MessageAdapter.estimate_tokens("")
        assert result == 0

    def test_none_text(self):
        """None text returns 0 tokens instead of crashing."""
        assert MessageAdapter.estimate_tokens(None) == 0

    def test_long_text(self):
        """Longer text estimation."""
        # 100 chars / 4 = 25 tokens
        text = "a" * 100
        result = MessageAdapter.estimate_tokens(text)
        assert result == 25

    def test_realistic_text(self):
        """Realistic text estimation."""
        text = "This is a realistic sentence that might appear in a conversation."
        result = MessageAdapter.estimate_tokens(text)
        # 67 chars / 4 = 16 tokens
        assert result == 16
