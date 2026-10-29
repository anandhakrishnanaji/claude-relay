from typing import List, Optional
from claude_relay.models import Message


class MessageAdapter:
    """Converts between OpenAI message format and Claude Code prompts."""

    @staticmethod
    def messages_to_prompt(messages: List[Message]) -> tuple[str, Optional[str]]:
        """
        Convert OpenAI messages to Claude Code prompt format.
        Returns (prompt, system_prompt)
        """
        system_parts = []
        conversation_parts = []

        for message in messages:
            content = message.content or ""
            if message.role in ("system", "developer"):
                # Join every system/developer message into the system prompt
                system_parts.append(content)
            elif message.role in ("user", "tool", "function"):
                # Tool results come back on the human side of the conversation;
                # Message has already rendered them as "Tool result (...): ..."
                conversation_parts.append(f"Human: {content}")
            elif message.role == "assistant":
                conversation_parts.append(f"Assistant: {content}")

        system_prompt = "\n\n".join(part for part in system_parts if part) or None

        # Join conversation parts
        prompt = "\n\n".join(conversation_parts)

        # If the last message wasn't from the user (or a tool result), add a prompt for assistant
        if messages and messages[-1].role not in ("user", "tool", "function"):
            prompt += "\n\nHuman: Please continue."

        return prompt, system_prompt

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Rough estimation of token count.
        OpenAI's rule of thumb: ~4 characters per token for English text.
        """
        return len(text or "") // 4
