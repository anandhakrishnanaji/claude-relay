"""POST /v1/messages (Anthropic Messages)."""

import json
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.security import HTTPAuthorizationCredentials

from claude_relay import usage as sdk_usage
from claude_relay.auth import security, verify_api_key
from claude_relay.completion import CompletionError, Run, prepare
from claude_relay.errors import anthropic_error
from claude_relay.models import (
    AnthropicMessagesRequest,
    AnthropicMessagesResponse,
    AnthropicTextBlock,
    AnthropicUsage,
    Message,
)
from claude_relay.rate_limiter import rate_limit_endpoint
from claude_relay.routers.streaming import start_stream

router = APIRouter()


def sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def messages_stream(run: Run) -> AsyncIterator[str]:
    """Anthropic Messages SSE events for one run (shapes from anthropic.types)."""
    message_id = f"msg_{uuid.uuid4().hex[:24]}"
    started = False

    def start_events() -> List[str]:
        message = AnthropicMessagesResponse(
            id=message_id,
            model=run.model,
            content=[],
            stop_reason=None,
            usage=AnthropicUsage(input_tokens=0, output_tokens=0),
        )
        return [
            sse("message_start", {"type": "message_start", "message": message.model_dump()}),
            sse(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""},
                },
            ),
        ]

    try:
        async for item in run.stream():
            if not started:
                for event in start_events():
                    yield event
                started = True
            if isinstance(item, str):
                delta = {"type": "text_delta", "text": item}
                yield sse(
                    "content_block_delta",
                    {"type": "content_block_delta", "index": 0, "delta": delta},
                )
            else:
                completion = item
    except CompletionError as e:
        if e.kind == "model_not_found" and not started:
            raise
        yield sse("error", anthropic_error(e.status, e.message))
        return

    stop_reason = sdk_usage.anthropic_stop_reason(completion.chunks)
    yield sse("content_block_stop", {"type": "content_block_stop", "index": 0})
    yield sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": completion.anthropic_usage(),
        },
    )
    yield sse("message_stop", {"type": "message_stop"})


@router.post("/v1/messages")
@rate_limit_endpoint("chat")
async def anthropic_messages(
    request_body: AnthropicMessagesRequest,
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Anthropic Messages API compatible endpoint."""
    await verify_api_key(request, credentials)

    system = request_body.system_text()
    messages = [Message(role="system", content=system)] if system else []
    run = prepare(
        model=request_body.model,
        messages=messages + request_body.to_openai_messages(),
        enable_tools=bool(request_body.enable_tools),
        thinking=(
            request_body.thinking.model_dump(exclude_none=True) if request_body.thinking else None
        ),
        effort=request_body.output_config.effort if request_body.output_config else None,
    )

    if request_body.stream:
        return await start_stream(messages_stream(run))

    completion = await run.complete()
    return AnthropicMessagesResponse(
        model=completion.model,
        content=[AnthropicTextBlock(text=completion.text)],
        stop_reason=sdk_usage.anthropic_stop_reason(completion.chunks),
        usage=AnthropicUsage(**completion.anthropic_usage()),
    )
