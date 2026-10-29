"""POST /v1/chat/completions (OpenAI Chat Completions)."""

import json
import os
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import APIRouter, Depends, Request
from fastapi.security import HTTPAuthorizationCredentials

from claude_relay import usage as sdk_usage
from claude_relay.auth import security, verify_api_key
from claude_relay.completion import CompletionError, Run, prepare
from claude_relay.errors import openai_error
from claude_relay.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    Choice,
    Message,
    StreamChoice,
    Usage,
)
from claude_relay.rate_limiter import rate_limit_endpoint
from claude_relay.routers.streaming import start_stream

router = APIRouter()


async def chat_stream(run: Run, request_id: str, include_usage: bool) -> AsyncIterator[str]:
    """OpenAI chat.completion.chunk events for one run."""

    def chunk(delta: Dict[str, Any], finish_reason=None, usage=None) -> str:
        choice = StreamChoice(index=0, delta=delta, finish_reason=finish_reason)
        event = ChatCompletionStreamResponse(
            id=request_id, model=run.model, choices=[choice], usage=usage
        )
        return f"data: {event.model_dump_json()}\n\n"

    role_sent = False
    try:
        async for item in run.stream():
            if not role_sent:
                yield chunk({"role": "assistant", "content": ""})
                role_sent = True
            if isinstance(item, str):
                yield chunk({"content": item})
            else:
                completion = item
    except CompletionError as e:
        if e.kind == "model_not_found" and not role_sent:
            raise
        # The 200 status line is already sent, so report it as a final error event
        yield f"data: {json.dumps(openai_error(e))}\n\n"
        return

    usage = Usage(**completion.openai_usage()) if include_usage else None
    yield chunk({}, sdk_usage.openai_finish_reason(completion.chunks), usage)
    yield "data: [DONE]\n\n"


@router.post("/v1/chat/completions")
@rate_limit_endpoint("chat")
async def chat_completions(
    request_body: ChatCompletionRequest,
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """OpenAI-compatible chat completions endpoint."""
    await verify_api_key(request, credentials)

    run = prepare(
        model=request_body.model,
        messages=request_body.messages,
        enable_tools=bool(request_body.enable_tools),
        headers=request.headers,
        effort=request_body.to_claude_options().get("effort"),
        system_suffix=request_body.get_sampling_instructions(),
        session_id=request_body.session_id,
    )
    request_id = f"chatcmpl-{os.urandom(8).hex()}"

    if request_body.stream:
        include_usage = bool(
            request_body.stream_options and request_body.stream_options.include_usage
        )
        return await start_stream(chat_stream(run, request_id, include_usage))

    completion = await run.complete()
    return ChatCompletionResponse(
        id=request_id,
        model=completion.model,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=completion.text),
                finish_reason=sdk_usage.openai_finish_reason(completion.chunks),
            )
        ],
        usage=Usage(**completion.openai_usage()),
    )
