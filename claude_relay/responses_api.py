"""OpenAI Responses API (POST /v1/responses), text only.

Shapes follow openai.types.responses. The server is stateless: nothing is
stored, so previous_response_id/conversation are rejected rather than
silently answered without their context. Function tools are rejected too.

claude_cli and validate_claude_code_auth are looked up on claude_relay.main at
request time: main includes this router, so importing main here at module
level would be circular (and tests patch them on main).
"""

import itertools
import json
import logging
import os
import time
from typing import Any, AsyncGenerator, Dict, List, Literal, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from claude_relay import usage as sdk_usage
from claude_relay.auth import security, verify_api_key
from claude_relay.constants import CLAUDE_TOOLS
from claude_relay.message_adapter import MessageAdapter
from claude_relay.models import ContentPart, Message, get_default_model
from claude_relay.parameter_validator import ParameterValidator
from claude_relay.rate_limiter import rate_limit_endpoint

logger = logging.getLogger(__name__)

router = APIRouter()

# SDK stop reasons that end a response as "incomplete", with the Responses API reason
_INCOMPLETE_REASONS = {"max_tokens": "max_output_tokens", "refusal": "content_filter"}


class ResponseInputMessage(BaseModel):
    type: Literal["message"] = "message"
    role: Literal["user", "assistant", "system", "developer"]
    # input_text/output_text parts are kept; Message drops images and files with a log
    content: Union[str, List[ContentPart]]


class ResponsesRequest(BaseModel):
    model: str = Field(default_factory=get_default_model)
    input: Union[str, List[ResponseInputMessage]]
    instructions: Optional[str] = None
    stream: bool = False
    # Accepted for compatibility; Claude Code has no output-token cap or sampling knobs
    max_output_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    metadata: Optional[Dict[str, str]] = None
    user: Optional[str] = None
    reasoning: Optional[Dict[str, Any]] = None
    # Rejected when they ask for something this server can't do
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    previous_response_id: Optional[str] = None
    conversation: Optional[Union[str, Dict[str, Any]]] = None


class ResponseOutputText(BaseModel):
    type: Literal["output_text"] = "output_text"
    text: str
    annotations: List[Any] = []


class ResponseOutputMessage(BaseModel):
    id: str
    type: Literal["message"] = "message"
    role: Literal["assistant"] = "assistant"
    status: Literal["in_progress", "completed", "incomplete"]
    content: List[ResponseOutputText]


class ResponseUsage(BaseModel):
    input_tokens: int
    input_tokens_details: Dict[str, int]
    output_tokens: int
    output_tokens_details: Dict[str, int]
    total_tokens: int


class ResponseObject(BaseModel):
    id: str
    object: Literal["response"] = "response"
    created_at: int
    status: Literal["in_progress", "completed", "incomplete", "failed"]
    model: str
    output: List[ResponseOutputMessage] = []
    # Not part of the wire format (the SDK computes it), but handy for raw HTTP clients
    output_text: str = ""
    error: Optional[Dict[str, Any]] = None
    incomplete_details: Optional[Dict[str, Any]] = None
    instructions: Optional[str] = None
    max_output_tokens: Optional[int] = None
    metadata: Optional[Dict[str, str]] = None
    parallel_tool_calls: bool = False
    previous_response_id: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tool_choice: str = "none"
    tools: List[Any] = []
    usage: Optional[ResponseUsage] = None
    user: Optional[str] = None


def _error(status: int, message: str, param: Optional[str] = None) -> JSONResponse:
    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": param,
                "code": "unsupported_parameter",
            }
        },
    )


def _unsupported(body: ResponsesRequest) -> Optional[JSONResponse]:
    if body.tools:
        return _error(
            400,
            "Function tools are not supported by this server's /v1/responses endpoint "
            "(text only). Remove 'tools' from the request.",
            "tools",
        )
    if isinstance(body.tool_choice, dict) or body.tool_choice == "required":
        return _error(
            400, "tool_choice requires tools, and function tools are not supported.", "tool_choice"
        )
    for param in ("previous_response_id", "conversation"):
        if getattr(body, param):
            return _error(
                400,
                f"{param} is not supported: this server stores no responses. "
                "Send the full conversation in 'input' instead.",
                param,
            )
    return None


def _to_messages(body: ResponsesRequest) -> List[Message]:
    items = (
        [ResponseInputMessage(role="user", content=body.input)]
        if isinstance(body.input, str)
        else body.input
    )
    messages = [Message(role="system", content=body.instructions)] if body.instructions else []
    messages += [Message(role=item.role, content=item.content) for item in items]
    return messages


def _usage(chunks: List[Dict[str, Any]], prompt: str, text: str) -> ResponseUsage:
    counts = sdk_usage.openai_usage_counts(
        chunks, MessageAdapter.estimate_tokens(prompt), MessageAdapter.estimate_tokens(text)
    )
    real = sdk_usage.extract_usage(chunks) or {}
    return ResponseUsage(
        input_tokens=counts["prompt_tokens"],
        input_tokens_details={
            "cached_tokens": real.get("cache_read_input_tokens", 0),
            "cache_write_tokens": real.get("cache_creation_input_tokens", 0),
        },
        output_tokens=counts["completion_tokens"],
        output_tokens_details={"reasoning_tokens": 0},
        total_tokens=counts["total_tokens"],
    )


def _finished(body, resp_id, msg_id, created, chunks, prompt, text) -> ResponseObject:
    """The final response object for a run that produced no error result."""
    reason = _INCOMPLETE_REASONS.get(sdk_usage.extract_stop_reason(chunks))
    status = "incomplete" if reason else "completed"
    return _response(
        body,
        resp_id,
        created,
        status,
        output=[
            ResponseOutputMessage(id=msg_id, status=status, content=[ResponseOutputText(text=text)])
        ],
        output_text=text,
        incomplete_details={"reason": reason} if reason else None,
        usage=_usage(chunks, prompt, text),
        model=sdk_usage.extract_model(chunks, body.model),
    )


def _response(
    body: ResponsesRequest, resp_id: str, created: int, status: str, model=None, **fields
):
    return ResponseObject(
        id=resp_id,
        created_at=created,
        status=status,
        model=model or body.model,
        instructions=body.instructions,
        max_output_tokens=body.max_output_tokens,
        metadata=body.metadata,
        temperature=body.temperature,
        top_p=body.top_p,
        user=body.user,
        **fields,
    )


@router.post("/v1/responses")
@rate_limit_endpoint("chat")
async def create_response(
    request_body: ResponsesRequest,
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """OpenAI Responses API compatible endpoint (text only, stateless)."""
    from claude_relay import main

    await verify_api_key(request, credentials)

    rejection = _unsupported(request_body)
    if rejection:
        return rejection

    auth_valid, auth_info = main.validate_claude_code_auth()
    if not auth_valid:
        error_detail = {
            "message": "Claude Code authentication failed",
            "errors": auth_info.get("errors", []),
            "method": auth_info.get("method", "none"),
            "help": "Check /v1/auth/status for detailed authentication information",
        }
        raise HTTPException(status_code=503, detail=error_detail)

    if request_body.max_output_tokens is not None:
        logger.info("max_output_tokens is not enforced by Claude Code; ignoring it")
    if request_body.reasoning:
        logger.info(f"reasoning={request_body.reasoning} is accepted but not applied")

    prompt, system_prompt = MessageAdapter.messages_to_prompt(_to_messages(request_body))
    # Apply MODEL_ALIASES and reject clearly non-Claude names, as chat completions does
    model = ParameterValidator.resolve_model(request_body.model)
    if model is None:
        return main.model_error_response(main.unknown_model_message(request_body.model))
    completion_args = dict(
        prompt=prompt,
        system_prompt=system_prompt,
        model=model,
        max_turns=1,
        disallowed_tools=CLAUDE_TOOLS,
    )
    resp_id = f"resp_{os.urandom(12).hex()}"
    msg_id = f"msg_{os.urandom(12).hex()}"
    created = int(time.time())

    if request_body.stream:
        return StreamingResponse(
            _stream(main.claude_cli, request_body, completion_args, resp_id, msg_id, created),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    try:
        chunks = [
            chunk async for chunk in main.claude_cli.run_completion(**completion_args, stream=False)
        ]
    except Exception as e:
        logger.error(f"Responses API error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    sdk_error = sdk_usage.result_error(chunks)
    if sdk_error and sdk_error.get("code") == "model_not_found":
        return main.model_error_response(sdk_error["message"])
    main._raise_for_sdk_error(chunks)  # 401 for a Claude Code login failure, else 429/502

    text = main.claude_cli.parse_claude_message(chunks)
    if text is None:
        raise HTTPException(status_code=500, detail="No response from Claude Code")
    return _finished(request_body, resp_id, msg_id, created, chunks, prompt, text)


async def _stream(
    claude_cli, body: ResponsesRequest, completion_args, resp_id, msg_id, created
) -> AsyncGenerator[str, None]:
    """Yield the Responses API SSE event sequence for one run."""
    sequence = itertools.count()

    def event(event_type: str, **data) -> str:
        data = {"type": event_type, "sequence_number": next(sequence), **data}
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    def text_part(text: str) -> Dict[str, Any]:
        return ResponseOutputText(text=text).model_dump()

    started = _response(body, resp_id, created, "in_progress").model_dump()
    added = ResponseOutputMessage(id=msg_id, status="in_progress", content=[]).model_dump()
    yield event("response.created", response=started)
    yield event("response.in_progress", response=started)
    yield event("response.output_item.added", output_index=0, item=added)
    part_ids = {"item_id": msg_id, "output_index": 0, "content_index": 0}
    yield event("response.content_part.added", **part_ids, part=text_part(""))

    chunks: List[Dict[str, Any]] = []
    deltas: List[str] = []
    sdk_error = None
    try:
        async for chunk in claude_cli.run_completion(**completion_args, stream=True):
            chunks.append(chunk)
            # The ResultMessage repeats the whole text, so only assistant chunks are streamed
            if sdk_usage.find_result_message([chunk]):
                continue
            # The CLI's own error text (login, model, rate limit...); reported as response.failed
            if chunk.get("error"):
                continue
            text = claude_cli.parse_claude_message([chunk])
            if text:
                deltas.append(text)
                yield event("response.output_text.delta", **part_ids, delta=text, logprobs=[])
        from claude_relay import main

        sdk_error = main._sdk_error(chunks)  # also records/clears the last login failure
    except Exception as e:
        logger.error(f"Responses streaming error: {e}")
        sdk_error = {"status": 500, "message": str(e)}

    if sdk_error:
        if sdk_error.get("auth_failure"):
            code = "claude_cli_not_authenticated"
        elif sdk_error.get("code") == "model_not_found":
            code = "model_not_found"
        else:
            code = "rate_limit_exceeded" if sdk_error["status"] == 429 else "server_error"
        failed = _response(
            body, resp_id, created, "failed", error={"code": code, "message": sdk_error["message"]}
        )
        yield event("response.failed", response=failed.model_dump())
        return

    text = "".join(deltas)
    final = _finished(body, resp_id, msg_id, created, chunks, completion_args["prompt"], text)
    yield event("response.output_text.done", **part_ids, text=text, logprobs=[])
    yield event("response.content_part.done", **part_ids, part=text_part(text))
    yield event("response.output_item.done", output_index=0, item=final.output[0].model_dump())
    final_event = "response.incomplete" if final.status == "incomplete" else "response.completed"
    yield event(final_event, response=final.model_dump())
