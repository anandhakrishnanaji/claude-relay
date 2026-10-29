"""Shared SSE response for the chat and messages streams."""

from typing import AsyncIterator

from fastapi.responses import StreamingResponse


async def start_stream(events: AsyncIterator[str]) -> StreamingResponse:
    """Stream SSE events, but wait for the first one so an unknown model is a plain 400."""
    first = await events.__anext__()  # a CompletionError here becomes a normal HTTP error

    async def all_events():
        yield first
        async for event in events:
            yield event

    return StreamingResponse(
        all_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )
