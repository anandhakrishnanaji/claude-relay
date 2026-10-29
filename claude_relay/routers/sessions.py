"""/v1/sessions endpoints: list, inspect and delete chat sessions."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials

from claude_relay import session_manager as session_manager_module
from claude_relay.auth import security, verify_api_key
from claude_relay.models import SessionListResponse
from claude_relay.session_manager import session_manager

router = APIRouter()


@router.get("/v1/sessions/stats")
async def get_session_stats(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Get session manager statistics."""
    await verify_api_key(request, credentials)
    stats = session_manager.get_stats()
    return {
        "session_stats": stats,
        "cleanup_interval_minutes": session_manager.cleanup_interval_minutes,
        "default_ttl_minutes": session_manager_module.SESSION_TTL_MINUTES,
    }


@router.get("/v1/sessions")
async def list_sessions(
    request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """List all active sessions."""
    await verify_api_key(request, credentials)
    sessions = session_manager.list_sessions()
    return SessionListResponse(sessions=sessions, total=len(sessions))


@router.get("/v1/sessions/{session_id}")
async def get_session(
    session_id: str,
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Get information about a specific session."""
    await verify_api_key(request, credentials)
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return session.to_session_info()


@router.delete("/v1/sessions/{session_id}")
async def delete_session(
    session_id: str,
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
):
    """Delete a specific session."""
    await verify_api_key(request, credentials)
    deleted = session_manager.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")

    return {"message": f"Session {session_id} deleted successfully"}
