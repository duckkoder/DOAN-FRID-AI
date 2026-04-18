"""
RAG Chat Endpoint – Phase 4
Provides a streaming SSE endpoint that Backend proxies to Frontend.
Auth: Bearer JWT issued by Backend (same secret).
"""
from __future__ import annotations

import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import jwt

from app.core.config import settings
from app.services.rag.chat_engine import stream_answer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rag")

_bearer = HTTPBearer()


# ---------------------------------------------------------------------------
# Auth: validate Backend-issued JWT to extract user_id + role
# ---------------------------------------------------------------------------

def _get_current_user(credentials: HTTPAuthorizationCredentials = Depends(_bearer)) -> dict:
    token = credentials.credentials
    try:
        payload = jwt.decode(
            token,
            settings.BACKEND_JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
        )
        user_id = payload.get("sub") or payload.get("user_id")
        if not user_id:
            raise ValueError("No user_id in token")
        return {"user_id": int(user_id), "role": payload.get("role", "student")}
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Invalid token: {e}")


# ---------------------------------------------------------------------------
# Request schema
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    class_id: int           # Integer class ID
    question: str           # User question (raw)
    document_ids: List[str] # UUIDs of selected documents


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post(
    "/chat",
    summary="Stream RAG chat answer (SSE)",
)
async def rag_chat(
    body: ChatRequest,
    user: dict = Depends(_get_current_user),
):
    """
    Streams a Server-Sent Events response.

    Each SSE event is one of:
    - `data: <token>\\n\\n`          – a text token from Gemini
    - `data: [SOURCES] {...}\\n\\n`  – JSON with page citations
    - `data: [DONE]\\n\\n`           – stream finished
    - `data: [ERROR] ...\\n\\n`      – error occurred

    The Backend API should proxy this stream transparently to the Frontend.
    """
    if not body.question.strip():
        raise HTTPException(status_code=422, detail="question cannot be empty")
    if not body.document_ids:
        raise HTTPException(status_code=422, detail="document_ids cannot be empty")

    user_id = user["user_id"]

    async def _event_generator():
        async for chunk in stream_answer(
            user_id=user_id,
            class_id=body.class_id,
            question=body.question,
            document_ids=body.document_ids,
        ):
            yield chunk

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable Nginx buffering
        },
    )


@router.get(
    "/chat/history",
    summary="Get chat history for a class session",
)
async def get_chat_history(
    class_id: int,
    user: dict = Depends(_get_current_user),
):
    """Return all messages in the user's session for a given class."""
    import psycopg2
    from psycopg2.extras import RealDictCursor

    conn = psycopg2.connect(settings.DATABASE_URL)
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                """
                SELECT cm.role, cm.content, cm.created_at
                FROM chat_messages cm
                JOIN chat_sessions cs ON cs.id = cm.session_id
                WHERE cs.student_id = %s AND cs.course_id = %s::uuid
                ORDER BY cm.created_at ASC
                """,
                (user["user_id"], str(class_id)),
            )
            messages = [dict(r) for r in cur.fetchall()]
        return {"success": True, "data": {"messages": messages}}
    except Exception as e:
        logger.error(f"History fetch error: {e}")
        raise HTTPException(status_code=500, detail="Could not load chat history")
    finally:
        conn.close()


@router.delete(
    "/chat/history",
    summary="Clear chat history for a class session",
)
async def clear_chat_history(
    class_id: int,
    user: dict = Depends(_get_current_user),
):
    """Delete all messages in the user's session for a given class."""
    import psycopg2
    conn = psycopg2.connect(settings.DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM chat_sessions
                WHERE student_id = %s AND course_id = %s::uuid
                """,
                (user["user_id"], str(class_id)),
            )
        conn.commit()
        return {"success": True, "message": "Lịch sử chat đã được xóa"}
    except Exception as e:
        logger.error(f"History clear error: {e}")
        raise HTTPException(status_code=500, detail="Could not clear chat history")
    finally:
        conn.close()
