"""
RAG Chat Endpoint.

Backend is responsible for user authentication, tenant API key lookup, document
authorization, retrieval, and history before proxying requests here.
"""
from __future__ import annotations

import logging
from typing import List, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.services.rag.chat_engine import stream_answer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/rag")


class ChatRequest(BaseModel):
    class_id: int
    question: str
    document_ids: List[str]
    context_chunks: List[dict] = []
    creativity_mode: Literal["strict", "expanded"] = "strict"
    detail_level: Literal["brief", "normal", "detailed"] = "normal"
    gemini_api_key: str


@router.post(
    "/chat",
    summary="Stream RAG chat answer (SSE)",
)
async def rag_chat(body: ChatRequest):
    """
    Streams a Server-Sent Events response.
    History management is handled by the Backend.
    """
    if not body.question.strip():
        raise HTTPException(status_code=422, detail="question cannot be empty")
    if not body.document_ids:
        raise HTTPException(status_code=422, detail="document_ids cannot be empty")
    if not body.gemini_api_key.strip():
        raise HTTPException(status_code=422, detail="gemini_api_key is required")

    async def _event_generator():
        async for chunk in stream_answer(
            user_id=0,
            class_id=body.class_id,
            question=body.question,
            document_ids=body.document_ids,
            creativity_mode=body.creativity_mode,
            detail_level=body.detail_level,
            gemini_api_key=body.gemini_api_key,
            context_chunks=body.context_chunks,
        ):
            yield chunk

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# Note: GET /chat/history and DELETE /chat/history are handled by Backend.
