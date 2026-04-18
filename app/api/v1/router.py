"""
API v1 Router - Tập trung tất cả endpoints
"""
from fastapi import APIRouter
from app.api.v1.endpoints import health, sessions, frames, detection, registration
from app.api.v1.endpoints import rag_ingest, rag_chat

api_router = APIRouter()

# Include tất cả endpoints
api_router.include_router(health.router, tags=["Health"])
api_router.include_router(detection.router, tags=["Detection"])
api_router.include_router(sessions.router, tags=["Sessions"])
api_router.include_router(frames.router, tags=["Frames"])
api_router.include_router(registration.router, tags=["Registration"])

# RAG endpoints
api_router.include_router(rag_ingest.router, tags=["RAG - Ingestion"])
api_router.include_router(rag_chat.router, tags=["RAG - Chat"])