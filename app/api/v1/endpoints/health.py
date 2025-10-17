"""
Health check endpoint
"""
from datetime import datetime, timezone
from fastapi import APIRouter

from app.models.schemas import HealthResponse
from app.core.config import settings
from app.services.session_manager import session_manager

router = APIRouter()


@router.get("/healthz", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    
    Returns:
        Thông tin health của service
    """
    active_sessions = await session_manager.get_active_sessions_count()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc),
        version=settings.APP_VERSION,
        active_sessions=active_sessions
    )
