"""
Session management endpoints
"""
from fastapi import APIRouter, HTTPException, status
from datetime import datetime, timezone

from app.models.schemas import (
    SessionCreateRequest, 
    SessionResponse, 
    ErrorResponse
)
from app.services.session_manager import session_manager
from app.services.storage import s3_storage
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/sessions", response_model=SessionResponse)
async def create_session(request: SessionCreateRequest):
    """
    Tạo session mới
    
    Args:
        request: Thông tin tạo session
        
    Returns:
        Thông tin session đã tạo
        
    Raises:
        HTTPException: Nếu có lỗi khi tạo session
    """
    try:
        # Kiểm tra S3 object có tồn tại không (stub)
        exists = await s3_storage.check_object_exists(request.s3)
        if not exists:
            logger.warning(
                "S3 embeddings object not found (STUB - continuing anyway)",
                bucket=request.s3.bucket,
                key=request.s3.key
            )
        
        # Tạo session
        session = await session_manager.create_session(request)
        
        logger.info(
            "Session created successfully",
            session_id=session.session_id,
            class_id=request.class_id
        )
        
        return session
        
    except Exception as e:
        logger.error("Failed to create session", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create session: {str(e)}"
        )


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """
    Lấy thông tin session
    
    Args:
        session_id: ID của session
        
    Returns:
        Thông tin session
        
    Raises:
        HTTPException: Nếu session không tồn tại
    """
    session = await session_manager.get_session(session_id)
    
    if not session:
        logger.warning("Session not found", session_id=session_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    return session


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    Xóa session
    
    Args:
        session_id: ID của session
        
    Returns:
        Success message
        
    Raises:
        HTTPException: Nếu session không tồn tại
    """
    success = await session_manager.delete_session(session_id)
    
    if not success:
        logger.warning("Session not found for deletion", session_id=session_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    logger.info("Session deleted successfully", session_id=session_id)
    
    return {"message": "Session deleted successfully", "session_id": session_id}
