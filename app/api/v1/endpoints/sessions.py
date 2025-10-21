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
    Tạo session mới và load embeddings vào VRAM.
    
    Flow:
    1. Backend gửi danh sách student_codes (100 students)
    2. AI-Service query embeddings từ pgvector (1 query duy nhất)
    3. Load embeddings vào VRAM (GPU memory)
    4. Return session info
    
    Args:
        request: SessionCreateRequest với student_codes
        
    Returns:
        Thông tin session đã tạo với embeddings loaded
        
    Raises:
        HTTPException: Nếu có lỗi khi tạo session hoặc load embeddings
    """
    try:
        # Validate request
        if not request.student_codes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="student_codes is required and cannot be empty"
            )
        
        logger.info(
            "Creating session with embeddings",
            class_id=request.class_id,
            student_count=len(request.student_codes)
        )
        
        # Tạo session và load embeddings vào VRAM
        session = await session_manager.create_session(request)
        
        if not session.embeddings_loaded:
            logger.error(
                "Session created but embeddings not loaded",
                session_id=session.session_id
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to load embeddings into VRAM"
            )
        
        logger.info(
            "Session created successfully with embeddings in VRAM",
            session_id=session.session_id,
            class_id=request.class_id,
            student_count=len(request.student_codes)
        )
        
        return session
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create session", error=str(e), exc_info=True)
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
