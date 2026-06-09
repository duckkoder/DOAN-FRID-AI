"""
Session management endpoints
"""
from fastapi import APIRouter, HTTPException, status
from datetime import datetime, timezone

from app.models.schemas import (
    SessionCreateRequest, 
    SessionResponse
)
from app.services.session_manager import session_manager
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/sessions", response_model=SessionResponse)
async def create_session(request: SessionCreateRequest):
    """
    Táº¡o session má»›i vÃ  load embeddings vÃ o VRAM.
    
    Flow:
    1. Backend sends student_codes and face_embeddings from tenant DB
    2. AI-Service loads embeddings into VRAM (GPU memory)
    3. Return session info
    
    Args:
        request: SessionCreateRequest vá»›i student_codes
        
    Returns:
        ThÃ´ng tin session Ä‘Ã£ táº¡o vá»›i embeddings loaded
        
    Raises:
        HTTPException: Náº¿u cÃ³ lá»—i khi táº¡o session hoáº·c load embeddings
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
        
        # Táº¡o session vÃ  load embeddings vÃ o VRAM
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
    Láº¥y thÃ´ng tin session
    
    Args:
        session_id: ID cá»§a session
        
    Returns:
        ThÃ´ng tin session
        
    Raises:
        HTTPException: Náº¿u session khÃ´ng tá»“n táº¡i
    """
    session = await session_manager.get_session(session_id)
    
    if not session:
        logger.warning("Session not found", session_id=session_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found"
        )
    
    return session


@router.get("/sessions/{session_id}/face-crops")
async def get_session_face_crops(session_id: str):
    """
    Láº¥y táº¥t cáº£ áº£nh face crops cá»§a students Ä‘Ã£ validated trong session.
    Endpoint nÃ y Ä‘Æ°á»£c Backend gá»i khi end_session Ä‘á»ƒ upload áº£nh lÃªn S3.
    
    Args:
        session_id: ID cá»§a session
        
    Returns:
        List of {student_code, face_crop_base64}
        
    Raises:
        HTTPException: Náº¿u session khÃ´ng tá»“n táº¡i
    """
    import base64
    import cv2
    
    # Láº¥y crops tá»« session memory
    crops_dict = await session_manager.get_validated_students_crops(session_id)
    
    if not crops_dict:
        logger.warning("No face crops found for session", session_id=session_id)
        return {"session_id": session_id, "face_crops": []}
    
    # Convert numpy arrays to base64
    face_crops_data = []
    for student_code, face_crop_rgb in crops_dict.items():
        try:
            # Convert RGB to BGR for cv2
            face_crop_bgr = cv2.cvtColor(face_crop_rgb, cv2.COLOR_RGB2BGR)
            
            # Encode to JPEG bytes
            success, buffer = cv2.imencode('.jpg', face_crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            if success:
                # Convert to base64 string
                face_crop_base64 = base64.b64encode(buffer).decode('utf-8')
                
                face_crops_data.append({
                    "student_code": student_code,
                    "face_crop_base64": face_crop_base64
                })
                
                logger.debug(f"Encoded face crop for {student_code}: {len(face_crop_base64)} bytes")
            else:
                logger.error(f"Failed to encode face crop for {student_code}")
                
        except Exception as e:
            logger.error(f"Error encoding face crop for {student_code}: {e}")
            continue
    
    logger.info(
        f"Returning {len(face_crops_data)} face crops for session",
        session_id=session_id
    )
    
    return {
        "session_id": session_id,
        "face_crops": face_crops_data
    }


@router.get("/sessions/{session_id}/spoof-faces")
async def get_session_spoof_faces(session_id: str):
    """
    Láº¥y táº¥t cáº£ áº£nh spoof faces phÃ¡t hiá»‡n trong session.
    Endpoint nÃ y Ä‘Æ°á»£c Backend gá»i khi end_session Ä‘á»ƒ upload áº£nh giáº£ máº¡o lÃªn S3.
    
    Args:
        session_id: ID cá»§a session
        
    Returns:
        List of {face_crop_base64, spoofing_type, spoofing_confidence, detected_at, frame_count}
        
    Raises:
        HTTPException: Náº¿u session khÃ´ng tá»“n táº¡i
    """
    import base64
    import cv2
    
    # Láº¥y spoof crops tá»« session memory
    spoof_crops = await session_manager.get_spoof_faces_crops(session_id)
    
    if not spoof_crops:
        logger.info("No spoof faces found for session", session_id=session_id)
        return {"session_id": session_id, "spoof_faces": []}
    
    # Convert numpy arrays to base64
    spoof_faces_data = []
    for idx, spoof_data in enumerate(spoof_crops):
        try:
            # Spoof crops are stored compressed in memory.
            face_crop = spoof_data.get_face_crop()
            face_crop_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
            
            # Encode to JPEG bytes
            success, buffer = cv2.imencode('.jpg', face_crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            
            if success:
                # Convert to base64 string
                face_crop_base64 = base64.b64encode(buffer).decode('utf-8')
                
                spoof_faces_data.append({
                    "face_crop_base64": face_crop_base64,
                    "spoofing_type": spoof_data.spoofing_type,
                    "spoofing_confidence": spoof_data.spoofing_confidence,
                    "detected_at": spoof_data.detected_at.isoformat(),
                    "frame_count": spoof_data.frame_count
                })
                
                logger.debug(f"Encoded spoof face #{idx}: {len(face_crop_base64)} bytes")
            else:
                logger.error(f"Failed to encode spoof face #{idx}")
                
        except Exception as e:
            logger.error(f"Error encoding spoof face #{idx}: {e}")
            continue
    
    logger.info(
        f"Returning {len(spoof_faces_data)} spoof faces for session",
        session_id=session_id
    )
    
    return {
        "session_id": session_id,
        "spoof_faces": spoof_faces_data
    }


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """
    XÃ³a session
    
    Args:
        session_id: ID cá»§a session
        
    Returns:
        Success message
        
    Raises:
        HTTPException: Náº¿u session khÃ´ng tá»“n táº¡i
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
