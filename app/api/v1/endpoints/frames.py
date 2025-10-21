"""
Frame processing endpoints
"""
import base64
from datetime import datetime, timezone
from typing import Dict, Any, Set

from fastapi import APIRouter, HTTPException, status, Depends

from app.models.schemas import FrameRequest, FrameResponse, AttendanceUpdate
from app.services.session_manager import session_manager
from app.services.face_engine import face_engine
from app.services.tracker import face_tracker
from app.services.notifier import backend_notifier
from app.services.recognition_validator import get_recognition_validator
from app.core.security import get_current_user
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/sessions/{session_id}/frames", response_model=FrameResponse)
async def process_frame(
    session_id: str,
    request: FrameRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Xử lý frame ảnh
    
    Args:
        session_id: ID của session
        request: Frame data và metadata
        current_user: Thông tin user từ JWT
        
    Returns:
        Kết quả xử lý frame
        
    Raises:
        HTTPException: Nếu session không tồn tại hoặc có lỗi xử lý
    """
    request_logger = logger.bind(
        session_id=session_id,
        user_id=current_user.get("sub"),
        client_seq=request.client_seq
    )
    
    try:
        # Kiểm tra session tồn tại và lấy SessionData thực (với embeddings)
        session = await session_manager.get_session_data(session_id)
        if not session:
            request_logger.warning("Session not found")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        if session.status != "active":
            request_logger.warning("Session not active", status=session.status)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Session is not active: {session.status}"
            )
        
        request_logger.info("Processing frame")
        
        # Decode frame data (optional - có thể bỏ qua trong stub)
        frame_data = None
        if request.frame_base64:
            try:
                frame_data = base64.b64decode(request.frame_base64)
                request_logger.debug("Frame decoded", frame_size=len(frame_data))
            except Exception as e:
                request_logger.warning("Failed to decode frame", error=str(e))
                # Tiếp tục xử lý với frame_data = None
        
        current_time = datetime.now(timezone.utc)
        
        # Face detection
        detections = await face_engine.detect_faces(frame_data)
        request_logger.debug("Face detection completed", num_faces=len(detections))
        
        # Face recognition - Use session embeddings from VRAM
        # session.gallery_embeddings: torch.Tensor shape (N, 512) on GPU
        # session.gallery_labels: List[str] student codes
        detections = await face_engine.recognize_faces(
            detections, 
            frame_data,
            gallery_embeddings=session.gallery_embeddings,
            gallery_labels=session.gallery_labels
        )
        
        # Face tracking - gán track_id cho mỗi detection
        detections = await face_tracker.update(detections)
        
        # **CƠ CHẾ MỚI: Cập nhật recognition history vào validator**
        await face_engine.update_recognition_history(detections, current_time)
        
        # **CƠ CHẾ MỚI: Chỉ lấy sinh viên đã được VALIDATED**
        # (pass đủ điều kiện: confirmation threshold, avg confidence, success rate)
        validated_student_ids: Set[str] = await face_engine.get_validated_students(current_time)
        
        # Tăng frame counter
        await session_manager.increment_frame_count(session_id)
        
        # Chuẩn bị response
        # recognized_student_ids: tất cả sinh viên được nhận diện trong frame này
        recognized_student_ids = [
            d.student_id for d in detections 
            if d.student_id is not None
        ]
        
        response = FrameResponse(
            session_id=session_id,
            timestamp=request.timestamp,
            client_seq=request.client_seq,
            processed_at=current_time,
            detections=detections,
            recognized_student_ids=recognized_student_ids,
            total_faces=len(detections),
            callback_sent=False
        )
        
        # **CƠ CHẾ MỚI: Chỉ gửi callback cho sinh viên đã được VALIDATED**
        # Không còn tin ngay vào kết quả nhận diện đầu tiên
        if validated_student_ids:
            attendance_data = AttendanceUpdate(
                session_id=session_id,
                class_id=session.class_id,
                timestamp=request.timestamp,
                recognized_students=list(validated_student_ids),
                total_faces_detected=len(detections)
            )
            
            # Gửi callback async (không chờ kết quả)
            callback_success = await _send_callback_async(
                session.backend_callback_url,
                attendance_data,
                session_id,
                request_logger
            )
            response.callback_sent = callback_success
            
            request_logger.info(
                "Validated students confirmed",
                validated_students=list(validated_student_ids),
                validation_passed=True
            )
        
        request_logger.info(
            "Frame processed successfully",
            total_faces=len(detections),
            recognized_students=len(recognized_student_ids),
            validated_students=len(validated_student_ids),
            callback_sent=response.callback_sent
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        request_logger.error("Frame processing failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Frame processing failed: {str(e)}"
        )


async def _send_callback_async(
    callback_url: str,
    attendance_data: AttendanceUpdate,
    session_id: str,
    request_logger
) -> bool:
    """
    Gửi callback async với error handling
    
    TODO: Implement background task hoặc message queue
    """
    try:
        async with backend_notifier:
            success = await backend_notifier.send_attendance_update_with_retry(
                callback_url, attendance_data, session_id
            )
            
            if success:
                request_logger.info("Callback sent successfully")
            else:
                request_logger.error("Callback failed after retries")
            
            return success
            
    except Exception as e:
        request_logger.error("Callback error", error=str(e))
        return False
