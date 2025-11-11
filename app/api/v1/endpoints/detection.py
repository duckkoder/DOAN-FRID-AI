"""
Detection endpoint - Phát hiện và nhận diện khuôn mặt từ ảnh đơn
"""
import base64
import time
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
import numpy as np
import cv2

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


class FaceInfo(BaseModel):
    """Thông tin một khuôn mặt"""
    bbox: List[float] = Field(..., description="[x1, y1, x2, y2]")
    confidence: float = Field(..., description="Độ tin cậy detection")
    person_name: str = Field("Unknown", description="Tên người (MSSV nếu nhận diện được, 'Unknown' nếu không)")
    recognition_confidence: Optional[float] = Field(None, description="Độ tin cậy recognition")


class DetectRequest(BaseModel):
    """Request"""
    image_base64: str = Field(..., description="Ảnh base64")
    session_id: Optional[str] = Field(None, description="Session ID (optional - để dùng embeddings từ session)")


class DetectResponse(BaseModel):
    """Response"""
    total_faces: int
    recognized_count: int
    faces: List[FaceInfo]
    processing_time_ms: float


@router.post("/detect", response_model=DetectResponse)
async def detect_faces(request: DetectRequest):
    """
    Phát hiện và nhận diện khuôn mặt
    
    **Request:**
    ```json
    {
        "image_base64": "iVBORw0KGgo..."
    }
    ```
    
    **Response:**
    ```json
    {
        "total_faces": 2,
        "recognized_count": 1,
        "faces": [
            {
                "bbox": [100, 150, 250, 350],
                "confidence": 0.95,
                "person_name": "20200001",
                "recognition_confidence": 0.87
            },
            {
                "bbox": [300, 200, 450, 400],
                "confidence": 0.92,
                "person_name": "Unknown",
                "recognition_confidence": null
            }
        ],
        "processing_time_ms": 234.5
    }
    ```
    """
    start_time = time.time()
    
    try:
        # Decode image
        try:
            image_data = base64.b64decode(request.image_base64)
            image_array = np.frombuffer(image_data, dtype=np.uint8)
            image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image_bgr is None:
                raise ValueError("Không decode được ảnh")
            
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Ảnh không hợp lệ: {str(e)}"
            )
        
        # Get services
        try:
            from app.services.face_detection_service import get_face_detection_service
            from app.services.face_recognition_service import get_face_recognition_service
            
            detector = get_face_detection_service()
            recognizer = get_face_recognition_service()
        except RuntimeError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Service chưa khởi tạo: {str(e)}"
            )
        
        # Detect faces
        detections, crops, _ = await detector.detect_faces_async(
            image_rgb,
            return_crops=True
        )
        
        # Handle None results
        if detections is None:
            detections = []
        if crops is None:
            crops = []
        
        logger.info(f"Detected {len(detections)} faces")
        
        # Get session embeddings if session_id provided
        gallery_embeddings = None
        gallery_labels = None
        
        if request.session_id:
            try:
                from app.services.session_manager import session_manager
                session_data = await session_manager.get_session_data(request.session_id)
                
                if session_data:
                    gallery_embeddings = session_data.gallery_embeddings
                    gallery_labels = session_data.gallery_labels
                    logger.info(f"Using session embeddings: {len(gallery_labels)} students")
                else:
                    logger.warning(f"Session {request.session_id} not found, using internal database")
            except Exception as e:
                logger.warning(f"Failed to load session embeddings: {e}")
        
        # Recognize faces
        faces = []
        recognized_count = 0
        
        for idx, (det, crop) in enumerate(zip(detections, crops)):
            face_info = FaceInfo(
                bbox=[float(x) for x in det.bbox],
                confidence=float(det.confidence),
                person_name="Unknown",  # ✅ Default: "Unknown" thay vì None
                recognition_confidence=None
            )
            
            # Try to recognize if crop available
            if crop is not None:
                # Use session embeddings if available, otherwise use internal database
                can_recognize = (gallery_embeddings is not None and gallery_labels is not None) or recognizer.has_database
                
                if can_recognize:
                    try:
                        identity = await recognizer.identify_async(
                            crop,
                            gallery_embeddings=gallery_embeddings,
                            gallery_labels=gallery_labels
                        )
                        
                        # ===== 🔍 DEBUG LOGGING =====
                        if identity:
                            person = identity.get('person', 'Unknown')
                            best_candidate = identity.get('best_candidate', 'N/A')
                            rejection_reason = identity.get('rejection_reason', 'N/A')
                            
                            logger.info(
                                f"[DETECTION] Face #{idx+1} Recognition Result",
                                person=person,
                                best_candidate=best_candidate,
                                rejection_reason=rejection_reason,
                                confidence=f"{identity.get('confidence', 0):.3f}",
                                vote_ratio=f"{identity.get('vote_ratio', 0):.3f}",
                                distance=f"{identity.get('distance', 0):.3f}",
                                threshold=f"{identity.get('threshold', 0):.3f}"
                            )
                        # ===== END DEBUG =====
                        
                        if identity and identity.get('person') != 'Unknown':
                            # ✅ Nhận diện thành công
                            face_info.person_name = identity.get('person')
                            face_info.recognition_confidence = float(identity.get('confidence', 0.0))
                            recognized_count += 1
                            logger.info(f"[DETECTION] ✅ Face #{idx+1} ACCEPTED as {identity.get('person')}")
                        else:
                            # ✅ Bị reject hoặc Unknown → Giữ nguyên "Unknown"
                            face_info.person_name = "Unknown"
                            logger.info(f"[DETECTION] ❌ Face #{idx+1} REJECTED (Unknown)")
                    except Exception as e:
                        # ✅ Lỗi recognition → Giữ nguyên "Unknown"
                        face_info.person_name = "Unknown"
                        logger.warning(f"Recognition failed: {e}")
            # else: crop is None → Giữ nguyên "Unknown" từ default
            
            faces.append(face_info)
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"Recognized {recognized_count}/{len(detections)} faces in {processing_time:.1f}ms")
        
        return DetectResponse(
            total_faces=len(detections),
            recognized_count=recognized_count,
            faces=faces,
            processing_time_ms=round(processing_time, 2)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi xử lý: {str(e)}"
        )

