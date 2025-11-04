"""
Frame processing endpoints with WebSocket support
"""
import base64
import time
from datetime import datetime, timezone
from typing import Dict, Any, Set

from fastapi import APIRouter, HTTPException, status, WebSocket, WebSocketDisconnect, Query

from app.models.schemas import FrameRequest, FrameResponse, AttendanceUpdate
from app.services.session_manager import session_manager
from app.services.face_engine import get_face_engine
from app.services.notifier import backend_notifier
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

# Get face engine instance (initialized in main.py lifespan)
face_engine = None

def get_engine():
    """Get face engine instance"""
    global face_engine
    if face_engine is None:
        try:
            face_engine = get_face_engine()
        except RuntimeError as e:
            logger.error(f"Face engine not initialized: {e}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Face recognition service not available"
            )
    return face_engine


@router.post("/sessions/{session_id}/frames", response_model=FrameResponse)
async def process_frame(
    session_id: str,
    request: FrameRequest
):
    """
    Xử lý frame ảnh
    
    Note: This endpoint is for testing/development. Production uses WebSocket with JWT auth.
    
    Args:
        session_id: ID của session
        request: Frame data và metadata
        
    Returns:
        Kết quả xử lý frame
        
    Raises:
        HTTPException: Nếu session không tồn tại hoặc có lỗi xử lý
    """
    request_logger = logger.bind(
        session_id=session_id,
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
        
        # ✅ Face tracking - Sử dụng per-session tracker
        if session.face_tracker:
            detections = await session.face_tracker.update(detections)
        else:
            request_logger.warning("No face_tracker in session, skipping tracking")
        
        # ✅ **CƠ CHẾ MỚI: Cập nhật recognition history vào per-session validator**
        if session.recognition_validator:
            for detection in detections:
                if detection.track_id and detection.student_id:
                    await session.recognition_validator.add_recognition(
                        track_id=detection.track_id,
                        student_code=detection.student_id,  # Using student_code
                        confidence=detection.recognition_confidence or detection.confidence,
                        timestamp=current_time
                    )
        
        # ✅ **CƠ CHẾ MỚI: Chỉ lấy sinh viên đã được VALIDATED**
        validated_student_ids: Set[str] = set()
        if session.recognition_validator:
            for detection in detections:
                if detection.track_id:
                    validation_result = await session.recognition_validator.validate_recognition(
                        track_id=detection.track_id,
                        current_time=current_time
                    )
                    if validation_result:
                        validated_student_ids.add(validation_result["student_code"])  # Changed from student_id
        
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
            from app.models.schemas import ValidatedStudent
            
            # Create ValidatedStudent objects (HTTP endpoint doesn't have tracking, use defaults)
            validated_students_data = [
                ValidatedStudent(
                    student_code=student_id,
                    student_name=student_id,  # HTTP endpoint doesn't have student names
                    track_id=0,  # No tracking in HTTP endpoint
                    avg_confidence=0.0,  # No multi-frame stats
                    frame_count=1,  # Single frame
                    recognition_count=1,  # Single recognition
                    validation_passed_at=request.timestamp
                )
                for student_id in validated_student_ids
            ]
            
            attendance_data = AttendanceUpdate(
                session_id=session_id,
                validated_students=validated_students_data,
                timestamp=request.timestamp
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
    """
    try:
        # ✅ Create new notifier instance with context manager
        from app.services.notifier import BackendNotifier
        
        async with BackendNotifier() as notifier:
            success = await notifier.send_attendance_update_with_retry(
                callback_url, attendance_data, session_id
            )
            
            if success:
                request_logger.info("Callback sent successfully to Backend")
            else:
                request_logger.error("Callback failed after all retries")
            
            return success
            
    except Exception as e:
        request_logger.error("Callback exception", error=str(e), exc_info=True)
        return False


# ============= WebSocket Endpoint =============

@router.websocket("/sessions/{session_id}/stream")
async def stream_frames(
    websocket: WebSocket,
    session_id: str,
    token: str = Query(..., description="JWT token for authentication")
):
    """
    WebSocket endpoint để nhận frames từ client.
    
    **Authentication:**
    - JWT token từ Backend
    - Token type = "websocket"
    - Contains: user_id, session_id (backend), role
    
    **Flow:**
    1. Verify JWT token
    2. Verify user permission (RBAC)
    3. Accept WebSocket connection
    4. Receive frames continuously
    5. Detect + Track + Recognize
    6. Validate (multi-frame)
    7. Send callbacks to Backend
    8. Send real-time updates to Client
    
    **Client Messages:**
    - Binary frames (JPEG/PNG bytes)
    
    **Server Messages:**
    ```json
    {
        "type": "frame_processed",
        "detections": [...],
        "total_faces": 2,
        "timestamp": "..."
    }
    
    {
        "type": "student_validated",
        "student": {
            "student_code": "102220347",
            "student_name": "Nguyen Van A",
            "track_id": 1,
            "avg_confidence": 0.85,
            "frame_count": 10,
            "recognition_count": 8,
            "validation_passed_at": "..."
        }
    }
    
    {
        "type": "session_status",
        "status": "active",
        "stats": {
            "total_frames_processed": 150,
            "total_faces_detected": 180,
            "validated_students": 25
        }
    }
    
    {
        "type": "error",
        "message": "..."
    }
    ```
    
    **Rate Limiting:**
    - Max 30 FPS
    - Max frame size: 2MB
    """
    from app.core.jwt_utils import verify_websocket_token, verify_user_permission
    
    ws_logger = logger.bind(session_id=session_id)
    
    try:
        # 1. Verify JWT token
        ws_logger.info("WebSocket connection attempt")
        
        try:
            token_payload = verify_websocket_token(token)
        except HTTPException as e:
            ws_logger.warning("Token verification failed", error=e.detail)
            await websocket.close(code=1008, reason=f"Unauthorized: {e.detail}")
            return
        
        # 2. Get session data
        session = await session_manager.get_session_data(session_id)
        if not session:
            ws_logger.warning("Session not found")
            await websocket.close(code=1008, reason="Session not found")
            return
        
        if session.status != "active":
            ws_logger.warning("Session not active", status=session.status)
            await websocket.close(code=1008, reason=f"Session not active: {session.status}")
            return
        
        # 3. Verify user permission (RBAC)
        backend_session_id = token_payload.get("session_id")
        if not verify_user_permission(token_payload, session, backend_session_id):
            ws_logger.warning(
                "Permission denied",
                user_id=token_payload.get("user_id"),
                backend_session_id=backend_session_id
            )
            await websocket.close(code=1008, reason="Permission denied")
            return
        
        # 4. Accept connection
        await websocket.accept()
        ws_logger.info(
            "WebSocket connected",
            user_id=token_payload.get("user_id"),
            role=token_payload.get("role")
        )
        
        # Send initial status
        await websocket.send_json({
            "type": "connection_established",
            "session_id": session_id,
            "message": "Connected to AI-Service"
        })
        
        # Get face engine
        engine = get_engine()
        
        # Rate limiting variables
        frame_count = 0
        last_frame_time = time.time()
        validated_students_sent = set()  # Track đã gửi để tránh duplicate
        
        # 5. Process frames
        while True:
            try:
                # Receive binary frame data
                frame_data = await websocket.receive_bytes()
                
                # Rate limiting: Max 30 FPS
                current_time = time.time()
                if current_time - last_frame_time < 1/30:
                    continue
                
                # Validate frame size (max 2MB)
                if len(frame_data) > 2 * 1024 * 1024:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Frame too large (max 2MB)"
                    })
                    continue
                
                frame_count += 1
                last_frame_time = current_time
                
                # 6. Detect faces
                detections, crops, original_image = await engine.detect_faces(frame_data)
                ws_logger.info(f"[Frame {frame_count}] Detected {len(detections)} faces")
                
                # ✅ 6.5. ANTI-SPOOFING CHECK - Lọc fake faces
                if detections and crops:
                    # Check anti-spoofing cho từng face crop
                    anti_spoofing_results = await engine.check_anti_spoofing(crops)
                    
                    # Prepare lists for live faces only
                    live_detections = []
                    live_crops = []
                    suspicious_faces = []
                    
                    # Filter: chỉ giữ live faces
                    for idx, (detection, crop, spoof_result) in enumerate(zip(detections, crops, anti_spoofing_results)):
                        # Update detection với anti-spoofing info
                        detection.is_live = spoof_result['is_live']
                        detection.spoofing_type = spoof_result['label']
                        detection.spoofing_confidence = spoof_result['confidence']
                        
                        if spoof_result['is_live']:
                            # ✅ Live face - giữ lại
                            live_detections.append(detection)
                            live_crops.append(crop)
                            ws_logger.info(
                                f"[Frame {frame_count}] Live face #{idx}",
                                bbox=detection.bbox,
                                confidence=f"{spoof_result['confidence']:.3f}"
                            )
                        else:
                            # ❌ Fake face - loại bỏ
                            suspicious_faces.append({
                                'bbox': detection.bbox,
                                'type': spoof_result['label'],
                                'confidence': spoof_result['confidence']
                            })
                            ws_logger.warning(
                                f"[Frame {frame_count}] 🚨 FAKE DETECTED #{idx}",
                                bbox=detection.bbox,
                                spoofing_type=spoof_result['label'],
                                confidence=f"{spoof_result['confidence']:.3f}"
                            )
                    
                    # Log summary
                    original_count = len(detections)
                    live_count = len(live_detections)
                    fake_count = len(suspicious_faces)
                    
                    ws_logger.info(
                        f"[Frame {frame_count}] Anti-spoofing summary",
                        original_faces=original_count,
                        live_faces=live_count,
                        fake_faces=fake_count
                    )
                    
                    # ⚠️ Gửi alert nếu phát hiện fake faces
                    if suspicious_faces:
                        await websocket.send_json({
                            "type": "anti_spoofing_alert",
                            "frame_count": frame_count,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "suspicious_faces": suspicious_faces,
                            "total_suspicious": fake_count,
                            "total_live": live_count,
                            "message": f"⚠️ Detected {fake_count} fake face(s) - Only {live_count} live face(s) will be processed"
                        })
                    
                    # Cập nhật detections và crops với CHỈ live faces
                    detections = live_detections
                    crops = live_crops
                
                # Nếu không có live faces sau filter
                if not detections:
                    await websocket.send_json({
                        "type": "frame_processed",
                        "frame_count": frame_count,
                        "detections": [],
                        "total_faces": 0,
                        "live_faces": 0,
                        "fake_faces": 0,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    continue
                
                # 7. Recognize faces - CHỈ với live faces
                detections = await engine.recognize_faces(
                    detections=detections,
                    crops=crops,
                    gallery_embeddings=session.gallery_embeddings,
                    gallery_labels=session.gallery_labels
                )
                
                ws_logger.info(f"[Frame {frame_count}] Recognized {len([d for d in detections if d.student_code])} students")
                
                # 8. ✅ Track faces - Sử dụng per-session tracker
                if session.face_tracker:
                    detections = await session.face_tracker.update(detections)
                    ws_logger.info(f"[Frame {frame_count}] Tracked faces: {[(d.track_id, d.student_id) for d in detections]}")
                else:
                    ws_logger.warning("No face_tracker in session")
                
                # 9. ✅ Update recognition history vào per-session validator
                current_timestamp = datetime.now(timezone.utc)
                if session.recognition_validator:
                    for detection in detections:
                        if detection.track_id and detection.student_id:
                            await session.recognition_validator.add_recognition(
                                track_id=detection.track_id,
                                student_code=detection.student_id,  # Using student_code
                                confidence=detection.recognition_confidence if hasattr(detection, 'recognition_confidence') else detection.confidence,
                                timestamp=current_timestamp
                            )
                
                # 10. ✅ Get validated students từ per-session validator
                validated_student_ids = set()
                if session.recognition_validator:
                    for detection in detections:
                        if detection.track_id:
                            validation_result = await session.recognition_validator.validate_recognition(
                                track_id=detection.track_id,
                                current_time=current_timestamp
                            )
                            if validation_result:
                                validated_student_ids.add(validation_result["student_code"])  # Changed from student_id
                
                # 9. Send response với anti-spoofing info
                detections_data = []
                for detection in detections:
                    det_dict = {
                        "bbox": detection.bbox,
                        "confidence": detection.confidence,
                        "track_id": detection.track_id,
                        "student_code": detection.student_code,
                        "student_name": detection.student_name,
                        "recognition_confidence": detection.recognition_confidence,
                        # ✅ Anti-spoofing fields
                        "is_live": detection.is_live,
                        "spoofing_type": detection.spoofing_type,
                        "spoofing_confidence": detection.spoofing_confidence
                    }
                    detections_data.append(det_dict)
                
                await websocket.send_json({
                    "type": "frame_processed",
                    "frame_count": frame_count,
                    "detections": detections_data,
                    "total_faces": len(detections),
                    "live_faces": len([d for d in detections if d.is_live]),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
                # 11. Send student_validated messages (only new ones)
                newly_validated = [student_id for student_id in validated_student_ids 
                                   if student_id not in validated_students_sent]
                
                if newly_validated:
                    from app.models.schemas import ValidatedStudent
                    
                    validated_students_data = []
                    
                    for student_id in newly_validated:
                        # Find detection with this student_id to get track_id and student_name
                        detection_with_student = next((d for d in detections if d.student_id == student_id), None)
                        
                        if detection_with_student and detection_with_student.track_id and session.face_tracker:
                            # ✅ Get track state and stats từ per-session tracker
                            track_state = await session.face_tracker.get_track_info(detection_with_student.track_id)
                            
                            if track_state:
                                stats = track_state.get_recognition_stats(window_size=5)
                                
                                validated_student = ValidatedStudent(
                                    student_code=student_id,
                                    student_name=getattr(detection_with_student, 'student_name', student_id),
                                    track_id=detection_with_student.track_id,
                                    avg_confidence=stats.get('avg_confidence', 0.0),
                                    frame_count=stats.get('total_frames', 0),
                                    recognition_count=stats.get('successful_frames', 0),
                                    validation_passed_at=datetime.now(timezone.utc)
                                )
                                
                                validated_students_data.append(validated_student)
                                
                                # Send student_validated WS message
                                await websocket.send_json({
                                    "type": "student_validated",
                                    "student": {
                                        "student_code": student_id,
                                        "student_name": validated_student.student_name,
                                        "confidence": validated_student.avg_confidence,
                                        "track_id": validated_student.track_id
                                    }
                                })
                                
                                validated_students_sent.add(student_id)
                    
                    # 13. Send callback to Backend for all newly validated students
                    if validated_students_data:
                        attendance_data = AttendanceUpdate(
                            session_id=session_id,
                            validated_students=validated_students_data,
                            timestamp=datetime.now(timezone.utc)
                        )
                        
                        await _send_callback_async(
                            session.backend_callback_url,
                            attendance_data,
                            session_id,
                            ws_logger
                        )
                
                # 14. Increment frame counter
                await session_manager.increment_frame_count(session_id)
                
                # 15. Periodically send session status
                if frame_count % 30 == 0:  # Every 30 frames
                    await websocket.send_json({
                        "type": "session_status",
                        "status": "active",
                        "stats": {
                            "total_frames_processed": session.total_frames_processed,
                            "total_faces_detected": frame_count,
                            "validated_students": len(validated_students_sent)
                        }
                    })
                
            except WebSocketDisconnect:
                ws_logger.info("WebSocket disconnected by client")
                break
            except Exception as e:
                ws_logger.error("Frame processing error", error=str(e), exc_info=True)
                await websocket.send_json({
                    "type": "error",
                    "message": f"Processing error: {str(e)}"
                })
    
    except Exception as e:
        ws_logger.error("WebSocket error", error=str(e))
        try:
            await websocket.close(code=1011, reason=f"Internal error: {str(e)}")
        except:
            pass
