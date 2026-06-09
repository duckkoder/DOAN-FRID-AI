"""
Frame processing endpoints with WebSocket support
"""
import gc
import asyncio
import time
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status, WebSocket, WebSocketDisconnect, Query

from app.models.schemas import AttendanceUpdate
from app.services.session_manager import session_manager
from app.services.face_engine import get_face_engine
from app.core.logging import get_logger
from app.core.memory_manager import get_memory_manager

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


async def _send_callback_async(
    callback_url: str,
    attendance_data: AttendanceUpdate,
    session_id: str,
    request_logger
) -> bool:
    """
    Gá»­i callback async vá»›i error handling
    """
    try:
        # âœ… Create new notifier instance with context manager
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
    WebSocket endpoint Ä‘á»ƒ nháº­n frames tá»« client.
    
    **Authentication:**
    - JWT token tá»« Backend
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
        ws_logger.debug("WebSocket connection attempt")
        
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
        
        # âœ… MEMORY OPTIMIZATION: Get memory manager and settings
        from app.core.config import settings as app_settings
        memory_manager = get_memory_manager()
        memory_manager.reset_frame_counter()
        
        # Rate limiting variables
        frame_count = 0
        last_frame_time = 0.0
        validated_students_sent = set()  # Track Ä‘Ã£ gá»­i Ä‘á»ƒ trÃ¡nh duplicate
        cached_detections_data = []
        cached_total_faces = 0
        cached_real_faces = None
        cached_spoof_faces = None
        last_detection_frame = 0
        
        # âœ… MEMORY: Max faces per frame tá»« config
        MAX_FACES_PER_FRAME = app_settings.MEMORY_MAX_FACES_PER_FRAME
        
        # 5. Process frames
        while True:
            try:
                # Receive binary frame data
                frame_data = await websocket.receive_bytes()
                
                # Rate limiting: Max 30 FPS
                current_time = time.time()
                if current_time - last_frame_time < 1/30:
                    del frame_data
                    await websocket.send_json({
                        "type": "frame_processed",
                        "processing_stage": "completed",
                        "heavy_processed": False,
                        "detections": cached_detections_data,
                        "total_faces": cached_total_faces,
                        "real_faces": cached_real_faces,
                        "spoof_faces": cached_spoof_faces,
                        "rate_limited": True,
                        "source_frame_count": last_detection_frame,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    continue
                
                # âœ… FIX: Cáº­p nháº­t last_frame_time ngay sau khi quyáº¿t Ä‘á»‹nh xá»­ lÃ½ frame
                # (pháº£i trÆ°á»›c má»i continue tiáº¿p theo Ä‘á»ƒ trÃ¡nh skip liÃªn tá»¥c)
                last_frame_time = current_time
                
                # Validate frame size (max 2MB)
                if len(frame_data) > 2 * 1024 * 1024:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Frame too large (max 2MB)"
                    })
                    continue
                
                frame_count += 1

                detection_interval = max(1, app_settings.ATTENDANCE_DETECTION_INTERVAL)
                should_run_detection = (
                    frame_count == 1
                    or last_detection_frame == 0
                    or ((frame_count - 1) % detection_interval == 0)
                )

                if not should_run_detection:
                    del frame_data
                    await websocket.send_json({
                        "type": "frame_processed",
                        "processing_stage": "completed",
                        "heavy_processed": False,
                        "frame_count": frame_count,
                        "detections": cached_detections_data,
                        "total_faces": cached_total_faces,
                        "real_faces": cached_real_faces,
                        "spoof_faces": cached_spoof_faces,
                        "reused_detection": True,
                        "source_frame_count": last_detection_frame,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    await session_manager.increment_frame_count(session_id)
                    continue
                
                # 6. Detect faces
                detections, crops, original_image = await engine.detect_faces(frame_data)
                
                # âœ… MEMORY: Giáº£i phÃ³ng frame_data ngay sau khi detect
                del frame_data
                
                ws_logger.debug(f"[Frame {frame_count}] Detected {len(detections)} faces")
                
                # âœ… MEMORY: Giá»›i háº¡n sá»‘ faces xá»­ lÃ½ má»—i frame
                if len(detections) > MAX_FACES_PER_FRAME:
                    ws_logger.warning(f"[Frame {frame_count}] Too many faces ({len(detections)}), limiting to {MAX_FACES_PER_FRAME}")
                    paired = sorted(
                        zip(detections, crops),
                        key=lambda item: item[0].confidence,
                        reverse=True
                    )[:MAX_FACES_PER_FRAME]
                    detections = [item[0] for item in paired]
                    crops = [item[1] for item in paired]

                # âœ… NOTE: KhÃ´ng gá»­i early 'detected' frame Ä‘á»ƒ trÃ¡nh gá»­i 2 message
                # liÃªn tiáº¿p (detected + completed) gÃ¢y nháº¥p nhÃ¡y UI.
                # Chá»‰ gá»­i 1 message duy nháº¥t 'completed' sau khi xá»­ lÃ½ xong.

                if not detections:
                    cached_detections_data = []
                    cached_total_faces = 0
                    cached_real_faces = 0
                    cached_spoof_faces = 0
                    last_detection_frame = frame_count
                    await websocket.send_json({
                        "type": "frame_processed",
                        "processing_stage": "completed",
                        "heavy_processed": False,
                        "frame_count": frame_count,
                        "detections": [],
                        "total_faces": 0,
                        "real_faces": 0,
                        "spoof_faces": 0,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    await session_manager.increment_frame_count(session_id)
                    cleanup_result = memory_manager.periodic_cleanup()
                    if cleanup_result.get('cleaned'):
                        ws_logger.debug(
                            f"[Frame {frame_count}] Memory cleanup performed",
                            gc_collected=cleanup_result.get('gc_collected', 0),
                            cuda_freed_mb=cleanup_result.get('cuda_freed_mb', 0)
                        )
                    del detections, crops, original_image
                    continue

                heavy_interval = max(1, app_settings.ATTENDANCE_HEAVY_PROCESS_INTERVAL)
                should_run_heavy = (frame_count - 1) % heavy_interval == 0

                if not should_run_heavy:
                    early_detections_data = [
                        {
                            "bbox": detection.bbox,
                            "confidence": detection.confidence,
                            "track_id": detection.track_id,
                            "student_id": "Unknown",
                            "student_code": "Unknown",
                            "student_name": "Unknown",
                            "recognition_confidence": None,
                            "is_live": None,
                            "spoofing_type": None,
                            "spoofing_confidence": None,
                        }
                        for detection in detections
                    ]
                    await websocket.send_json({
                        "type": "frame_processed",
                        "processing_stage": "completed",
                        "heavy_processed": False,
                        "frame_count": frame_count,
                        "detections": early_detections_data,
                        "total_faces": len(detections),
                        "real_faces": None,
                        "spoof_faces": None,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    cached_detections_data = early_detections_data
                    cached_total_faces = len(detections)
                    cached_real_faces = None
                    cached_spoof_faces = None
                    last_detection_frame = frame_count
                    await session_manager.increment_frame_count(session_id)
                    cleanup_result = memory_manager.periodic_cleanup()
                    if cleanup_result.get('cleaned'):
                        ws_logger.debug(
                            f"[Frame {frame_count}] Memory cleanup performed",
                            gc_collected=cleanup_result.get('gc_collected', 0),
                            cuda_freed_mb=cleanup_result.get('cuda_freed_mb', 0)
                        )
                    del detections, crops, original_image
                    continue
                
                # âœ… 6.5. ANTI-SPOOFING CHECK - Keep ALL faces but mark spoof status
                # recognition_crops includes only live faces. Spoof faces are never
                # recognized; they are only shown as alerts/evidence.
                recognition_crops = []
                recognition_indices = []
                spoof_count = 0
                real_count = 0
                
                if detections and crops:
                    # Check anti-spoofing cho tá»«ng face crop
                    anti_spoofing_results = await engine.check_anti_spoofing(crops)
                    
                    # Update ALL detections vá»›i anti-spoofing info + temporal smoothing
                    for idx, (detection, crop, spoof_result) in enumerate(zip(detections, crops, anti_spoofing_results)):
                        raw_is_live = spoof_result['is_live']

                        # âœ… Ãp dá»¥ng temporal smoothing qua track vote buffer
                        # Náº¿u face chÆ°a cÃ³ track_id, fallback vá» raw prediction
                        smoothed_live = raw_is_live
                        if session.face_tracker and detection.track_id is not None:
                            track_state = await session.face_tracker.get_track_info(detection.track_id)
                            if track_state is not None:
                                smoothed_live = track_state.add_spoof_vote(raw_is_live)

                        # Update detection vá»›i smoothed result
                        detection.is_live = smoothed_live
                        detection.spoofing_type = spoof_result['label']
                        detection.spoofing_confidence = spoof_result['confidence']

                        if smoothed_live:
                            recognition_crops.append(crop)
                            recognition_indices.append(idx)
                            real_count += 1
                            ws_logger.debug(
                                f"[Frame {frame_count}] âœ… REAL/LIVE face #{idx} (raw={raw_is_live}, smooth={smoothed_live})",
                                bbox=detection.bbox,
                                label=spoof_result['label'],
                                confidence=f"{spoof_result['confidence']:.1%}",
                            )
                        else:
                            # ðŸš¨ Spoof face - KHÃ”NG loáº¡i bá», CHá»ˆ Ä‘Ã¡nh dáº¥u
                            spoof_count += 1
                            ws_logger.warning(
                                f"[Frame {frame_count}] ðŸš¨ SPOOF CONFIRMED #{idx} (raw={raw_is_live}, smooth={smoothed_live})",
                                bbox=detection.bbox,
                                label=spoof_result['label'],
                                confidence=f"{spoof_result['confidence']:.1%}",
                            )

                            # âœ… LÆ°u spoof face crop vÃ o session memory (Ä‘á»ƒ upload S3 khi end_session)
                            await session_manager.store_spoof_face_crop(
                                session_id=session_id,
                                face_crop=crop,
                                spoofing_type=spoof_result['label'],
                                spoofing_confidence=spoof_result['confidence'],
                                frame_count=frame_count
                            )
                    
                    # Log summary
                    ws_logger.debug(
                        f"[Frame {frame_count}] Anti-spoofing summary",
                        total_faces=len(detections),
                        real_faces=real_count,
                        spoof_faces=spoof_count
                    )
                    
                    # âš ï¸ Gá»­i alert náº¿u phÃ¡t hiá»‡n spoof faces
                    if spoof_count > 0:
                        await websocket.send_json({
                            "type": "anti_spoofing_alert",
                            "frame_count": frame_count,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "total_spoof": spoof_count,
                            "total_real": real_count,
                            "message": f"Detected {spoof_count} spoof face(s)"
                        })
                
                # âœ… KEEP detections (bao gá»“m cáº£ spoof faces Ä‘á»ƒ frontend hiá»ƒn thá»‹)
                # Only live faces continue into recognition.
                
                # Náº¿u khÃ´ng cÃ³ real faces sau filter - VáºªN Gá»¬I spoof faces vá» Ä‘á»ƒ hiá»ƒn thá»‹
                if not recognition_crops and spoof_count > 0:
                    # Chá»‰ cÃ³ spoof faces - gá»­i vá» nhÆ°ng khÃ´ng lÃ m gÃ¬ thÃªm
                    ws_logger.warning(f"[Frame {frame_count}] Only spoof faces detected - sending to frontend for display")
                    
                    detections_data = []
                    for detection in detections:
                        det_dict = {
                            "bbox": detection.bbox,
                            "confidence": detection.confidence,
                            "track_id": None,  # KhÃ´ng track spoof faces
                            "student_id": "Unknown",
                            "student_code": "Unknown",  # âœ… Thay null thÃ nh "Unknown"
                            "student_name": "Unknown",  # âœ… Thay null thÃ nh "Unknown"
                            "recognition_confidence": None,
                            # âœ… Anti-spoofing fields
                            "is_live": detection.is_live,
                            "spoofing_type": detection.spoofing_type,
                            "spoofing_confidence": detection.spoofing_confidence
                        }
                        detections_data.append(det_dict)
                    
                    await websocket.send_json({
                        "type": "frame_processed",
                        "processing_stage": "completed",
                        "frame_count": frame_count,
                        "detections": detections_data,
                        "total_faces": len(detections),
                        "real_faces": 0,
                        "spoof_faces": spoof_count,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    cached_detections_data = detections_data
                    cached_total_faces = len(detections)
                    cached_real_faces = 0
                    cached_spoof_faces = spoof_count
                    last_detection_frame = frame_count
                    await session_manager.increment_frame_count(session_id)
                    del detections, crops, original_image, recognition_crops
                    if 'anti_spoofing_results' in locals():
                        del anti_spoofing_results
                    continue
                
                # Náº¿u khÃ´ng cÃ³ faces nÃ o (cáº£ real láº«n spoof)
                if not detections:
                    await websocket.send_json({
                        "type": "frame_processed",
                        "processing_stage": "completed",
                        "frame_count": frame_count,
                        "detections": [],
                        "total_faces": 0,
                        "real_faces": 0,
                        "spoof_faces": 0,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                    cached_detections_data = []
                    cached_total_faces = 0
                    cached_real_faces = 0
                    cached_spoof_faces = 0
                    last_detection_frame = frame_count
                    await session_manager.increment_frame_count(session_id)
                    del detections, crops, original_image, recognition_crops
                    continue
                
                # 7. Recognize faces - only live faces are allowed.
                if recognition_crops:
                    recognized_detections = await engine.recognize_faces(
                        detections=[detections[i] for i in recognition_indices],
                        crops=recognition_crops,
                        gallery_embeddings=session.gallery_embeddings,
                        gallery_labels=session.gallery_labels
                    )
                    
                    for i, idx in enumerate(recognition_indices):
                        detections[idx] = recognized_detections[i]
                    
                    ws_logger.debug(f"[Frame {frame_count}] Recognized {len([d for d in recognized_detections if d.student_code])} students")
                else:
                    ws_logger.debug(f"[Frame {frame_count}] No faces allowed to recognize")
                
                # 8. âœ… Track faces - CHá»ˆ track REAL faces
                if session.face_tracker and recognition_indices:
                    detections_for_tracking = [detections[i] for i in recognition_indices]
                    tracked_detections = await session.face_tracker.update(detections_for_tracking)
                    
                    # Update track_id vÃ o detections gá»‘c
                    for tracked_idx, detection_idx in enumerate(recognition_indices):
                        detections[detection_idx] = tracked_detections[tracked_idx]
                    
                    ws_logger.debug(f"[Frame {frame_count}] Tracked faces: {[(d.track_id, d.student_id) for d in tracked_detections]}")
                elif not session.face_tracker:
                    ws_logger.warning("No face_tracker in session")
                
                # 9. âœ… Update recognition history vÃ o per-session validator - CHá»ˆ REAL faces
                current_timestamp = datetime.now(timezone.utc)
                recognition_index_set = set(recognition_indices)
                if session.recognition_validator:
                    for detection_idx, detection in enumerate(detections):
                        if detection_idx in recognition_index_set and detection.track_id and detection.student_id:
                            await session.recognition_validator.add_recognition(
                                track_id=detection.track_id,
                                student_code=detection.student_id,  # Using student_code
                                confidence=detection.recognition_confidence if hasattr(detection, 'recognition_confidence') else detection.confidence,
                                timestamp=current_timestamp
                            )
                
                # 10. âœ… Get validated students tá»« per-session validator - CHá»ˆ REAL faces
                validated_student_ids = set()
                
                if session.recognition_validator:
                    detection_to_crop_idx = {
                        detection_idx: crop_idx
                        for crop_idx, detection_idx in enumerate(recognition_indices)
                    }
                    
                    for i, detection in enumerate(detections):
                        if i in recognition_index_set and detection.track_id:
                            validation_result = await session.recognition_validator.validate_recognition(
                                track_id=detection.track_id,
                                current_time=current_timestamp
                            )
                            if validation_result:
                                student_code = validation_result["student_code"]
                                validated_student_ids.add(student_code)
                                
                                # âœ… LÆ°u face crop vÃ o session memory (Ä‘á»ƒ láº¥y sau khi end_session)
                                if i in detection_to_crop_idx:
                                    crop_idx = detection_to_crop_idx[i]
                                    if crop_idx < len(recognition_crops):
                                        await session_manager.store_validated_student_crop(
                                            session_id=session_id,
                                            student_code=student_code,
                                            face_crop=recognition_crops[crop_idx]
                                        )
                
                # 11. âœ… Send response vá»›i Táº¤T Cáº¢ detections (bao gá»“m cáº£ spoof faces)
                detections_data = []
                for detection in detections:
                    det_dict = {
                        "bbox": detection.bbox,
                        "confidence": detection.confidence,
                        "track_id": detection.track_id,
                        "student_id": detection.student_code or "Unknown",
                        "student_code": detection.student_code or "Unknown",  # âœ… Thay null thÃ nh "Unknown"
                        "student_name": detection.student_name or "Unknown",  # âœ… Thay null thÃ nh "Unknown"
                        "recognition_confidence": detection.recognition_confidence,
                        # âœ… Anti-spoofing fields
                        "is_live": detection.is_live,
                        "spoofing_type": detection.spoofing_type,
                        "spoofing_confidence": detection.spoofing_confidence
                    }
                    detections_data.append(det_dict)
                
                await websocket.send_json({
                    "type": "frame_processed",
                    "processing_stage": "completed",
                    "frame_count": frame_count,
                    "detections": detections_data,
                    "total_faces": len(detections),
                    "real_faces": real_count,
                    "spoof_faces": spoof_count,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                cached_detections_data = detections_data
                cached_total_faces = len(detections)
                cached_real_faces = real_count
                cached_spoof_faces = spoof_count
                last_detection_frame = frame_count
                
                # 11. Send student_validated messages (only new ones)
                newly_validated = [student_id for student_id in validated_student_ids 
                                   if student_id not in validated_students_sent]
                
                if newly_validated:
                    from app.models.schemas import ValidatedStudent
                    
                    validated_students_data = []
                    
                    for student_id in newly_validated:
                        # Find detection with this student_id to get track_id and student_name
                        # âš ï¸ CHá»ˆ tÃ¬m trong REAL faces (is_live = True)
                        detection_with_student = next(
                            (d for i, d in enumerate(detections) if d.student_id == student_id and i in recognition_index_set),
                            None
                        )
                        
                        if detection_with_student and detection_with_student.track_id and session.face_tracker:
                            # âœ… Get track state and stats tá»« per-session tracker
                            track_state = await session.face_tracker.get_track_info(detection_with_student.track_id)
                            
                            if track_state:
                                stats = track_state.get_recognition_stats(window_size=5)
                                
                                # âœ… KHÃ”NG encode base64 ná»¯a, áº£nh Ä‘Ã£ lÆ°u trong session memory
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
                        
                        asyncio.create_task(_send_callback_async(
                            session.backend_callback_url,
                            attendance_data,
                            session_id,
                            ws_logger
                        ))
                
                # 14. Increment frame counter
                await session_manager.increment_frame_count(session_id)
                
                # âœ… MEMORY CLEANUP: Giáº£i phÃ³ng cÃ¡c objects khÃ´ng cáº§n thiáº¿t
                del detections, crops, original_image, recognition_crops
                # âœ… FIX: DÃ¹ng locals() thay dir() Ä‘á»ƒ kiá»ƒm tra biáº¿n local Ä‘Ãºng cÃ¡ch
                if 'anti_spoofing_results' in locals():
                    del anti_spoofing_results
                
                # âœ… MEMORY: Periodic cleanup sau má»—i N frames
                cleanup_result = memory_manager.periodic_cleanup()
                if cleanup_result.get('cleaned'):
                    ws_logger.debug(
                        f"[Frame {frame_count}] Memory cleanup performed",
                        gc_collected=cleanup_result.get('gc_collected', 0),
                        cuda_freed_mb=cleanup_result.get('cuda_freed_mb', 0)
                    )
                
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
                # âœ… MEMORY: Cleanup khi disconnect
                memory_manager.force_cleanup()
                break
            except Exception as e:
                ws_logger.error("Frame processing error", error=str(e), exc_info=True)
                await websocket.send_json({
                    "type": "error",
                    "message": f"Processing error: {str(e)}"
                })
                # âœ… MEMORY: Cleanup khi cÃ³ lá»—i
                memory_manager.cleanup_python_gc()
    
    except Exception as e:
        ws_logger.error("WebSocket error", error=str(e))
        # âœ… MEMORY: Final cleanup
        memory_manager.force_cleanup()
        try:
            await websocket.close(code=1011, reason=f"Internal error: {str(e)}")
        except:
            pass
