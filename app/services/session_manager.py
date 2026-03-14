"""
Session Manager - Quản lý sessions in-memory
"""
import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
import torch
import numpy as np
import cv2

from app.models.schemas import SessionCreateRequest, SessionResponse
from app.core.logging import LoggerMixin
from app.services.database_service import get_database_service


@dataclass
class SpoofFaceCrop:
    """
    Dữ liệu một ảnh spoof face - LƯU DẠNG NÉN ĐỂ TIẾT KIỆM MEMORY
    """
    face_crop_jpeg: bytes  # ✅ JPEG bytes thay vì numpy array để tiết kiệm RAM
    spoofing_type: str  # 'spoof', 'print', 'replay', etc.
    spoofing_confidence: float  # Độ tin cậy của prediction
    detected_at: datetime  # Thời điểm phát hiện
    frame_count: int  # Frame số mấy phát hiện
    
    def get_face_crop(self) -> np.ndarray:
        """Decompress JPEG bytes back to RGB numpy array"""
        nparr = np.frombuffer(self.face_crop_jpeg, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@dataclass
class ValidatedStudentCrop:
    """
    Dữ liệu face crop của student đã validated - LƯU DẠNG NÉN
    """
    face_crop_jpeg: bytes  # ✅ JPEG bytes thay vì numpy array
    
    def get_face_crop(self) -> np.ndarray:
        """Decompress JPEG bytes back to RGB numpy array"""
        nparr = np.frombuffer(self.face_crop_jpeg, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _compress_face_crop(face_crop: np.ndarray, quality: int = 85) -> bytes:
    """Compress face crop to JPEG bytes"""
    bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode('.jpg', bgr, encode_params)
    return buffer.tobytes()


@dataclass
class SessionData:
    """Dữ liệu session lưu trong memory với embeddings loaded vào VRAM"""
    session_id: str
    backend_session_id: int  # Backend session ID để mapping
    class_id: str
    backend_callback_url: str
    status: str
    created_at: datetime
    allowed_users: List[str] = field(default_factory=list)  # RBAC: user_ids được phép
    embeddings_loaded: bool = False
    total_frames_processed: int = 0
    max_duration_minutes: int = 60
    
    # Embeddings data loaded vào VRAM (GPU memory)
    gallery_embeddings: Optional[torch.Tensor] = None  # Shape: (N, 512) - N embeddings
    gallery_labels: Optional[List[str]] = None  # List of student_codes
    gallery_student_ids: Optional[List[int]] = None  # List of student_ids
    student_codes: List[str] = field(default_factory=list)  # Original list
    embedding_count: int = 0  # Total embeddings loaded
    
    # Per-session Tracker and Validator instances
    face_tracker: Optional[Any] = None  # FaceTracker instance
    recognition_validator: Optional[Any] = None  # RecognitionValidator instance
    
    # ✅ Storage for validated students with face crops (for end_session upload) - COMPRESSED
    validated_students_crops: Dict[str, ValidatedStudentCrop] = field(default_factory=dict)
    
    # ✅ Storage for spoof faces detected during session (for evidence upload) - COMPRESSED
    spoof_faces_crops: List[SpoofFaceCrop] = field(default_factory=list)


class SessionManager(LoggerMixin):
    """Quản lý sessions in-memory với thread safety"""
    
    def __init__(self):
        super().__init__()
        self._sessions: Dict[str, SessionData] = {}
        self._lock = asyncio.Lock()
    
    async def create_session(self, request: SessionCreateRequest) -> SessionResponse:
        """
        Tạo session mới và load embeddings vào VRAM.
        
        BƯỚC 1: Query embeddings từ pgvector (1 lần DUY NHẤT)
        BƯỚC 2: Load embeddings vào VRAM (GPU memory)
        
        Args:
            request: Thông tin tạo session với student_codes
            
        Returns:
            Thông tin session đã tạo
        """
        session_id = str(uuid.uuid4())
        
        async with self._lock:
            session_data = SessionData(
                session_id=session_id,
                backend_session_id=request.backend_session_id,
                class_id=request.class_id,
                backend_callback_url=request.backend_callback_url,
                status="active",
                created_at=datetime.now(timezone.utc),
                allowed_users=request.allowed_users,
                max_duration_minutes=request.max_duration_minutes or 60,
                student_codes=request.student_codes
            )
            
            self._sessions[session_id] = session_data
            
            # BƯỚC 1: Query embeddings từ database (1 query duy nhất)
            try:
                embeddings_data = await self._load_embeddings_from_database(request.student_codes)
                
                # BƯỚC 2: Load vào VRAM
                await self._load_embeddings_to_vram(session_data, embeddings_data)
                
                session_data.embeddings_loaded = True
                
                # BƯỚC 3: ✅ Tạo per-session Tracker và Validator
                await self._initialize_session_tracker_and_validator(session_data)
                
                self.logger.info(
                    "Session created with embeddings loaded to VRAM",
                    session_id=session_id,
                    class_id=request.class_id,
                    student_count=len(request.student_codes),
                    embedding_count=session_data.embedding_count,
                    avg_per_student=session_data.embedding_count / len(request.student_codes) if request.student_codes else 0
                )
            
            except Exception as e:
                session_data.embeddings_loaded = False
                self.logger.error(
                    "Failed to load embeddings for session",
                    session_id=session_id,
                    error=str(e)
                )
                raise
            
            return self._session_data_to_response(session_data)
    
    async def _load_embeddings_from_database(self, student_codes: List[str]) -> List[Dict[str, Any]]:
        """
        BƯỚC 1: Query embeddings từ pgvector (1 lần DUY NHẤT).
        
        Args:
            student_codes: List of student codes
        
        Returns:
            List of embedding dictionaries
        """
        self.logger.info("Loading embeddings from database", student_count=len(student_codes))
        
        # Run sync database query in thread pool
        loop = asyncio.get_event_loop()
        db_service = get_database_service()
        
        embeddings_data = await loop.run_in_executor(
            None,
            db_service.get_embeddings_by_student_codes,
            student_codes,
            "approved"
        )
        
        self.logger.info(
            "Embeddings loaded from database",
            embedding_count=len(embeddings_data)
        )
        
        return embeddings_data
    
    async def _load_embeddings_to_vram(
        self,
        session_data: SessionData,
        embeddings_data: List[Dict[str, Any]]
    ) -> None:
        """
        BƯỚC 2: Load embeddings vào VRAM (GPU memory).
        Gộp 500 vectors thành 1 tensor và lưu vào SessionData.
        
        Args:
            session_data: Session data to update
            embeddings_data: List of embeddings from database
        """
        if not embeddings_data:
            self.logger.warning("No embeddings to load to VRAM")
            session_data.gallery_embeddings = torch.tensor([]).cuda() if torch.cuda.is_available() else torch.tensor([])
            session_data.gallery_labels = []
            session_data.gallery_student_ids = []
            session_data.embedding_count = 0
            return
        
        # Extract data
        embeddings_list = []
        labels_list = []
        student_ids_list = []
        
        for emb_data in embeddings_data:
            embedding = emb_data['embedding']
            
            # Convert to numpy if needed
            if isinstance(embedding, list):
                embedding = np.array(embedding, dtype=np.float32)
            
            embeddings_list.append(embedding)
            labels_list.append(emb_data['student_code'])
            student_ids_list.append(emb_data['student_id'])
        
        # Stack into single array: (N, 512)
        embeddings_array = np.stack(embeddings_list, axis=0)  # Shape: (N, 512)
        
        # Convert to torch tensor
        embeddings_tensor = torch.from_numpy(embeddings_array).float()
        
        # ⚠️ CRITICAL: L2 normalization - MUST normalize before comparison
        # Face embeddings MUST be normalized for distance calculation to work correctly
        embeddings_tensor = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=1)
        
        if torch.cuda.is_available():
            embeddings_tensor = embeddings_tensor.cuda()
            self.logger.info("Embeddings loaded to GPU (CUDA) and L2 normalized")
        else:
            self.logger.warning("CUDA not available, using CPU")
            self.logger.info("Embeddings L2 normalized")
        
        # Update session data
        session_data.gallery_embeddings = embeddings_tensor
        session_data.gallery_labels = labels_list
        session_data.gallery_student_ids = student_ids_list
        session_data.embedding_count = len(embeddings_data)
        
        self.logger.info(
            "Embeddings loaded to VRAM",
            embedding_count=len(embeddings_data),
            tensor_shape=tuple(embeddings_tensor.shape),
            device=embeddings_tensor.device,
            memory_mb=embeddings_tensor.element_size() * embeddings_tensor.nelement() / (1024 * 1024)
        )
    
    async def _initialize_session_tracker_and_validator(
        self,
        session_data: SessionData
    ) -> None:
        """
        BƯỚC 3: Khởi tạo per-session Tracker và Validator
        
        Mỗi session có:
        - FaceTracker riêng (không chia sẻ tracks giữa sessions)
        - RecognitionValidator riêng (không chia sẻ history/debounce)
        
        Args:
            session_data: Session data to initialize
        """
        from app.services.tracker import create_face_tracker
        from app.services.recognition_validator import create_recognition_validator
        from app.core.config import settings
        
        try:
            # Tạo FaceTracker per-session với IoU enabled
            face_tracker = create_face_tracker(
                max_disappeared=30,
                distance_threshold=200,
                iou_threshold=0.3,
                use_iou=True  # ✅ Sử dụng IoU thay vì distance
            )
            
            # Tạo RecognitionValidator per-session với auto-adjust FPS
            recognition_validator = create_recognition_validator(
                face_tracker=face_tracker,
                confirmation_threshold=getattr(settings, 'RECOGNITION_CONFIRMATION_THRESHOLD', 3),
                window_size=getattr(settings, 'RECOGNITION_WINDOW_SIZE', 5),
                min_avg_confidence=getattr(settings, 'RECOGNITION_MIN_AVG_CONFIDENCE', 0.5),
                min_success_rate=getattr(settings, 'RECOGNITION_MIN_FRAME_SUCCESS_RATE', 0.6),
                debounce_seconds=getattr(settings, 'RECOGNITION_DEBOUNCE_SECONDS', 30),
                auto_adjust_to_fps=True,  # ✅ Tự động điều chỉnh theo FPS
                target_fps=5.0
            )
            
            # Lưu vào session data
            session_data.face_tracker = face_tracker
            session_data.recognition_validator = recognition_validator
            
            self.logger.info(
                "Per-session Tracker and Validator initialized",
                session_id=session_data.session_id,
                use_iou=True,
                auto_adjust_fps=True
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to initialize tracker and validator",
                session_id=session_data.session_id,
                error=str(e)
            )
            # Don't raise - session can work without tracking
    
    async def get_session(self, session_id: str) -> Optional[SessionResponse]:
        """
        Lấy thông tin session (DTO)
        
        Args:
            session_id: ID của session
            
        Returns:
            Thông tin session hoặc None nếu không tồn tại
        """
        async with self._lock:
            session_data = self._sessions.get(session_id)
            if not session_data:
                return None
            
            # Kiểm tra session có hết hạn không
            if self._is_session_expired(session_data):
                session_data.status = "expired"
            
            return self._session_data_to_response(session_data)
    
    async def get_session_data(self, session_id: str) -> Optional[SessionData]:
        """
        Lấy SessionData thực (với embeddings) - for internal use
        
        Args:
            session_id: ID của session
            
        Returns:
            SessionData object hoặc None nếu không tồn tại
        """
        async with self._lock:
            session_data = self._sessions.get(session_id)
            if not session_data:
                return None
            
            # Kiểm tra session có hết hạn không
            if self._is_session_expired(session_data):
                session_data.status = "expired"
            
            return session_data
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Xóa session
        
        Args:
            session_id: ID của session
            
        Returns:
            True nếu xóa thành công, False nếu không tồn tại
        """
        async with self._lock:
            session_data = self._sessions.get(session_id)
            if not session_data:
                return False
            
            session_data.status = "ended"
            del self._sessions[session_id]
            
            self.logger.info(
                "Session deleted",
                session_id=session_id,
                class_id=session_data.class_id,
                total_frames=session_data.total_frames_processed
            )
            
            return True
    
    async def increment_frame_count(self, session_id: str) -> bool:
        """
        Tăng số lượng frame đã xử lý
        
        Args:
            session_id: ID của session
            
        Returns:
            True nếu thành công, False nếu session không tồn tại
        """
        async with self._lock:
            session_data = self._sessions.get(session_id)
            if not session_data:
                return False
            
            session_data.total_frames_processed += 1
            return True
    
    async def get_session_data(self, session_id: str) -> Optional[SessionData]:
        """
        Get full session data including embeddings in VRAM.
        Used by face recognition service.
        
        Args:
            session_id: Session ID
        
        Returns:
            SessionData object or None if not found
        """
        async with self._lock:
            return self._sessions.get(session_id)
    
    async def get_active_sessions_count(self) -> int:
        """Lấy số lượng session đang hoạt động"""
        async with self._lock:
            active_count = 0
            for session_data in self._sessions.values():
                if session_data.status == "active" and not self._is_session_expired(session_data):
                    active_count += 1
            return active_count
    
    async def cleanup_expired_sessions(self) -> int:
        """
        Dọn dẹp các session đã hết hạn
        
        Returns:
            Số lượng session đã được dọn dẹp
        """
        async with self._lock:
            expired_sessions = []
            
            for session_id, session_data in self._sessions.items():
                if self._is_session_expired(session_data):
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self._sessions[session_id]
                self.logger.info("Expired session cleaned up", session_id=session_id)
            
            return len(expired_sessions)
    
    async def store_validated_student_crop(
        self,
        session_id: str,
        student_code: str,
        face_crop: np.ndarray
    ) -> bool:
        """
        Lưu face crop của student đã validated vào session memory.
        Sẽ được lấy ra khi end_session để upload S3.
        
        ✅ MEMORY OPTIMIZATION: Lưu dạng JPEG compressed
        
        Args:
            session_id: ID của session
            student_code: Mã sinh viên
            face_crop: Ảnh khuôn mặt crop (numpy array RGB)
            
        Returns:
            True nếu lưu thành công
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                self.logger.warning(f"Session not found: {session_id}")
                return False
            
            # ✅ MEMORY: Compress to JPEG bytes (~ 10-20x smaller)
            jpeg_bytes = _compress_face_crop(face_crop, quality=85)
            
            # Lưu compressed crop (overwrite nếu đã có)
            session.validated_students_crops[student_code] = ValidatedStudentCrop(
                face_crop_jpeg=jpeg_bytes
            )
            
            self.logger.debug(
                f"Stored compressed face crop for {student_code}",
                session_id=session_id,
                original_size=face_crop.nbytes,
                compressed_size=len(jpeg_bytes),
                compression_ratio=f"{face_crop.nbytes / len(jpeg_bytes):.1f}x"
            )
            
            return True
    
    async def get_validated_students_crops(self, session_id: str) -> Dict[str, np.ndarray]:
        """
        Lấy tất cả face crops của students đã validated trong session.
        
        Args:
            session_id: ID của session
            
        Returns:
            Dict mapping student_code -> face_crop (numpy array RGB)
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                self.logger.warning(f"Session not found: {session_id}")
                return {}
            
            # ✅ Decompress when retrieving
            result = {}
            for student_code, crop_data in session.validated_students_crops.items():
                result[student_code] = crop_data.get_face_crop()
            
            return result
    
    async def store_spoof_face_crop(
        self,
        session_id: str,
        face_crop: np.ndarray,
        spoofing_type: str,
        spoofing_confidence: float,
        frame_count: int
    ) -> bool:
        """
        Lưu spoof face crop vào session memory để upload lên S3 khi end_session.
        
        ⚠️ QUALITY FILTER để tránh spam:
        - Phải cách ít nhất 15 frame so với ảnh trước đó
        - ✅ MEMORY: Giới hạn tối đa 50 ảnh spoof mỗi session
        - ✅ MEMORY: Lưu dạng JPEG compressed
        
        Args:
            session_id: ID của session
            face_crop: Ảnh khuôn mặt crop (numpy array RGB)
            spoofing_type: Loại giả mạo ('spoof', 'print', 'replay', etc.)
            spoofing_confidence: Độ tin cậy của prediction
            frame_count: Frame số mấy phát hiện
            
        Returns:
            True nếu lưu thành công, False nếu bị skip
        """
        from app.core.config import settings
        
        MIN_FRAME_GAP = 15     # Phải cách ít nhất 15 frame
        MAX_SPOOF_CROPS = settings.MEMORY_MAX_SPOOF_CROPS  # ✅ Từ config
        
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                self.logger.warning(f"Session not found: {session_id}")
                return False
            
            # ⚠️ QUALITY CHECK: Chưa đủ khoảng cách frame?
            if session.spoof_faces_crops:
                last_frame = session.spoof_faces_crops[-1].frame_count
                if frame_count - last_frame < MIN_FRAME_GAP:
                    self.logger.debug(
                        f"Frame gap too small ({frame_count - last_frame} < {MIN_FRAME_GAP}), skipping",
                        session_id=session_id,
                        frame_count=frame_count
                    )
                    return False
            
            # ✅ MEMORY: Giới hạn tối đa số ảnh spoof lưu trữ
            if len(session.spoof_faces_crops) >= MAX_SPOOF_CROPS:
                self.logger.warning(
                    f"Max spoof crops reached ({MAX_SPOOF_CROPS}), skipping",
                    session_id=session_id,
                    frame_count=frame_count
                )
                return False
            
            # ✅ MEMORY: Compress to JPEG bytes
            jpeg_bytes = _compress_face_crop(face_crop, quality=80)  # Lower quality for spoofs
            
            # Tạo SpoofFaceCrop object với compressed data
            spoof_data = SpoofFaceCrop(
                face_crop_jpeg=jpeg_bytes,
                spoofing_type=spoofing_type,
                spoofing_confidence=spoofing_confidence,
                detected_at=datetime.now(timezone.utc),
                frame_count=frame_count
            )
            
            # Lưu vào list
            session.spoof_faces_crops.append(spoof_data)
            
            self.logger.info(
                f"✅ Stored spoof #{len(session.spoof_faces_crops)} (compressed)",
                session_id=session_id,
                spoofing_type=spoofing_type,
                confidence=f"{spoofing_confidence:.1%}",
                frame_count=frame_count,
                compressed_size=len(jpeg_bytes)
            )
            
            return True
    
    async def get_spoof_faces_crops(self, session_id: str) -> List[SpoofFaceCrop]:
        """
        Lấy tất cả spoof face crops trong session.
        
        Args:
            session_id: ID của session
            
        Returns:
            List of SpoofFaceCrop objects
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                self.logger.warning(f"Session not found: {session_id}")
                return []
            
            return session.spoof_faces_crops.copy()
    
    def _session_data_to_response(self, session_data: SessionData) -> SessionResponse:
        """Chuyển đổi SessionData thành SessionResponse"""
        return SessionResponse(
            session_id=session_data.session_id,
            class_id=session_data.class_id,
            status=session_data.status,
            created_at=session_data.created_at,
            backend_callback_url=session_data.backend_callback_url,
            embeddings_loaded=session_data.embeddings_loaded,
            total_frames_processed=session_data.total_frames_processed
        )
    
    def _is_session_expired(self, session_data: SessionData) -> bool:
        """Kiểm tra session có hết hạn không"""
        if session_data.status != "active":
            return True
        
        expiry_time = session_data.created_at + timedelta(minutes=session_data.max_duration_minutes)
        return datetime.now(timezone.utc) > expiry_time


# Global session manager instance
session_manager = SessionManager()
