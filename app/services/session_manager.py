"""
Session Manager - Quáº£n lÃ½ sessions in-memory
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


@dataclass
class SpoofFaceCrop:
    """
    Dá»¯ liá»‡u má»™t áº£nh spoof face - LÆ¯U Dáº NG NÃ‰N Äá»‚ TIáº¾T KIá»†M MEMORY
    """
    face_crop_jpeg: bytes  # âœ… JPEG bytes thay vÃ¬ numpy array Ä‘á»ƒ tiáº¿t kiá»‡m RAM
    spoofing_type: str  # 'spoof', 'print', 'replay', etc.
    spoofing_confidence: float  # Äá»™ tin cáº­y cá»§a prediction
    detected_at: datetime  # Thá»i Ä‘iá»ƒm phÃ¡t hiá»‡n
    frame_count: int  # Frame sá»‘ máº¥y phÃ¡t hiá»‡n
    
    def get_face_crop(self) -> np.ndarray:
        """Decompress JPEG bytes back to RGB numpy array"""
        nparr = np.frombuffer(self.face_crop_jpeg, np.uint8)
        bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@dataclass
class ValidatedStudentCrop:
    """
    Dá»¯ liá»‡u face crop cá»§a student Ä‘Ã£ validated - LÆ¯U Dáº NG NÃ‰N
    """
    face_crop_jpeg: bytes  # âœ… JPEG bytes thay vÃ¬ numpy array
    
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
    """Dá»¯ liá»‡u session lÆ°u trong memory vá»›i embeddings loaded vÃ o VRAM"""
    session_id: str
    backend_session_id: int  # Backend session ID Ä‘á»ƒ mapping
    class_id: str
    backend_callback_url: str
    status: str
    created_at: datetime
    allowed_users: List[str] = field(default_factory=list)  # RBAC: user_ids Ä‘Æ°á»£c phÃ©p
    embeddings_loaded: bool = False
    total_frames_processed: int = 0
    max_duration_minutes: int = 60
    
    # Embeddings data loaded vÃ o VRAM (GPU memory)
    gallery_embeddings: Optional[torch.Tensor] = None  # Shape: (N, 512) - N embeddings
    gallery_labels: Optional[List[str]] = None  # List of student_codes
    gallery_student_ids: Optional[List[int]] = None  # List of student_ids
    student_codes: List[str] = field(default_factory=list)  # Original list
    embedding_count: int = 0  # Total embeddings loaded
    
    # Per-session Tracker and Validator instances
    face_tracker: Optional[Any] = None  # FaceTracker instance
    recognition_validator: Optional[Any] = None  # RecognitionValidator instance
    
    # âœ… Storage for validated students with face crops (for end_session upload) - COMPRESSED
    validated_students_crops: Dict[str, ValidatedStudentCrop] = field(default_factory=dict)
    
    # âœ… Storage for spoof faces detected during session (for evidence upload) - COMPRESSED
    spoof_faces_crops: List[SpoofFaceCrop] = field(default_factory=list)


class SessionManager(LoggerMixin):
    """Quáº£n lÃ½ sessions in-memory vá»›i thread safety"""
    
    def __init__(self):
        super().__init__()
        self._sessions: Dict[str, SessionData] = {}
        self._lock = asyncio.Lock()
    
    async def create_session(self, request: SessionCreateRequest) -> SessionResponse:
        """
        Táº¡o session má»›i vÃ  load embeddings vÃ o VRAM.
        
        Step 1: Receive embeddings from Backend tenant DB
        Step 2: Load embeddings into VRAM (GPU memory)
        
        Args:
            request: ThÃ´ng tin táº¡o session vá»›i student_codes
            
        Returns:
            ThÃ´ng tin session Ä‘Ã£ táº¡o
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
            
            #         Step 1: Receive embeddings from Backend tenant DB
            try:
                embeddings_data = request.face_embeddings
                if not embeddings_data:
                    raise ValueError("face_embeddings is required")
                
                # BÆ¯á»šC 2: Load vÃ o VRAM
                await self._load_embeddings_to_vram(session_data, embeddings_data)
                
                session_data.embeddings_loaded = True
                
                # BÆ¯á»šC 3: âœ… Táº¡o per-session Tracker vÃ  Validator
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
    
    async def _load_embeddings_to_vram(
        self,
        session_data: SessionData,
        embeddings_data: List[Dict[str, Any]]
    ) -> None:
        """
                Step 2: Load embeddings into VRAM (GPU memory)
        Gá»™p 500 vectors thÃ nh 1 tensor vÃ  lÆ°u vÃ o SessionData.
        
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
        
        # âš ï¸ CRITICAL: L2 normalization - MUST normalize before comparison
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
        BÆ¯á»šC 3: Khá»Ÿi táº¡o per-session Tracker vÃ  Validator
        
        Má»—i session cÃ³:
        - FaceTracker riÃªng (khÃ´ng chia sáº» tracks giá»¯a sessions)
        - RecognitionValidator riÃªng (khÃ´ng chia sáº» history/debounce)
        
        Args:
            session_data: Session data to initialize
        """
        from app.services.tracker import create_face_tracker
        from app.services.recognition_validator import create_recognition_validator
        from app.core.config import settings
        
        try:
            # Táº¡o FaceTracker per-session vá»›i IoU enabled
            face_tracker = create_face_tracker(
                max_disappeared=30,
                distance_threshold=200,
                iou_threshold=0.3,
                use_iou=True  # âœ… Sá»­ dá»¥ng IoU thay vÃ¬ distance
            )
            
            auto_adjust_to_fps = getattr(settings, 'RECOGNITION_AUTO_ADJUST_TO_FPS', False)
            target_fps = getattr(settings, 'RECOGNITION_TARGET_FPS', 5.0)

            # Táº¡o RecognitionValidator per-session
            recognition_validator = create_recognition_validator(
                face_tracker=face_tracker,
                confirmation_threshold=getattr(settings, 'RECOGNITION_CONFIRMATION_THRESHOLD', 3),
                window_size=getattr(settings, 'RECOGNITION_WINDOW_SIZE', 5),
                min_avg_confidence=getattr(settings, 'RECOGNITION_MIN_AVG_CONFIDENCE', 0.5),
                min_success_rate=getattr(settings, 'RECOGNITION_MIN_FRAME_SUCCESS_RATE', 0.6),
                debounce_seconds=getattr(settings, 'RECOGNITION_DEBOUNCE_SECONDS', 30),
                auto_adjust_to_fps=auto_adjust_to_fps,
                target_fps=target_fps
            )
            
            # LÆ°u vÃ o session data
            session_data.face_tracker = face_tracker
            session_data.recognition_validator = recognition_validator
            
            self.logger.info(
                "Per-session Tracker and Validator initialized",
                session_id=session_data.session_id,
                use_iou=True,
                auto_adjust_fps=auto_adjust_to_fps,
                target_fps=target_fps
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
        Láº¥y thÃ´ng tin session (DTO)
        
        Args:
            session_id: ID cá»§a session
            
        Returns:
            ThÃ´ng tin session hoáº·c None náº¿u khÃ´ng tá»“n táº¡i
        """
        async with self._lock:
            session_data = self._sessions.get(session_id)
            if not session_data:
                return None
            
            # Kiá»ƒm tra session cÃ³ háº¿t háº¡n khÃ´ng
            if self._is_session_expired(session_data):
                session_data.status = "expired"
            
            return self._session_data_to_response(session_data)
    
    async def get_session_data(self, session_id: str) -> Optional[SessionData]:
        """
        Láº¥y SessionData thá»±c (vá»›i embeddings) - for internal use
        
        Args:
            session_id: ID cá»§a session
            
        Returns:
            SessionData object hoáº·c None náº¿u khÃ´ng tá»“n táº¡i
        """
        async with self._lock:
            session_data = self._sessions.get(session_id)
            if not session_data:
                return None
            
            # Kiá»ƒm tra session cÃ³ háº¿t háº¡n khÃ´ng
            if self._is_session_expired(session_data):
                session_data.status = "expired"
            
            return session_data
    
    async def delete_session(self, session_id: str) -> bool:
        """
        XÃ³a session
        
        Args:
            session_id: ID cá»§a session
            
        Returns:
            True náº¿u xÃ³a thÃ nh cÃ´ng, False náº¿u khÃ´ng tá»“n táº¡i
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
        TÄƒng sá»‘ lÆ°á»£ng frame Ä‘Ã£ xá»­ lÃ½
        
        Args:
            session_id: ID cá»§a session
            
        Returns:
            True náº¿u thÃ nh cÃ´ng, False náº¿u session khÃ´ng tá»“n táº¡i
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
        """Láº¥y sá»‘ lÆ°á»£ng session Ä‘ang hoáº¡t Ä‘á»™ng"""
        async with self._lock:
            active_count = 0
            for session_data in self._sessions.values():
                if session_data.status == "active" and not self._is_session_expired(session_data):
                    active_count += 1
            return active_count
    
    async def cleanup_expired_sessions(self) -> int:
        """
        Dá»n dáº¹p cÃ¡c session Ä‘Ã£ háº¿t háº¡n
        
        Returns:
            Sá»‘ lÆ°á»£ng session Ä‘Ã£ Ä‘Æ°á»£c dá»n dáº¹p
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
        LÆ°u face crop cá»§a student Ä‘Ã£ validated vÃ o session memory.
        Sáº½ Ä‘Æ°á»£c láº¥y ra khi end_session Ä‘á»ƒ upload S3.
        
        âœ… MEMORY OPTIMIZATION: LÆ°u dáº¡ng JPEG compressed
        
        Args:
            session_id: ID cá»§a session
            student_code: MÃ£ sinh viÃªn
            face_crop: áº¢nh khuÃ´n máº·t crop (numpy array RGB)
            
        Returns:
            True náº¿u lÆ°u thÃ nh cÃ´ng
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                self.logger.warning(f"Session not found: {session_id}")
                return False
            
            # âœ… MEMORY: Compress to JPEG bytes (~ 10-20x smaller)
            jpeg_bytes = _compress_face_crop(face_crop, quality=85)
            
            # LÆ°u compressed crop (overwrite náº¿u Ä‘Ã£ cÃ³)
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
        Láº¥y táº¥t cáº£ face crops cá»§a students Ä‘Ã£ validated trong session.
        
        Args:
            session_id: ID cá»§a session
            
        Returns:
            Dict mapping student_code -> face_crop (numpy array RGB)
        """
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                self.logger.warning(f"Session not found: {session_id}")
                return {}
            
            # âœ… Decompress when retrieving
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
        LÆ°u spoof face crop vÃ o session memory Ä‘á»ƒ upload lÃªn S3 khi end_session.
        
        âš ï¸ QUALITY FILTER Ä‘á»ƒ trÃ¡nh spam:
        - Pháº£i cÃ¡ch Ã­t nháº¥t 15 frame so vá»›i áº£nh trÆ°á»›c Ä‘Ã³
        - âœ… MEMORY: Giá»›i háº¡n tá»‘i Ä‘a 50 áº£nh spoof má»—i session
        - âœ… MEMORY: LÆ°u dáº¡ng JPEG compressed
        
        Args:
            session_id: ID cá»§a session
            face_crop: áº¢nh khuÃ´n máº·t crop (numpy array RGB)
            spoofing_type: Loáº¡i giáº£ máº¡o ('spoof', 'print', 'replay', etc.)
            spoofing_confidence: Äá»™ tin cáº­y cá»§a prediction
            frame_count: Frame sá»‘ máº¥y phÃ¡t hiá»‡n
            
        Returns:
            True náº¿u lÆ°u thÃ nh cÃ´ng, False náº¿u bá»‹ skip
        """
        from app.core.config import settings
        
        MIN_FRAME_GAP = 15     # Pháº£i cÃ¡ch Ã­t nháº¥t 15 frame
        MAX_SPOOF_CROPS = settings.MEMORY_MAX_SPOOF_CROPS  # âœ… Tá»« config
        
        async with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                self.logger.warning(f"Session not found: {session_id}")
                return False
            
            # âš ï¸ QUALITY CHECK: ChÆ°a Ä‘á»§ khoáº£ng cÃ¡ch frame?
            if session.spoof_faces_crops:
                last_frame = session.spoof_faces_crops[-1].frame_count
                if frame_count - last_frame < MIN_FRAME_GAP:
                    self.logger.debug(
                        f"Frame gap too small ({frame_count - last_frame} < {MIN_FRAME_GAP}), skipping",
                        session_id=session_id,
                        frame_count=frame_count
                    )
                    return False
            
            # âœ… MEMORY: Giá»›i háº¡n tá»‘i Ä‘a sá»‘ áº£nh spoof lÆ°u trá»¯
            if len(session.spoof_faces_crops) >= MAX_SPOOF_CROPS:
                self.logger.warning(
                    f"Max spoof crops reached ({MAX_SPOOF_CROPS}), skipping",
                    session_id=session_id,
                    frame_count=frame_count
                )
                return False
            
            # âœ… MEMORY: Compress to JPEG bytes
            jpeg_bytes = _compress_face_crop(face_crop, quality=80)  # Lower quality for spoofs
            
            # Táº¡o SpoofFaceCrop object vá»›i compressed data
            spoof_data = SpoofFaceCrop(
                face_crop_jpeg=jpeg_bytes,
                spoofing_type=spoofing_type,
                spoofing_confidence=spoofing_confidence,
                detected_at=datetime.now(timezone.utc),
                frame_count=frame_count
            )
            
            # LÆ°u vÃ o list
            session.spoof_faces_crops.append(spoof_data)
            
            self.logger.info(
                f"âœ… Stored spoof #{len(session.spoof_faces_crops)} (compressed)",
                session_id=session_id,
                spoofing_type=spoofing_type,
                confidence=f"{spoofing_confidence:.1%}",
                frame_count=frame_count,
                compressed_size=len(jpeg_bytes)
            )
            
            return True
    
    async def get_spoof_faces_crops(self, session_id: str) -> List[SpoofFaceCrop]:
        """
        Láº¥y táº¥t cáº£ spoof face crops trong session.
        
        Args:
            session_id: ID cá»§a session
            
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
        """Chuyá»ƒn Ä‘á»•i SessionData thÃ nh SessionResponse"""
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
        """Kiá»ƒm tra session cÃ³ háº¿t háº¡n khÃ´ng"""
        if session_data.status != "active":
            return True
        
        expiry_time = session_data.created_at + timedelta(minutes=session_data.max_duration_minutes)
        return datetime.now(timezone.utc) > expiry_time


# Global session manager instance
session_manager = SessionManager()
