"""
Face Engine - Orchestrator cho face detection và recognition services
"""
import base64
import io
from typing import List, Optional, Dict, Any, Set
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import cv2
from PIL import Image

from app.models.schemas import Detection
from app.core.logging import LoggerMixin
from app.core.config import settings


class FaceEngine(LoggerMixin):
    """
    Face detection và recognition engine orchestrator
    Kết hợp các services: detection, recognition, embedding management, validation
    """
    
    def __init__(
        self,
        detector_service=None,
        recognizer_service=None,
        embedding_manager=None,
        recognition_validator=None,
        anti_spoofing_service=None  # ✅ THÊM PARAMETER MỚI
    ):
        """
        Khởi tạo FaceEngine
        
        Args:
            detector_service: FaceDetectionService instance (optional)
            recognizer_service: FaceRecognitionService instance (optional)
            embedding_manager: EmbeddingManager instance (optional)
            recognition_validator: RecognitionValidator instance (optional)
            anti_spoofing_service: AntiSpoofingService instance (optional) ✅ MỚI
        """
        super().__init__()
        
        self.detector = detector_service
        self.recognizer = recognizer_service
        self.embedding_manager = embedding_manager
        self.validator = recognition_validator
        self.anti_spoofing = anti_spoofing_service  # ✅ THÊM ATTRIBUTE MỚI
        self._next_track_id = 1
        
        self.logger.info(
            "FaceEngine initialized",
            has_detector=self.detector is not None,
            has_recognizer=self.recognizer is not None,
            has_embedding_manager=self.embedding_manager is not None,
            has_validator=self.validator is not None,
            has_anti_spoofing=self.anti_spoofing is not None  # ✅ THÊM LOG MỚI
        )
    
    async def detect_faces(self, frame_data: Optional[bytes] = None) -> tuple[List[Detection], List[np.ndarray], np.ndarray]:
        """
        Detect faces trong frame - TRẢ VỀ CROPS giống endpoint /detect
        
        Args:
            frame_data: Raw frame data (bytes hoặc base64)
            
        Returns:
            Tuple of (detections, crops, original_image)
        """
        if self.detector is None:
            self.logger.warning("Detector not initialized - returning empty list")
            return [], [], None
        
        try:
            # Convert frame_data to numpy array
            if frame_data is None:
                # Return empty if no data
                return [], [], None
            
            # Decode image from bytes
            image_array = np.frombuffer(frame_data, dtype=np.uint8)
            image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image_bgr is None:
                self.logger.warning("Failed to decode image")
                return [], [], None
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            # Detect faces - ✅ RETURN CROPS GIỐNG /detect ENDPOINT
            detections, crops, _ = await self.detector.detect_faces_async(
                image_rgb,
                return_crops=True
            )
            
            # Handle None results
            if detections is None:
                detections = []
            if crops is None:
                crops = []
            
            # Convert to API Detection schema
            api_detections = []
            for det in detections:
                api_det = Detection(
                    bbox=[float(x) for x in det.bbox],
                    confidence=float(det.confidence),
                    track_id=None  # ✅ Let tracker assign track_id
                )
                api_detections.append(api_det)
            
            self.logger.debug(f"Detected {len(api_detections)} faces")
            return api_detections, crops, image_rgb
            
        except Exception as e:
            self.logger.error("Face detection failed", error=str(e))
            return [], [], None
    
    async def recognize_faces(
        self,
        detections: List[Detection],
        crops: List[np.ndarray],  # ✅ THÊM CROPS PARAMETER
        gallery_embeddings: Optional[Any] = None,  # torch.Tensor
        gallery_labels: Optional[List[str]] = None,
    ) -> List[Detection]:
        """
        Nhận diện faces dựa trên embeddings database - GIỐNG ENDPOINT /detect
        
        Args:
            detections: Danh sách detections từ detect_faces
            crops: Danh sách face crops từ detect_faces ✅
            gallery_embeddings: Gallery embeddings tensor (N, 512) on GPU [from session]
            gallery_labels: Gallery labels (student codes) [from session]
            
        Returns:
            Danh sách detections với thông tin recognition
            
        Note:
            If gallery_embeddings and gallery_labels are provided, they will be used
            for recognition (session-based). Otherwise, recognizer's internal database
            will be used (legacy mode).
        """
        if self.recognizer is None:
            self.logger.warning("Recognizer not initialized")
            return detections
        
        # Check if we have session-based embeddings
        use_session_embeddings = gallery_embeddings is not None and gallery_labels is not None
        
        if not use_session_embeddings:
            self.logger.warning("No session embeddings provided - using internal database (legacy)")
        
        # Check if we have crops
        if not crops or len(crops) != len(detections):
            self.logger.warning(f"Crops mismatch: {len(crops)} crops vs {len(detections)} detections")
            return detections
        
        try:
            # ✅ RECOGNIZE GIỐNG ENDPOINT /detect - SỬ DỤNG CROPS SẴN CÓ
            for detection, crop in zip(detections, crops):
                try:
                    if crop is None or crop.size == 0:
                        continue
                    
                    # Identify face - pass session embeddings if available
                    identity = await self.recognizer.identify_async(
                        crop,  # ✅ SỬ DỤNG CROP SẴN CÓ - KHÔNG DECODE/CROP LẠI
                        gallery_embeddings=gallery_embeddings,
                        gallery_labels=gallery_labels
                    )
                    
                    if identity and identity.get('person') != 'Unknown':
                        detection.student_code = identity.get('person')  # ✅ Use student_code
                        detection.student_name = identity.get('person')  # ✅ THÊM student_name
                        detection.recognition_confidence = float(identity.get('confidence', 0.0))
                    
                except Exception as e:
                    self.logger.warning(
                        "Failed to recognize face",
                        track_id=detection.track_id,
                        error=str(e)
                    )
                    continue
            
            recognized_count = sum(1 for d in detections if d.student_id)
            self.logger.debug(f"Recognized {recognized_count}/{len(detections)} faces")
            
            return detections
            
        except Exception as e:
            self.logger.error("Face recognition failed", error=str(e))
            return detections
    
    async def extract_embeddings(
        self,
        frame_data: bytes,
        bbox: List[float]
    ) -> Optional[List[float]]:
        """
        Trích xuất face embeddings từ bounding box
        
        Args:
            frame_data: Raw frame data
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Face embedding vector hoặc None
        """
        if self.recognizer is None:
            self.logger.warning("Recognizer not initialized")
            return None
        
        try:
            # Decode image
            image_array = np.frombuffer(frame_data, dtype=np.uint8)
            image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            # Crop face
            x1, y1, x2, y2 = [int(x) for x in bbox]
            face_crop = image_rgb[y1:y2, x1:x2]
            
            if face_crop.size == 0:
                return None
            
            # Extract features - returns np.ndarray (512,)
            embedding = self.recognizer.extract_features(face_crop)
            
            self.logger.debug("Extracted face embedding", bbox=bbox)
            # Convert numpy array to list
            return embedding.tolist() if embedding.ndim == 1 else embedding.squeeze().tolist()
            
        except Exception as e:
            self.logger.error("Embedding extraction failed", error=str(e))
            return None
    
    async def load_embeddings_from_data(self, embeddings_data: bytes) -> dict:
        """
        Load embeddings database từ data (từ S3)
        
        Args:
            embeddings_data: Raw embeddings data (npz format)
            
        Returns:
            Dictionary mapping student_id -> embedding
        """
        try:
            # Load from npz bytes
            buffer = io.BytesIO(embeddings_data)
            data = np.load(buffer)
            
            embeddings_db = {key: data[key] for key in data.keys()}
            
            self.logger.info(f"Loaded embeddings for {len(embeddings_db)} students")
            return embeddings_db
            
        except Exception as e:
            self.logger.error("Failed to load embeddings", error=str(e))
            return {}
    
    def load_embeddings_from_directory(self, embedding_dir: Path) -> dict:
        """
        Load embeddings từ thư mục
        
        Args:
            embedding_dir: Đường dẫn thư mục chứa embeddings
            
        Returns:
            Database dictionary
        """
        if self.recognizer is None:
            self.logger.warning("Recognizer not initialized")
            return {}
        
        try:
            database = self.recognizer.load_embedding_directory(embedding_dir)
            return database
        except Exception as e:
            self.logger.error("Failed to load embeddings from directory", error=str(e))
            return {}
    
    def get_database_stats(self) -> Dict[str, int]:
        """Lấy thống kê database"""
        if self.recognizer is None:
            return {"num_people": 0, "total_vectors": 0}
        
        return self.recognizer.get_database_stats()
    
    async def update_recognition_history(
        self,
        detections: List[Detection],
        timestamp: datetime
    ):
        """
        Cập nhật lịch sử nhận diện vào validator
        
        Args:
            detections: Danh sách detections với recognition info
            timestamp: Thời điểm nhận diện
        """
        if self.validator is None:
            self.logger.warning("Validator is None - skipping recognition history update")
            return
        
        updated_count = 0
        for detection in detections:
            if detection.track_id is not None:
                await self.validator.add_recognition(
                    track_id=detection.track_id,
                    student_id=detection.student_id,
                    confidence=detection.recognition_confidence or 0.0,
                    timestamp=timestamp
                )
                updated_count += 1
                self.logger.debug(
                    f"Added recognition: track_id={detection.track_id}, "
                    f"student_id={detection.student_id}, "
                    f"confidence={detection.recognition_confidence or 0.0:.3f}"
                )
        
        if updated_count > 0:
            self.logger.info(f"Updated {updated_count} recognition records in validator")
    
    async def get_validated_students(
        self,
        current_time: datetime
    ) -> Set[str]:
        """
        Lấy danh sách sinh viên đã được validated (pass tất cả điều kiện)
        
        Args:
            current_time: Thời điểm hiện tại
            
        Returns:
            Set các student_id đã được validated và chưa gửi callback
        """
        if self.validator is None:
            self.logger.warning("Validator not initialized")
            return set()
        
        # Cleanup old confirmations
        self.validator.cleanup_old_confirmations(current_time)
        
        # Lấy các sinh viên mới được validated
        validated_students = await self.validator.get_newly_confirmed_students(current_time)
        
        if validated_students:
            self.logger.info(f"✅ Validated students: {list(validated_students)}")
        
        return validated_students
    
    # ============================================================
    # ✅ ANTI-SPOOFING CHECK - Model Mới
    # ============================================================
    async def check_anti_spoofing(
        self,
        face_crops: List[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """
        Kiểm tra anti-spoofing cho danh sách face crops
        Model mới: ResNet18_MSFF_AntiSpoof với 2 classes (real/spoof)
        
        Args:
            face_crops: Danh sách ảnh khuôn mặt đã crop (numpy arrays RGB)
            
        Returns:
            Danh sách kết quả anti-spoofing:
            [
                {
                    'is_live': bool,      # True nếu real, False nếu spoof
                    'label': str,         # 'real' hoặc 'spoof'
                    'confidence': float   # 0.0 - 1.0
                },
                ...
            ]
        """
        if self.anti_spoofing is None:
            self.logger.warning("Anti-spoofing service not initialized, assuming all faces are real")
            # Trả về tất cả là real nếu không có service
            return [
                {
                    'is_live': True,
                    'label': 'real',
                    'confidence': 1.0
                }
                for _ in face_crops
            ]
        
        results = []
        for i, crop in enumerate(face_crops):
            try:
                # ✅ Call anti-spoofing service - Trả về Dict format mới
                result = await self.anti_spoofing.predict_async(crop)
                
                results.append(result)
                
                # Log chi tiết
                if not result['is_live']:
                    self.logger.warning(
                        f"🚨 Spoof face detected in crop #{i}",
                        label=result['label'],
                        confidence=f"{result['confidence']:.3f}"
                    )
                else:
                    self.logger.debug(
                        f"✅ Real face in crop #{i}",
                        confidence=f"{result['confidence']:.3f}"
                    )
                
            except Exception as e:
                self.logger.error(
                    f"Anti-spoofing failed for crop #{i}",
                    error=str(e),
                    exc_info=True
                )
                # Fallback: assume real nếu có lỗi (safe default)
                results.append({
                    'is_live': True,
                    'label': 'unknown',
                    'confidence': 0.0
                })
        
        # Log summary
        total = len(results)
        live_count = sum(1 for r in results if r['is_live'])
        spoof_count = total - live_count
        
        self.logger.info(
            "Anti-spoofing batch completed",
            total_faces=total,
            real_faces=live_count,
            spoof_faces=spoof_count
        )
        
        return results


# Global face engine instance
face_engine: Optional[FaceEngine] = None


def initialize_face_engine(
    detector_service=None,
    recognizer_service=None,
    embedding_manager=None,
    recognition_validator=None,
    anti_spoofing_service=None  # ✅ THÊM PARAMETER MỚI
) -> FaceEngine:
    """
    Khởi tạo global face engine
    
    Args:
        detector_service: FaceDetectionService instance
        recognizer_service: FaceRecognitionService instance
        embedding_manager: EmbeddingManager instance
        recognition_validator: RecognitionValidator instance
        anti_spoofing_service: AntiSpoofingService instance ✅ MỚI
        
    Returns:
        FaceEngine instance
    """
    global face_engine
    face_engine = FaceEngine(
        detector_service=detector_service,
        recognizer_service=recognizer_service,
        embedding_manager=embedding_manager,
        recognition_validator=recognition_validator,
        anti_spoofing_service=anti_spoofing_service  # ✅ THÊM PARAMETER MỚI
    )
    return face_engine


def get_face_engine() -> FaceEngine:
    """
    Lấy global face engine instance
    
    Returns:
        FaceEngine instance
        
    Raises:
        RuntimeError: Nếu face engine chưa được khởi tạo
    """
    if face_engine is None:
        raise RuntimeError("FaceEngine chưa được khởi tạo. Gọi initialize_face_engine() trước.")
    return face_engine
