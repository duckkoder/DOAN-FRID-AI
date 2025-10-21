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
        recognition_validator=None
    ):
        """
        Khởi tạo FaceEngine
        
        Args:
            detector_service: FaceDetectionService instance (optional)
            recognizer_service: FaceRecognitionService instance (optional)
            embedding_manager: EmbeddingManager instance (optional)
            recognition_validator: RecognitionValidator instance (optional)
        """
        super().__init__()
        
        self.detector = detector_service
        self.recognizer = recognizer_service
        self.embedding_manager = embedding_manager
        self.validator = recognition_validator
        self._next_track_id = 1
        
        self.logger.info(
            "FaceEngine initialized",
            has_detector=self.detector is not None,
            has_recognizer=self.recognizer is not None,
            has_embedding_manager=self.embedding_manager is not None,
            has_validator=self.validator is not None
        )
    
    async def detect_faces(self, frame_data: Optional[bytes] = None) -> List[Detection]:
        """
        Detect faces trong frame
        
        Args:
            frame_data: Raw frame data (bytes hoặc base64)
            
        Returns:
            Danh sách detections
        """
        if self.detector is None:
            self.logger.warning("Detector not initialized - returning empty list")
            return []
        
        try:
            # Convert frame_data to numpy array
            if frame_data is None:
                # Return empty if no data
                return []
            
            # Decode image from bytes
            image_array = np.frombuffer(frame_data, dtype=np.uint8)
            image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image_bgr is None:
                self.logger.warning("Failed to decode image")
                return []
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            detections, _, _ = await self.detector.detect_faces_async(
                image_rgb,
                return_crops=False
            )
            
            # Convert to API Detection schema
            api_detections = []
            for det in detections:
                api_det = Detection(
                    bbox=[float(x) for x in det.bbox],
                    confidence=float(det.confidence),
                    track_id=self._next_track_id
                )
                self._next_track_id += 1
                api_detections.append(api_det)
            
            self.logger.debug(f"Detected {len(api_detections)} faces")
            return api_detections
            
        except Exception as e:
            self.logger.error("Face detection failed", error=str(e))
            return []
    
    async def recognize_faces(
        self,
        detections: List[Detection],
        frame_data: Optional[bytes] = None,
        gallery_embeddings: Optional[Any] = None,  # torch.Tensor
        gallery_labels: Optional[List[str]] = None,
    ) -> List[Detection]:
        """
        Nhận diện faces dựa trên embeddings database
        
        Args:
            detections: Danh sách detections từ detect_faces
            frame_data: Raw frame data để crop faces
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
        
        if frame_data is None:
            self.logger.warning("No frame data for face crops")
            return detections
        
        try:
            # Decode image
            image_array = np.frombuffer(frame_data, dtype=np.uint8)
            image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            # Crop and recognize each face
            for detection in detections:
                try:
                    # Crop face from bbox
                    x1, y1, x2, y2 = [int(x) for x in detection.bbox]
                    face_crop = image_rgb[y1:y2, x1:x2]
                    
                    if face_crop.size == 0:
                        continue
                    
                    # Identify face - pass session embeddings if available
                    identity = await self.recognizer.identify_async(
                        face_crop,
                        gallery_embeddings=gallery_embeddings,
                        gallery_labels=gallery_labels
                    )
                    
                    if identity and identity.get('person') != 'Unknown':
                        detection.student_id = identity.get('person')
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
            
            # Extract features
            embedding = self.recognizer.extract_features(face_crop)
            
            self.logger.debug("Extracted face embedding", bbox=bbox)
            return embedding.squeeze(0).tolist()
            
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
            return
        
        for detection in detections:
            if detection.track_id is not None:
                await self.validator.add_recognition(
                    track_id=detection.track_id,
                    student_id=detection.student_id,
                    confidence=detection.recognition_confidence or 0.0,
                    timestamp=timestamp
                )
    
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
        
        return validated_students


# Global face engine instance
face_engine: Optional[FaceEngine] = None


def initialize_face_engine(
    detector_service=None,
    recognizer_service=None,
    embedding_manager=None,
    recognition_validator=None
) -> FaceEngine:
    """
    Khởi tạo global face engine
    
    Args:
        detector_service: FaceDetectionService instance
        recognizer_service: FaceRecognitionService instance
        embedding_manager: EmbeddingManager instance
        recognition_validator: RecognitionValidator instance
        
    Returns:
        FaceEngine instance
    """
    global face_engine
    face_engine = FaceEngine(
        detector_service=detector_service,
        recognizer_service=recognizer_service,
        embedding_manager=embedding_manager,
        recognition_validator=recognition_validator
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
