"""
Face Engine - Orchestrator cho face detection vÃ  recognition services
"""
import base64
import io
from typing import List, Optional, Dict, Any, Set
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import cv2
from PIL import Image
import torch

from app.models.schemas import Detection
from app.core.logging import LoggerMixin
from app.core.config import settings


class FaceEngine(LoggerMixin):
    """
    Face detection vÃ  recognition engine orchestrator
    Káº¿t há»£p cÃ¡c services: detection, recognition, embedding management, validation
    """
    
    def __init__(
        self,
        detector_service=None,
        recognizer_service=None,
        embedding_manager=None,
        recognition_validator=None,
        anti_spoofing_service=None  # âœ… THÃŠM PARAMETER Má»šI
    ):
        """
        Khá»Ÿi táº¡o FaceEngine
        
        Args:
            detector_service: FaceDetectionService instance (optional)
            recognizer_service: FaceRecognitionService instance (optional)
            embedding_manager: EmbeddingManager instance (optional)
            recognition_validator: RecognitionValidator instance (optional)
            anti_spoofing_service: AntiSpoofingService instance (optional) âœ… Má»šI
        """
        super().__init__()
        
        self.detector = detector_service
        self.recognizer = recognizer_service
        self.embedding_manager = embedding_manager
        self.validator = recognition_validator
        self.anti_spoofing = anti_spoofing_service  # âœ… THÃŠM ATTRIBUTE Má»šI
        self._next_track_id = 1
        
        self.logger.info(
            "FaceEngine initialized",
            has_detector=self.detector is not None,
            has_recognizer=self.recognizer is not None,
            has_embedding_manager=self.embedding_manager is not None,
            has_validator=self.validator is not None,
            has_anti_spoofing=self.anti_spoofing is not None  # âœ… THÃŠM LOG Má»šI
        )
    
    async def detect_faces(self, frame_data: Optional[bytes] = None) -> tuple[List[Detection], List[np.ndarray], np.ndarray]:
        """
        Detect faces in a frame and return crops for the WebSocket flow
        
        Args:
            frame_data: Raw frame data (bytes hoáº·c base64)
            
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
            
            # âœ… MEMORY OPTIMIZATION: Resize image náº¿u quÃ¡ lá»›n (giáº£m VRAM usage)
            h, w = image_rgb.shape[:2]
            max_size = 1280  # Max dimension
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                self.logger.debug(f"Resized image from {w}x{h} to {new_w}x{new_h}")
            
            # âœ… Ensure contiguous memory layout
            if not image_rgb.flags['C_CONTIGUOUS']:
                image_rgb = np.ascontiguousarray(image_rgb)
            
            # Detect faces and return crops for recognition/anti-spoofing
            detections, crops, _ = await self.detector.detect_faces_async(
                image_rgb,
                return_crops=True
            )
            
            # âœ… MEMORY: Giáº£i phÃ³ng image_array sau khi decode xong
            del image_array
            del image_bgr
            
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
                    track_id=None  # âœ… Let tracker assign track_id
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
        crops: List[np.ndarray],  # âœ… THÃŠM CROPS PARAMETER
        gallery_embeddings: Optional[Any] = None,  # torch.Tensor
        gallery_labels: Optional[List[str]] = None,
    ) -> List[Detection]:
        """
        Nháº­n diá»‡n faces dá»±a trÃªn embeddings database.

        âœ… Phase 3: True Batch GPU Inference - toÃ n bá»™ N faces Ä‘Æ°á»£c gom
        thÃ nh 1 tensor vÃ  Ä‘áº©y qua GPU trong 1 láº§n duy nháº¥t thay vÃ¬ N láº§n.

        Args:
            detections: Danh sÃ¡ch detections tá»« detect_faces
            crops: Danh sÃ¡ch face crops tá»« detect_faces âœ…
            gallery_embeddings: Gallery embeddings tensor (N, 512) on GPU [from session]
            gallery_labels: Gallery labels (student codes) [from session]

        Returns:
            Danh sÃ¡ch detections vá»›i thÃ´ng tin recognition
        """
        if self.recognizer is None:
            self.logger.warning("Recognizer not initialized")
            return detections

        use_session_embeddings = gallery_embeddings is not None and gallery_labels is not None
        if not use_session_embeddings:
            self.logger.warning("No session embeddings provided - using internal database (legacy)")

        if not crops or len(crops) != len(detections):
            self.logger.warning(
                f"Crops mismatch: {len(crops) if crops else 0} crops vs {len(detections)} detections"
            )
            return detections

        try:
            # â”€â”€ Thu tháº­p valid crops vÃ  index tÆ°Æ¡ng á»©ng â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            valid_crops: List[np.ndarray] = []
            valid_indices: List[int] = []

            for i, (detection, crop) in enumerate(zip(detections, crops)):
                if crop is not None and crop.size > 0:
                    valid_crops.append(crop)
                    valid_indices.append(i)

            if not valid_crops:
                self.logger.debug("No valid crops to recognize")
                return detections

            # âœ… Phase 3: 1 GPU call cho toÃ n bá»™ batch
            self.logger.debug(f"Running TRUE BATCH recognition for {len(valid_crops)} faces (1 GPU call)")
            results = await self.recognizer.identify_batch_async(
                valid_crops,
                gallery_embeddings=gallery_embeddings,
                gallery_labels=gallery_labels,
            )

            # â”€â”€ GÃ¡n káº¿t quáº£ vÃ o detections â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            for idx, result in zip(valid_indices, results):
                if result is None:
                    continue
                if isinstance(result, Exception):
                    self.logger.warning("Failed to recognize face", detection_index=idx, error=str(result))
                    continue

                if result.get("person") != "Unknown":
                    detections[idx].student_code = result.get("person")
                    detections[idx].student_name = result.get("person")
                    detections[idx].recognition_confidence = float(result.get("confidence", 0.0))

            recognized_count = sum(1 for d in detections if d.student_id)
            self.logger.debug(f"Batch recognized {recognized_count}/{len(detections)} faces")

            return detections

        except Exception as e:
            self.logger.error("Batch face recognition failed", error=str(e))
            return detections

    
    async def extract_embeddings(
        self,
        frame_data: bytes,
        bbox: List[float]
    ) -> Optional[List[float]]:
        """
        TrÃ­ch xuáº¥t face embeddings tá»« bounding box
        
        Args:
            frame_data: Raw frame data
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Face embedding vector hoáº·c None
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
        Load embeddings database tá»« data (tá»« S3)
        
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
        Load embeddings tá»« thÆ° má»¥c
        
        Args:
            embedding_dir: ÄÆ°á»ng dáº«n thÆ° má»¥c chá»©a embeddings
            
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
        """Láº¥y thá»‘ng kÃª database"""
        if self.recognizer is None:
            return {"num_people": 0, "total_vectors": 0}
        
        return self.recognizer.get_database_stats()
    
    async def update_recognition_history(
        self,
        detections: List[Detection],
        timestamp: datetime
    ):
        """
        Cáº­p nháº­t lá»‹ch sá»­ nháº­n diá»‡n vÃ o validator
        
        Args:
            detections: Danh sÃ¡ch detections vá»›i recognition info
            timestamp: Thá»i Ä‘iá»ƒm nháº­n diá»‡n
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
        Láº¥y danh sÃ¡ch sinh viÃªn Ä‘Ã£ Ä‘Æ°á»£c validated (pass táº¥t cáº£ Ä‘iá»u kiá»‡n)
        
        Args:
            current_time: Thá»i Ä‘iá»ƒm hiá»‡n táº¡i
            
        Returns:
            Set cÃ¡c student_id Ä‘Ã£ Ä‘Æ°á»£c validated vÃ  chÆ°a gá»­i callback
        """
        if self.validator is None:
            self.logger.warning("Validator not initialized")
            return set()
        
        # Cleanup old confirmations
        self.validator.cleanup_old_confirmations(current_time)
        
        # Láº¥y cÃ¡c sinh viÃªn má»›i Ä‘Æ°á»£c validated
        validated_students = await self.validator.get_newly_confirmed_students(current_time)
        
        if validated_students:
            self.logger.info(f"âœ… Validated students: {list(validated_students)}")
        
        return validated_students
    
    # ============================================================
    # âœ… ANTI-SPOOFING CHECK - BATCH PROCESSING
    # ============================================================
    async def check_anti_spoofing(
        self,
        face_crops: List[np.ndarray]
    ) -> List[Dict[str, Any]]:
        """
        Kiá»ƒm tra anti-spoofing cho danh sÃ¡ch face crops - BATCH PROCESSING
        Model má»›i: ResNet18_MSFF_AntiSpoof vá»›i 2 classes (real/spoof)
        
        Args:
            face_crops: Danh sÃ¡ch áº£nh khuÃ´n máº·t Ä‘Ã£ crop (numpy arrays RGB)
            
        Returns:
            Danh sÃ¡ch káº¿t quáº£ anti-spoofing:
            [
                {
                    'is_live': bool,      # True náº¿u real, False náº¿u spoof
                    'label': str,         # 'real' hoáº·c 'spoof'
                    'confidence': float   # 0.0 - 1.0
                },
                ...
            ]
            
        Note:
            Gom táº¥t cáº£ crops thÃ nh má»™t batch vÃ  cháº¡y model má»™t láº§n.
        """
        if not face_crops:
            return []
        
        if self.anti_spoofing is None:
            self.logger.warning("Anti-spoofing service not initialized, assuming all faces are real")
            return [
                {
                    'is_live': True,
                    'label': 'real',
                    'confidence': 1.0
                }
                for _ in face_crops
            ]
        
        try:
            self.logger.debug(f"Running TRUE BATCH anti-spoofing for {len(face_crops)} faces")
            results = await self.anti_spoofing.predict_batch_async(face_crops)

            for i, result in enumerate(results):
                if not result.get('is_live', True):
                    self.logger.warning(
                        f"ðŸš¨ Spoof face detected in crop #{i}",
                        label=result.get('label'),
                        confidence=f"{result.get('confidence', 0.0):.3f}"
                    )
            
            # Log summary
            total = len(results)
            live_count = sum(1 for r in results if r['is_live'])
            spoof_count = total - live_count
            
            self.logger.debug(
                "Anti-spoofing batch completed",
                total_faces=total,
                real_faces=live_count,
                spoof_faces=spoof_count
            )
            
            return results
            
        except Exception as e:
            self.logger.error("Batch anti-spoofing failed", error=str(e))
            # Fallback: return all as real
            return [
                {'is_live': True, 'label': 'unknown', 'confidence': 0.0}
                for _ in face_crops
            ]


# Global face engine instance
face_engine: Optional[FaceEngine] = None


def initialize_face_engine(
    detector_service=None,
    recognizer_service=None,
    embedding_manager=None,
    recognition_validator=None,
    anti_spoofing_service=None  # âœ… THÃŠM PARAMETER Má»šI
) -> FaceEngine:
    """
    Khá»Ÿi táº¡o global face engine
    
    Args:
        detector_service: FaceDetectionService instance
        recognizer_service: FaceRecognitionService instance
        embedding_manager: EmbeddingManager instance
        recognition_validator: RecognitionValidator instance
        anti_spoofing_service: AntiSpoofingService instance âœ… Má»šI
        
    Returns:
        FaceEngine instance
    """
    global face_engine
    face_engine = FaceEngine(
        detector_service=detector_service,
        recognizer_service=recognizer_service,
        embedding_manager=embedding_manager,
        recognition_validator=recognition_validator,
        anti_spoofing_service=anti_spoofing_service  # âœ… THÃŠM PARAMETER Má»šI
    )
    return face_engine


def get_face_engine() -> FaceEngine:
    """
    Láº¥y global face engine instance
    
    Returns:
        FaceEngine instance
        
    Raises:
        RuntimeError: Náº¿u face engine chÆ°a Ä‘Æ°á»£c khá»Ÿi táº¡o
    """
    if face_engine is None:
        raise RuntimeError("FaceEngine chÆ°a Ä‘Æ°á»£c khá»Ÿi táº¡o. Gá»i initialize_face_engine() trÆ°á»›c.")
    return face_engine
