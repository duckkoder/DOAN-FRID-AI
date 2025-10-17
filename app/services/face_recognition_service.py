"""
Face Recognition Service - Wrapper cho FaceRecognizer
"""
from typing import Dict, Optional, List, Any
import numpy as np
from pathlib import Path

from models.face_recognizer import FaceRecognizer
from app.core.config import settings
from app.core.logging import LoggerMixin


class FaceRecognitionService(LoggerMixin):
    """
    Service quản lý face recognition và identification
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        threshold: Optional[float] = None,
        knn_k: Optional[int] = None,
        enable_dynamic_threshold: Optional[bool] = None,
        per_identity_quantile: Optional[float] = None,
        per_identity_margin: Optional[float] = None,
        identity_threshold_min_scale: Optional[float] = None,
        identity_threshold_max_scale: Optional[float] = None,
    ):
        """
        Khởi tạo Face Recognition Service
        
        Args:
            checkpoint_path: Đường dẫn checkpoint model
            device: Device (cuda/cpu)
            threshold: Ngưỡng distance cho recognition
            knn_k: Số lượng K cho KNN
            enable_dynamic_threshold: Bật dynamic threshold
            per_identity_quantile: Quantile cho per-identity threshold
            per_identity_margin: Margin cho per-identity threshold
            identity_threshold_min_scale: Min scale cho identity threshold
            identity_threshold_max_scale: Max scale cho identity threshold
        """
        super().__init__()
        
        # Load config từ settings
        self.checkpoint_path = checkpoint_path or settings.RECOGNIZER_CHECKPOINT
        self.device = device or settings.RECOGNIZER_DEVICE
        self.threshold = threshold or settings.RECOGNIZER_THRESHOLD
        self.knn_k = knn_k or settings.RECOGNIZER_KNN_K
        self.min_confidence = settings.RECOGNIZER_MIN_CONFIDENCE
        self.min_vote_ratio = settings.RECOGNIZER_MIN_VOTE_RATIO
        self.require_stable = settings.RECOGNIZER_REQUIRE_STABLE
        
        if not self.checkpoint_path:
            raise ValueError("RECOGNIZER_CHECKPOINT phải được cấu hình trong settings hoặc truyền vào constructor")
        
        # Khởi tạo recognizer
        self.logger.info(
            "Initializing Face Recognizer",
            checkpoint=self.checkpoint_path,
            device=self.device,
            threshold=self.threshold
        )
        
        self.recognizer = FaceRecognizer(
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            threshold=self.threshold,
            knn_k=self.knn_k,
            enable_dynamic_threshold=enable_dynamic_threshold or settings.REC_ENABLE_DYNAMIC_THRESHOLD,
            per_identity_quantile=per_identity_quantile or settings.REC_IDENTITY_QUANTILE,
            per_identity_margin=per_identity_margin or settings.REC_IDENTITY_MARGIN,
            identity_threshold_min_scale=identity_threshold_min_scale or settings.REC_IDENTITY_MIN_SCALE,
            identity_threshold_max_scale=identity_threshold_max_scale or settings.REC_IDENTITY_MAX_SCALE,
        )
        
        self.logger.info("Face Recognizer initialized successfully")
    
    def identify(
        self,
        face_crop: np.ndarray,
        tta: Optional[bool] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Nhận diện khuôn mặt
        
        Args:
            face_crop: Ảnh crop của khuôn mặt (RGB)
            tta: Bật Test Time Augmentation
            
        Returns:
            Dictionary chứa thông tin identification:
            - person: Tên người (hoặc "Unknown")
            - confidence: Độ tin cậy
            - distance: Khoảng cách embedding
            - vote_ratio: Tỷ lệ vote (nếu dùng KNN)
            - threshold: Ngưỡng được sử dụng
        """
        try:
            tta_enabled = tta if tta is not None else settings.TTA_ENABLED
            
            identity = self.recognizer.identify(face_crop, tta=tta_enabled)
            
            if identity:
                # Áp dụng filters
                conf = float(identity.get('confidence', 0.0))
                vote = float(identity.get('vote_ratio', 0.0))
                stable = bool(identity.get('stable', False))
                
                # Kiểm tra ngưỡng
                if (conf < self.min_confidence) or \
                   (vote < self.min_vote_ratio) or \
                   (self.require_stable and not stable):
                    identity['person'] = 'Unknown'
                    self.logger.debug(
                        "Identity filtered as Unknown",
                        confidence=conf,
                        vote_ratio=vote,
                        stable=stable
                    )
            
            return identity
            
        except RuntimeError as e:
            self.logger.warning("Recognition failed - database may be empty", error=str(e))
            return None
        except Exception as e:
            self.logger.error("Face recognition error", error=str(e))
            raise
    
    async def identify_async(
        self,
        face_crop: np.ndarray,
        tta: Optional[bool] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Async wrapper cho identify
        
        TODO: Implement thật sự async nếu cần
        """
        return self.identify(face_crop, tta)
    
    def extract_features(
        self,
        face_crop: np.ndarray,
        tta: Optional[bool] = None
    ) -> np.ndarray:
        """
        Trích xuất embedding features từ face crop
        
        Args:
            face_crop: Ảnh crop của khuôn mặt (RGB)
            tta: Bật TTA
            
        Returns:
            Embedding vector (numpy array)
        """
        try:
            tta_enabled = tta if tta is not None else settings.TTA_ENABLED
            embedding = self.recognizer.extract_features(face_crop, tta=tta_enabled)
            
            self.logger.debug("Features extracted", embedding_shape=embedding.shape)
            return embedding
            
        except Exception as e:
            self.logger.error("Feature extraction error", error=str(e))
            raise
    
    def load_embedding_directory(self, embedding_dir: Path) -> Dict[str, np.ndarray]:
        """
        Load embeddings từ thư mục
        
        Args:
            embedding_dir: Đường dẫn thư mục chứa embeddings
            
        Returns:
            Database dictionary {identity: embeddings}
        """
        try:
            database = self.recognizer.load_embedding_directory(embedding_dir)
            
            total_vectors = sum(t.shape[0] for t in database.values())
            num_people = len(database)
            
            self.logger.info(
                "Embeddings loaded from directory",
                embedding_dir=str(embedding_dir),
                num_people=num_people,
                total_vectors=total_vectors
            )
            
            return database
            
        except Exception as e:
            self.logger.error("Failed to load embeddings", embedding_dir=str(embedding_dir), error=str(e))
            raise
    
    def save_embedding(
        self,
        embedding_dir: Path,
        identity: str,
        embedding: np.ndarray
    ) -> Path:
        """
        Lưu embedding vào thư mục
        
        Args:
            embedding_dir: Thư mục lưu embeddings
            identity: Tên người
            embedding: Embedding vector
            
        Returns:
            Path đến file embedding đã lưu
        """
        try:
            embedding_path = self.recognizer.save_embedding(
                embedding_dir,
                identity,
                embedding
            )
            
            self.logger.debug(
                "Embedding saved",
                identity=identity,
                path=str(embedding_path)
            )
            
            return embedding_path
            
        except Exception as e:
            self.logger.error("Failed to save embedding", identity=identity, error=str(e))
            raise
    
    def build_database(self, database_dir: Path) -> None:
        """
        Build database từ thư mục ảnh
        
        Args:
            database_dir: Thư mục chứa ảnh (mỗi người một folder)
        """
        try:
            self.recognizer.build_database(database_dir)
            
            self.logger.info(
                "Database built from images",
                database_dir=str(database_dir)
            )
            
        except Exception as e:
            self.logger.error("Failed to build database", database_dir=str(database_dir), error=str(e))
            raise
    
    def sanitize_identity(self, identity: str) -> str:
        """
        Làm sạch tên identity (loại bỏ ký tự đặc biệt)
        
        Args:
            identity: Tên người cần làm sạch
            
        Returns:
            Tên đã được làm sạch
        """
        return self.recognizer.sanitize_identity(identity)
    
    def assess_face_quality(self, face_crop: np.ndarray) -> Dict[str, float]:
        """
        Đánh giá chất lượng khuôn mặt
        
        Args:
            face_crop: Ảnh crop của khuôn mặt
            
        Returns:
            Dictionary chứa các metrics chất lượng
        """
        try:
            quality_metrics = self.recognizer.assess_face_quality(face_crop)
            
            self.logger.debug(
                "Face quality assessed",
                quality_score=quality_metrics.get('quality_score', 0.0)
            )
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error("Face quality assessment error", error=str(e))
            raise
    
    @property
    def database(self) -> Dict[str, np.ndarray]:
        """Trả về database hiện tại"""
        return self.recognizer._database
    
    @property
    def has_database(self) -> bool:
        """Kiểm tra có database không"""
        return bool(self.recognizer._database)
    
    def get_database_stats(self) -> Dict[str, int]:
        """
        Lấy thống kê database
        
        Returns:
            Dictionary chứa số lượng người và vectors
        """
        if not self.has_database:
            return {"num_people": 0, "total_vectors": 0}
        
        total_vectors = sum(t.shape[0] for t in self.database.values())
        num_people = len(self.database)
        
        return {
            "num_people": num_people,
            "total_vectors": total_vectors
        }


# Global face recognition service instance
_face_recognition_service: Optional[FaceRecognitionService] = None


def get_face_recognition_service() -> FaceRecognitionService:
    """
    Lấy singleton instance của FaceRecognitionService
    
    Returns:
        FaceRecognitionService instance
        
    Raises:
        RuntimeError: Nếu service chưa được khởi tạo
    """
    global _face_recognition_service
    
    if _face_recognition_service is None:
        if not settings.RECOGNIZER_CHECKPOINT:
            raise RuntimeError(
                "RECOGNIZER_CHECKPOINT chưa được cấu hình. "
                "Vui lòng set biến môi trường hoặc gọi initialize_face_recognition_service()"
            )
        _face_recognition_service = FaceRecognitionService()
    
    return _face_recognition_service


def initialize_face_recognition_service(
    checkpoint_path: Optional[str] = None,
    **kwargs
) -> FaceRecognitionService:
    """
    Khởi tạo hoặc cập nhật face recognition service
    
    Args:
        checkpoint_path: Đường dẫn checkpoint
        **kwargs: Các tham số khác
        
    Returns:
        FaceRecognitionService instance
    """
    global _face_recognition_service
    _face_recognition_service = FaceRecognitionService(
        checkpoint_path=checkpoint_path,
        **kwargs
    )
    return _face_recognition_service

