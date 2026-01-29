"""
Face Recognition Service - Wrapper cho FaceRecognizer
"""
from typing import Dict, Optional, List, Any
import numpy as np
from pathlib import Path
import torch

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
        knn_voting_threshold: Optional[float] = None,
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
            knn_voting_threshold: Threshold cho KNN voting (chỉ vote neighbors < threshold này)
            enable_dynamic_threshold: Bật dynamic threshold
            per_identity_quantile: Quantile cho per-identity threshold
            per_identity_margin: Margin cho per-identity threshold
            identity_threshold_min_scale: Min scale cho identity threshold
            identity_threshold_max_scale: Max scale cho identity threshold
        """
        super().__init__()
        
        # Load config từ settings
        self.checkpoint_path = checkpoint_path or settings.RECOGNIZER_CHECKPOINT
        self.device = device or settings.MODEL_DEVICE
        self.threshold = threshold or settings.RECOGNIZER_THRESHOLD
        self.knn_k = knn_k or settings.RECOGNIZER_KNN_K
        self.knn_voting_threshold = knn_voting_threshold or settings.RECOGNIZER_KNN_VOTING_THRESHOLD
        
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
            knn_voting_threshold=self.knn_voting_threshold,
            enable_dynamic_threshold=enable_dynamic_threshold or settings.REC_ENABLE_DYNAMIC_THRESHOLD,
            per_identity_quantile=per_identity_quantile or settings.REC_IDENTITY_QUANTILE,
            per_identity_margin=per_identity_margin or settings.REC_IDENTITY_MARGIN,
            identity_threshold_min_scale=identity_threshold_min_scale or settings.REC_IDENTITY_MIN_SCALE,
            confidence_distance_weight=settings.REC_CONFIDENCE_DISTANCE_WEIGHT,
            confidence_vote_weight=settings.REC_CONFIDENCE_VOTE_WEIGHT,
        )
        
        self.logger.info("Face Recognizer initialized successfully")
    
    def _should_accept_recognition(
        self,
        identity: Dict[str, Any],
        min_confidence: float,
        min_vote_ratio: float,
        min_valid_neighbors_ratio: float,
        max_distance_ratio: float,
    ) -> tuple[bool, str]:
        """
        Kiểm tra xem kết quả recognition có nên chấp nhận không.
        Giúp tránh false positive (nhận nhầm người lạ thành người trong DB).
        
        Args:
            identity: Kết quả từ recognizer.identify()
            min_confidence: Confidence tối thiểu
            min_vote_ratio: Vote ratio tối thiểu
            min_valid_neighbors_ratio: Tỷ lệ neighbors tham gia vote tối thiểu
            max_distance_ratio: Distance ratio tối đa (distance/threshold)
            
        Returns:
            (should_accept, reason)
        """
        person = identity.get('person', 'Unknown')
        
        # Đã là Unknown thì không cần check
        if person == 'Unknown':
            return True, 'already_unknown'
        
        confidence = float(identity.get('confidence', 0.0))
        vote_ratio = float(identity.get('vote_ratio', 0.0))
        distance = float(identity.get('distance', float('inf')))
        threshold = float(identity.get('threshold', 1.0))
        valid_neighbors = int(identity.get('valid_neighbors_count', 0))
        total_neighbors = int(identity.get('total_neighbors', 5))
        
        # Check 1: Confidence quá thấp
        if confidence < min_confidence:
            return False, f'low_confidence ({confidence:.3f} < {min_confidence})'
        
        # Check 2: Vote ratio quá thấp (KNN không đủ đồng thuận)
        if vote_ratio < min_vote_ratio:
            return False, f'low_vote_ratio ({vote_ratio:.3f} < {min_vote_ratio})'
        
        # Check 3: ✅ NEW - Quá ít neighbors tham gia vote (thiếu thông tin)
        # Ví dụ: 2/5 neighbors = 40% < 60% → reject
        valid_neighbors_ratio = valid_neighbors / total_neighbors if total_neighbors > 0 else 0
        if valid_neighbors_ratio < min_valid_neighbors_ratio:
            return False, f'too_few_valid_neighbors ({valid_neighbors}/{total_neighbors} = {valid_neighbors_ratio:.1%} < {min_valid_neighbors_ratio:.0%})'
        
        # Check 4: Distance quá gần threshold (không đủ "chắc chắn")
        # Ví dụ: distance=1.2, threshold=1.3 → ratio=0.92 > 0.85 → reject
        distance_ratio = distance / threshold if threshold > 0 else 0
        if distance_ratio >= max_distance_ratio:
            return False, f'distance_too_close_to_threshold ({distance_ratio:.3f} >= {max_distance_ratio})'
        
        return True, 'accepted'
    
    def identify(
        self,
        face_crop: np.ndarray,
        tta: Optional[bool] = None,
        gallery_embeddings: Optional[Any] = None,  # torch.Tensor
        gallery_labels: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Nhận diện khuôn mặt với filtering để tránh false positive
        
        Args:
            face_crop: Ảnh crop của khuôn mặt (RGB)
            tta: Bật Test Time Augmentation
            gallery_embeddings: External gallery embeddings tensor (N, 512) on GPU [OPTIONAL]
            gallery_labels: External gallery labels (student codes) [OPTIONAL]
            
        Returns:
            Dictionary chứa thông tin identification:
            - person: Tên người (hoặc "Unknown")
            - confidence: Độ tin cậy
            - distance: Khoảng cách embedding
            - vote_ratio: Tỷ lệ vote (nếu dùng KNN)
            - threshold: Ngưỡng được sử dụng
            - rejection_reason: Lý do reject (nếu bị reject)
            
        Note:
            If gallery_embeddings and gallery_labels are provided, they will be used
            instead of the internal database. This enables session-based recognition.
        """
        try:
            tta_enabled = tta if tta is not None else settings.TTA_ENABLED
            
            # Gọi recognizer để nhận diện
            identity = self.recognizer.identify(
                face_crop, 
                tta=tta_enabled,
                gallery_embeddings=gallery_embeddings,
                gallery_labels=gallery_labels
            )
            
            if not identity:
                return None
            
            # ===== THÊM FILTERING ĐỂ TRÁNH FALSE POSITIVE =====
            should_accept, reason = self._should_accept_recognition(
                identity,
                min_confidence=settings.REC_MIN_CONFIDENCE,
                min_vote_ratio=settings.REC_MIN_VOTE_RATIO,
                min_valid_neighbors_ratio=settings.REC_MIN_VALID_NEIGHBORS_RATIO,
                max_distance_ratio=settings.REC_MAX_DISTANCE_RATIO,
            )
            
            if not should_accept:
                # Log lý do reject để debug
                neighbor_stats = identity.get('neighbor_distance_stats', {})
                voted_stats = identity.get('voted_distance_stats', {})
                
                self.logger.info(
                    "❌ Recognition REJECTED",
                    best_candidate=identity.get('best_candidate'),
                    reason=reason,
                    confidence_calibrated=f"{identity.get('confidence', 0):.3f}",
                    confidence_legacy=f"{identity.get('confidence_legacy', 0):.3f}",
                    vote_ratio=f"{identity.get('vote_ratio', 0):.3f}",
                    valid_neighbors=f"{identity.get('valid_neighbors_count', 0)}/{identity.get('total_neighbors', 0)}",
                    distance=f"{identity.get('distance', 0):.3f}",
                    threshold=f"{identity.get('threshold', 0):.3f}",
                    voting_threshold=f"{identity.get('knn_voting_threshold', 0):.3f}",
                    nearest_distance=f"{neighbor_stats.get('nearest_distance', 0):.3f}",
                    neighbor_distance_range=f"[{neighbor_stats.get('min_distance', 0):.3f}, {neighbor_stats.get('max_distance', 0):.3f}]",
                    mean_neighbor_distance=f"{neighbor_stats.get('mean_distance', 0):.3f}",
                    voted_distance_range=f"[{voted_stats.get('min_voted_distance', 0) or 'N/A'}, {voted_stats.get('max_voted_distance', 0) or 'N/A'}]",
                )
                
                # Ghi đè thành Unknown nhưng giữ best_candidate để debug
                identity['person'] = 'Unknown'
                identity['rejection_reason'] = reason
            else:
                # Log khi ACCEPT (để debug)
                neighbor_stats = identity.get('neighbor_distance_stats', {})
                voted_stats = identity.get('voted_distance_stats', {})
                
                self.logger.info(
                    "✅ Recognition ACCEPTED",
                    person=identity.get('person'),
                    confidence_calibrated=f"{identity.get('confidence', 0):.3f}",
                    confidence_legacy=f"{identity.get('confidence_legacy', 0):.3f}",
                    vote_ratio=f"{identity.get('vote_ratio', 0):.3f}",
                    valid_neighbors=f"{identity.get('valid_neighbors_count', 0)}/{identity.get('total_neighbors', 0)}",
                    distance=f"{identity.get('distance', 0):.3f}",
                    threshold=f"{identity.get('threshold', 0):.3f}",
                    voting_threshold=f"{identity.get('knn_voting_threshold', 0):.3f}",
                    nearest_distance=f"{neighbor_stats.get('nearest_distance', 0):.3f}",
                    neighbor_distance_range=f"[{neighbor_stats.get('min_distance', 0):.3f}, {neighbor_stats.get('max_distance', 0):.3f}]",
                    mean_neighbor_distance=f"{neighbor_stats.get('mean_distance', 0):.3f}",
                    voted_distance_range=f"[{voted_stats.get('min_voted_distance', 0):.3f}, {voted_stats.get('max_voted_distance', 0):.3f}]" if voted_stats.get('min_voted_distance') else "N/A",
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
        tta: Optional[bool] = None,
        gallery_embeddings: Optional[torch.Tensor] = None,
        gallery_labels: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Async wrapper cho identify - sử dụng ThreadPoolExecutor
        
        Tránh block event loop khi model inference, cho phép xử lý
        song song nhiều faces trong batch processing.
        """
        from app.services.executor import get_model_executor
        
        executor = get_model_executor()
        return await executor.execute(
            self.identify,
            face_crop,
            tta,
            gallery_embeddings,
            gallery_labels
        )
    
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

