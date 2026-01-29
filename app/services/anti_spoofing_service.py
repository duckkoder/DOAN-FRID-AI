"""
Anti-Spoofing Service - Phát hiện tấn công giả mạo (real/spoof)
Model mới: ResNet18_MSFF_AntiSpoof với 2 classes
"""
from typing import Tuple, Optional, Dict, Any
import numpy as np
from PIL import Image

from app.core.logging import LoggerMixin
from app.core.config import settings
from models.AntiSpoofing import AntiSpoofingClassifier


class AntiSpoofingService(LoggerMixin):
    """
    Service phát hiện tấn công giả mạo khuôn mặt
    Model mới: ResNet18_MSFF_AntiSpoof
    - real: Khuôn mặt thật (class 0)
    - spoof: Khuôn mặt giả (class 1) - bao gồm print, replay, mask, etc.
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        threshold: Optional[float] = None
    ):
        """
        Khởi tạo Anti-Spoofing Service
        
        Args:
            checkpoint_path: Đường dẫn checkpoint model
            device: Device để chạy model ('cuda' hoặc 'cpu')
            threshold: Ngưỡng confidence tối thiểu
        """
        super().__init__()
        
        self.checkpoint_path = checkpoint_path or settings.ANTISPOOFING_CHECKPOINT
        self.device = device or settings.ANTISPOOFING_DEVICE
        self.threshold = threshold or settings.ANTISPOOFING_THRESHOLD
        
        self.logger.info(
            "Initializing Anti-Spoofing Service",
            checkpoint=self.checkpoint_path,
            device=self.device,
            threshold=self.threshold
        )
        
        try:
            self.classifier = AntiSpoofingClassifier(
                checkpoint_path=self.checkpoint_path,
                device=self.device
            )
            self.logger.info("Anti-Spoofing model loaded successfully")
        except Exception as e:
            self.logger.error("Failed to load Anti-Spoofing model", error=str(e))
            raise
    
    def predict(self, face_image: np.ndarray) -> Dict[str, Any]:
        """
        Dự đoán loại khuôn mặt với format mới
        
        Args:
            face_image: Ảnh khuôn mặt (numpy array RGB)
            
        Returns:
            Dict với keys:
                - is_live: bool (True nếu real, False nếu spoof)
                - label: str ('real' hoặc 'spoof')
                - confidence: float (0.0 - 1.0)
        """
        try:
            if face_image is None or face_image.size == 0:
                self.logger.warning("Empty face image provided")
                return {
                    'is_live': False,
                    'label': 'unknown',
                    'confidence': 0.0
                }
            
            # Predict using classifier
            label, confidence = self.classifier.predict(face_image)
            
            # ✅ Determine is_live based on label
            is_live = (label == 'real')
            
            self.logger.debug(
                "Anti-spoofing prediction",
                is_live=is_live,
                label=label,
                confidence=f"{confidence:.3f}"
            )
            
            return {
                'is_live': is_live,
                'label': label,
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.error("Anti-spoofing prediction failed", error=str(e), exc_info=True)
            return {
                'is_live': False,
                'label': 'unknown',
                'confidence': 0.0
            }
    
    def is_live(self, face_image: np.ndarray) -> Tuple[bool, str, float]:
        """
        Kiểm tra xem khuôn mặt có phải là real không
        
        Args:
            face_image: Ảnh khuôn mặt (numpy array RGB)
            
        Returns:
            Tuple (is_live, label, confidence):
                - is_live: True nếu là real và confidence >= threshold
                - label: 'real' hoặc 'spoof'
                - confidence: Độ tin cậy
        """
        result = self.predict(face_image)
        
        label = result['label']
        confidence = result['confidence']
        is_live = result['is_live'] and (confidence >= self.threshold)
        
        if not is_live:
            self.logger.warning(
                "⚠️ Suspicious face detected",
                label=label,
                confidence=f"{confidence:.3f}",
                threshold=self.threshold
            )
        
        return is_live, label, confidence
    
    async def predict_async(self, face_image: np.ndarray) -> Dict[str, Any]:
        """
        Async version of predict - sử dụng ThreadPoolExecutor
        
        Tránh block event loop khi model inference, cho phép xử lý
        song song nhiều faces trong batch processing.
        """
        from app.services.executor import get_model_executor
        
        executor = get_model_executor()
        return await executor.execute(self.predict, face_image)
    
    async def is_live_async(self, face_image: np.ndarray) -> Tuple[bool, str, float]:
        """
        Async version of is_live - sử dụng ThreadPoolExecutor
        """
        from app.services.executor import get_model_executor
        
        executor = get_model_executor()
        return await executor.execute(self.is_live, face_image)


# ============================================================
# ✅ THÊM GLOBAL INSTANCE PATTERN GIỐNG face_detection_service
# ============================================================

# Global service instance
_anti_spoofing_service: Optional[AntiSpoofingService] = None


def initialize_anti_spoofing_service(
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None,
    threshold: Optional[float] = None
) -> AntiSpoofingService:
    """
    Khởi tạo global anti-spoofing service instance
    
    Args:
        checkpoint_path: Đường dẫn checkpoint model
        device: Device ('cuda' hoặc 'cpu')
        threshold: Ngưỡng confidence
        
    Returns:
        AntiSpoofingService instance
        
    Raises:
        Exception: Nếu khởi tạo thất bại
    """
    global _anti_spoofing_service
    
    if _anti_spoofing_service is not None:
        # Đã khởi tạo rồi, trả về instance hiện tại
        return _anti_spoofing_service
    
    _anti_spoofing_service = AntiSpoofingService(
        checkpoint_path=checkpoint_path,
        device=device,
        threshold=threshold
    )
    return _anti_spoofing_service


def get_anti_spoofing_service() -> AntiSpoofingService:
    """
    Lấy global Anti-Spoofing service instance
    
    Returns:
        AntiSpoofingService instance
        
    Raises:
        RuntimeError: Nếu service chưa được khởi tạo
    """
    global _anti_spoofing_service
    
    if _anti_spoofing_service is None:
        raise RuntimeError(
            "Anti-spoofing service not initialized. "
            "Call initialize_anti_spoofing_service() first."
        )
    
    return _anti_spoofing_service