"""
Face Detection Service - Wrapper cho YOLOFaceDetector
"""
from typing import List, Optional, Tuple
import numpy as np

from models.face_detector import YOLOFaceDetector, Detection
from app.core.config import settings
from app.core.logging import LoggerMixin
from app.services.executor import get_model_executor


class FaceDetectionService(LoggerMixin):
    """
    Service quản lý face detection sử dụng YOLO
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        conf_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None,
        device: Optional[str] = None,
        pad: Optional[int] = None
    ):
        """
        Khởi tạo Face Detection Service
        
        Args:
            checkpoint_path: Đường dẫn đến YOLO checkpoint
            conf_threshold: Ngưỡng confidence cho detection
            nms_threshold: Ngưỡng NMS
            device: Device để chạy model (cuda/cpu)
            pad: Padding cho bounding box
        """
        super().__init__()
        
        # Sử dụng config từ settings nếu không được truyền vào
        self.checkpoint_path = checkpoint_path or settings.DETECTOR_CHECKPOINT
        self.conf_threshold = conf_threshold or settings.DETECTOR_CONF_THRESHOLD
        self.nms_threshold = nms_threshold or settings.DETECTOR_NMS_THRESHOLD
        self.device = device or settings.MODEL_DEVICE
        self.pad = pad or settings.DETECTOR_PAD
        
        if not self.checkpoint_path:
            raise ValueError("DETECTOR_CHECKPOINT phải được cấu hình trong settings hoặc truyền vào constructor")
        
        # Khởi tạo detector
        self.logger.info(
            "Initializing YOLO Face Detector",
            checkpoint=self.checkpoint_path,
            device=self.device,
            conf_threshold=self.conf_threshold
        )
        
        self.detector = YOLOFaceDetector(
            checkpoint_path=self.checkpoint_path,
            conf_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
            device=self.device
        )
        
        self.logger.info("Face Detector initialized successfully")
    
    def detect_faces(
        self,
        image_rgb: np.ndarray,
        return_crops: bool = False,
        pad: Optional[int] = None
    ) -> Tuple[List[Detection], Optional[List[np.ndarray]], np.ndarray]:
        """
        Phát hiện khuôn mặt trong ảnh
        
        Args:
            image_rgb: Ảnh RGB dạng numpy array
            return_crops: Có trả về crops của từng face không
            pad: Padding cho crops (sử dụng self.pad nếu None)
            
        Returns:
            Tuple[detections, crops, original_image]
        """
        try:
            pad_value = pad if pad is not None else self.pad
            
            detections, crops, original_rgb = self.detector.detect(
                image_rgb,
                return_crops=return_crops,
                pad=pad_value
            )
            
            self.logger.debug(
                "Face detection completed",
                num_faces=len(detections),
                return_crops=return_crops
            )
            
            return detections, crops, original_rgb
            
        except Exception as e:
            self.logger.error("Face detection failed", error=str(e))
            raise
    
    async def detect_faces_async(
        self,
        image_rgb: np.ndarray,
        return_crops: bool = False,
        pad: Optional[int] = None
    ) -> Tuple[List[Detection], Optional[List[np.ndarray]], np.ndarray]:
        """
        Async wrapper cho detect_faces - sử dụng ThreadPoolExecutor
        
        Tránh block event loop khi model inference
        """
        executor = get_model_executor()
        return await executor.execute(
            self.detect_faces, 
            image_rgb, 
            return_crops, 
            pad
        )
    
    def get_best_detection(
        self,
        detections: List[Detection],
        min_confidence: Optional[float] = None
    ) -> Optional[Detection]:
        """
        Lấy detection tốt nhất (confidence cao nhất)
        
        Args:
            detections: Danh sách detections
            min_confidence: Ngưỡng confidence tối thiểu
            
        Returns:
            Detection tốt nhất hoặc None
        """
        if not detections:
            return None
        
        best_det = max(detections, key=lambda d: d.confidence)
        
        min_conf = min_confidence or self.conf_threshold
        if best_det.confidence < min_conf:
            self.logger.debug(
                "Best detection below threshold",
                confidence=best_det.confidence,
                threshold=min_conf
            )
            return None
        
        return best_det
    
    def filter_detections(
        self,
        detections: List[Detection],
        min_confidence: Optional[float] = None
    ) -> List[Detection]:
        """
        Lọc detections theo confidence
        
        Args:
            detections: Danh sách detections
            min_confidence: Ngưỡng confidence tối thiểu
            
        Returns:
            Danh sách detections đã lọc
        """
        min_conf = min_confidence or self.conf_threshold
        filtered = [d for d in detections if d.confidence >= min_conf]
        
        self.logger.debug(
            "Filtered detections",
            original_count=len(detections),
            filtered_count=len(filtered),
            threshold=min_conf
        )
        
        return filtered


# Global face detection service instance (khởi tạo khi có config)
_face_detection_service: Optional[FaceDetectionService] = None


def get_face_detection_service() -> FaceDetectionService:
    """
    Lấy singleton instance của FaceDetectionService
    
    Returns:
        FaceDetectionService instance
        
    Raises:
        RuntimeError: Nếu service chưa được khởi tạo
    """
    global _face_detection_service
    
    if _face_detection_service is None:
        if not settings.DETECTOR_CHECKPOINT:
            raise RuntimeError(
                "DETECTOR_CHECKPOINT chưa được cấu hình. "
                "Vui lòng set biến môi trường hoặc gọi initialize_face_detection_service()"
            )
        _face_detection_service = FaceDetectionService()
    
    return _face_detection_service


def initialize_face_detection_service(
    checkpoint_path: Optional[str] = None,
    **kwargs
) -> FaceDetectionService:
    """
    Khởi tạo hoặc cập nhật face detection service
    
    Args:
        checkpoint_path: Đường dẫn checkpoint
        **kwargs: Các tham số khác cho FaceDetectionService
        
    Returns:
        FaceDetectionService instance
    """
    global _face_detection_service
    _face_detection_service = FaceDetectionService(
        checkpoint_path=checkpoint_path,
        **kwargs
    )
    return _face_detection_service

