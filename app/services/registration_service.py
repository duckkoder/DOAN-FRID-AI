"""
Registration Service - Đăng ký người dùng mới
"""
from typing import Optional, Dict, List
from pathlib import Path
import numpy as np
import cv2

from models.face_detector import Detection
from app.core.logging import LoggerMixin
from app.core.config import settings


class RegistrationService(LoggerMixin):
    """
    Service quản lý đăng ký người dùng mới
    """
    
    def __init__(
        self,
        detector,
        recognizer,
        embedding_manager,
        *,
        min_confidence: Optional[float] = None,
        save_images: bool = True,
        augmentations: Optional[int] = None
    ):
        """
        Khởi tạo Registration Service
        
        Args:
            detector: FaceDetectionService instance
            recognizer: FaceRecognitionService instance
            embedding_manager: EmbeddingManager instance
            min_confidence: Ngưỡng confidence tối thiểu
            save_images: Có lưu ảnh crops không
            augmentations: Số lượng ảnh augmented cho mỗi sample
        """
        super().__init__()
        
        self.detector = detector
        self.recognizer = recognizer
        self.embedding_manager = embedding_manager
        
        self.min_confidence = min_confidence or settings.REGISTRATION_MIN_CONFIDENCE
        self.save_images = save_images
        self.augmentations = augmentations or settings.REGISTRATION_AUGMENTATIONS
        
        self.logger.info(
            "RegistrationService initialized",
            min_confidence=self.min_confidence,
            augmentations=self.augmentations
        )
    
    def register_from_image(
        self,
        image_rgb: np.ndarray,
        person_name: str,
        save_dir: Path,
        *,
        min_confidence: Optional[float] = None,
        save_image: bool = True,
        augmentations: Optional[int] = None,
    ) -> Dict[str, any]:
        """
        Đăng ký người dùng từ một ảnh
        
        Args:
            image_rgb: Ảnh RGB (numpy array)
            person_name: Tên người cần đăng ký
            save_dir: Thư mục lưu embeddings
            min_confidence: Ngưỡng confidence (override default)
            save_image: Có lưu ảnh crop không
            augmentations: Số ảnh augmented (override default)
            
        Returns:
            Dictionary chứa kết quả:
            - success: bool
            - message: str
            - embeddings_saved: int
            - identity: str (sanitized name)
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        
        min_conf = min_confidence or self.min_confidence
        num_aug = augmentations if augmentations is not None else self.augmentations
        
        # Sanitize identity
        clean_name = self.recognizer.sanitize_identity(person_name)
        
        self.logger.info(
            "Registering user from image",
            person_name=person_name,
            clean_name=clean_name,
            save_dir=str(save_dir)
        )
        
        # Detect faces
        detections, crops, _ = self.detector.detect_faces(
            image_rgb,
            return_crops=True
        )
        
        if not detections:
            self.logger.warning("No faces detected")
            return {
                "success": False,
                "message": "Không phát hiện khuôn mặt trong ảnh",
                "embeddings_saved": 0,
                "identity": clean_name
            }
        
        # Get best detection
        best_det = self.detector.get_best_detection(detections, min_conf)
        
        if best_det is None:
            self.logger.warning(
                "Best detection below threshold",
                max_confidence=max(d.confidence for d in detections),
                threshold=min_conf
            )
            return {
                "success": False,
                "message": f"Độ tự tin phát hiện khuôn mặt thấp hơn ngưỡng {min_conf:.2f}",
                "embeddings_saved": 0,
                "identity": clean_name
            }
        
        # Get crop of best face
        best_idx = detections.index(best_det)
        crop = crops[best_idx]
        
        if crop is None:
            return {
                "success": False,
                "message": "Không thể crop khuôn mặt",
                "embeddings_saved": 0,
                "identity": clean_name
            }
        
        embeddings_saved = 0
        
        try:
            # Extract and save embedding from original image
            embedding = self.recognizer.extract_features(crop, tta=settings.TTA_ENABLED).squeeze(0)
            emb_path = self.recognizer.save_embedding(save_dir, clean_name, embedding)
            embeddings_saved += 1
            
            self.logger.debug(
                "Original embedding saved",
                identity=clean_name,
                path=str(emb_path)
            )
            
            # Save image if requested
            if save_image:
                image_path = emb_path.with_suffix('.jpg')
                crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(image_path), crop_bgr)
            
            # Generate and save augmented embeddings
            if num_aug > 0:
                aug_images = self.embedding_manager.generate_augmented_images(crop, num_aug)
                
                for aug_idx, aug_image in enumerate(aug_images, start=1):
                    try:
                        aug_embedding = self.recognizer.extract_features(
                            aug_image,
                            tta=settings.TTA_ENABLED
                        ).squeeze(0)
                        
                        aug_path = self.recognizer.save_embedding(save_dir, clean_name, aug_embedding)
                        embeddings_saved += 1
                        
                        if save_image:
                            aug_image_path = aug_path.with_suffix('.jpg')
                            aug_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(str(aug_image_path), aug_bgr)
                        
                        self.logger.debug(
                            "Augmented embedding saved",
                            identity=clean_name,
                            aug_index=aug_idx
                        )
                        
                    except Exception as e:
                        self.logger.warning(
                            "Failed to save augmented embedding",
                            aug_index=aug_idx,
                            error=str(e)
                        )
                        continue
            
            self.logger.info(
                "User registration completed",
                identity=clean_name,
                embeddings_saved=embeddings_saved
            )
            
            return {
                "success": True,
                "message": f"Đã đăng ký thành công {embeddings_saved} embeddings",
                "embeddings_saved": embeddings_saved,
                "identity": clean_name,
                "detection_confidence": float(best_det.confidence)
            }
            
        except Exception as e:
            self.logger.error(
                "Registration failed",
                identity=clean_name,
                error=str(e)
            )
            return {
                "success": False,
                "message": f"Lỗi khi lưu embeddings: {str(e)}",
                "embeddings_saved": embeddings_saved,
                "identity": clean_name
            }
    
    def register_from_video_stream(
        self,
        camera_id: int,
        person_name: str,
        save_dir: Path,
        *,
        max_samples: Optional[int] = None,
        min_confidence: Optional[float] = None,
        save_images: bool = True,
        augmentations: Optional[int] = None,
        callback = None
    ) -> Dict[str, any]:
        """
        Đăng ký người dùng từ video stream (interactive mode)
        
        Args:
            camera_id: ID camera
            person_name: Tên người cần đăng ký
            save_dir: Thư mục lưu embeddings
            max_samples: Số sample tối đa (None = unlimited)
            min_confidence: Ngưỡng confidence
            save_images: Có lưu ảnh crops không
            augmentations: Số ảnh augmented
            callback: Callback function được gọi mỗi khi capture frame
                      callback(frame_bgr, detections, saved_count) -> should_continue
            
        Returns:
            Dictionary chứa kết quả đăng ký
        """
        save_dir.mkdir(parents=True, exist_ok=True)
        
        min_conf = min_confidence or self.min_confidence
        num_aug = augmentations if augmentations is not None else self.augmentations
        clean_name = self.recognizer.sanitize_identity(person_name)
        
        self.logger.info(
            "Starting video stream registration",
            person_name=person_name,
            clean_name=clean_name,
            camera_id=camera_id,
            max_samples=max_samples
        )
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            return {
                "success": False,
                "message": f"Không mở được camera {camera_id}",
                "embeddings_saved": 0,
                "identity": clean_name
            }
        
        saved = 0
        
        try:
            while True:
                ret, frame_bgr = cap.read()
                if not ret:
                    self.logger.warning("Cannot read frame from camera")
                    break
                
                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                detections, crops, _ = self.detector.detect_faces(
                    frame_rgb,
                    return_crops=True
                )
                
                # Call callback if provided
                if callback is not None:
                    should_continue = callback(frame_bgr, detections, saved)
                    if not should_continue:
                        break
                
                # Check max samples
                if max_samples is not None and saved >= max_samples:
                    self.logger.info("Max samples reached", saved=saved)
                    break
            
            return {
                "success": saved > 0,
                "message": f"Đã lưu {saved} embeddings cho {clean_name}",
                "embeddings_saved": saved,
                "identity": clean_name
            }
            
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def register_batch_from_folder(
        self,
        source_dir: Path,
        save_dir: Path,
        *,
        min_images: int = 1,
        augmentations: Optional[int] = None
    ) -> Dict[str, int]:
        """
        Đăng ký hàng loạt người dùng từ thư mục
        
        Args:
            source_dir: Thư mục nguồn (mỗi người một folder)
            save_dir: Thư mục đích lưu embeddings
            min_images: Số ảnh tối thiểu cho mỗi người
            augmentations: Số ảnh augmented
            
        Returns:
            Dictionary {identity: num_embeddings}
        """
        num_aug = augmentations if augmentations is not None else self.augmentations
        
        self.logger.info(
            "Starting batch registration",
            source_dir=str(source_dir),
            save_dir=str(save_dir),
            min_images=min_images,
            augmentations=num_aug
        )
        
        stats = self.embedding_manager.ingest_from_folder(
            source_dir=source_dir,
            embedding_root=save_dir,
            recognizer=self.recognizer,
            tta=settings.TTA_ENABLED,
            min_images=min_images,
            augmentations=num_aug
        )
        
        # Refresh database
        self.embedding_manager.refresh_database(save_dir, self.recognizer)
        
        return stats


# Factory function
def create_registration_service(detector, recognizer, embedding_manager) -> RegistrationService:
    """
    Factory function tạo RegistrationService
    
    Args:
        detector: FaceDetectionService instance
        recognizer: FaceRecognitionService instance
        embedding_manager: EmbeddingManager instance
        
    Returns:
        RegistrationService instance
    """
    return RegistrationService(
        detector=detector,
        recognizer=recognizer,
        embedding_manager=embedding_manager
    )

