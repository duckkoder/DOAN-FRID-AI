"""
Embedding Manager - Quản lý embeddings và augmentation
"""
import random
from typing import List, Dict, Optional
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from app.core.logging import LoggerMixin
from app.core.config import settings


class EmbeddingManager(LoggerMixin):
    """
    Service quản lý embeddings: load, save, augment
    """
    
    def __init__(self):
        super().__init__()
        self.logger.info("EmbeddingManager initialized")
    
    def generate_augmented_images(
        self,
        image: np.ndarray,
        count: int,
        seed: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Tạo các ảnh augmented từ ảnh gốc
        
        Args:
            image: Ảnh gốc (RGB numpy array)
            count: Số lượng ảnh augmented cần tạo
            seed: Random seed (optional)
            
        Returns:
            Danh sách ảnh augmented
        """
        if count <= 0:
            return []
        
        pil_image = Image.fromarray(image)
        rng = random.Random(seed)
        augmented: List[np.ndarray] = []
        
        for i in range(count):
            try:
                aug = pil_image.copy()
                
                # Random horizontal flip (30% probability)
                if rng.random() < 0.3:
                    aug = ImageOps.mirror(aug)
                
                # Small random rotation (-12° to +12°)
                angle = rng.uniform(-12.0, 12.0)
                if abs(angle) > 1e-3:
                    aug = aug.rotate(angle, resample=Image.BILINEAR)
                
                # Random translation (up to 4% of image size)
                max_shift_x = aug.width * 0.04
                max_shift_y = aug.height * 0.04
                shift_x = rng.uniform(-max_shift_x, max_shift_x)
                shift_y = rng.uniform(-max_shift_y, max_shift_y)
                
                if abs(shift_x) > 1e-3 or abs(shift_y) > 1e-3:
                    aug = aug.transform(
                        aug.size,
                        Image.AFFINE,
                        (1, 0, shift_x, 0, 1, shift_y),
                        resample=Image.BILINEAR,
                    )
                
                # Color jitter (brightness/contrast/saturation)
                brightness = rng.uniform(0.7, 1.3)
                contrast = rng.uniform(0.8, 1.3)
                saturation = rng.uniform(0.8, 1.2)
                
                aug = ImageEnhance.Brightness(aug).enhance(brightness)
                aug = ImageEnhance.Contrast(aug).enhance(contrast)
                aug = ImageEnhance.Color(aug).enhance(saturation)
                
                # Optional blur (15% probability)
                if rng.random() < 0.15:
                    radius = rng.uniform(0.0, 0.5)
                    aug = aug.filter(ImageFilter.GaussianBlur(radius=radius))
                
                augmented.append(np.array(aug))
                
            except Exception as e:
                self.logger.warning(
                    "Failed to generate augmented image",
                    index=i,
                    error=str(e)
                )
                continue
        
        self.logger.debug(
            "Generated augmented images",
            requested=count,
            generated=len(augmented)
        )
        
        return augmented
    
    def ingest_from_folder(
        self,
        source_dir: Path,
        embedding_root: Path,
        recognizer,
        *,
        tta: bool = False,
        min_images: int = 1,
        augmentations: int = 0,
    ) -> Dict[str, int]:
        """
        Nhập embeddings từ thư mục ảnh
        
        Args:
            source_dir: Thư mục nguồn chứa ảnh (mỗi người một folder)
            embedding_root: Thư mục đích lưu embeddings
            recognizer: FaceRecognitionService instance
            tta: Bật TTA khi extract features
            min_images: Số ảnh tối thiểu cho mỗi người
            augmentations: Số ảnh augmented cho mỗi ảnh gốc
            
        Returns:
            Dictionary {identity: num_embeddings_saved}
        """
        source_dir = source_dir.expanduser().resolve()
        embedding_root.mkdir(parents=True, exist_ok=True)
        
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        stats: Dict[str, int] = {}
        
        self.logger.info(
            "Starting folder ingestion",
            source_dir=str(source_dir),
            embedding_root=str(embedding_root),
            min_images=min_images,
            augmentations=augmentations
        )
        
        # Duyệt qua các thư mục con (mỗi thư mục = 1 người)
        person_dirs = [p for p in source_dir.iterdir() if p.is_dir()]
        
        for person_dir in sorted(person_dirs):
            # Lấy danh sách ảnh
            image_files = [
                p for p in sorted(person_dir.iterdir()) 
                if p.suffix.lower() in extensions
            ]
            
            if len(image_files) < min_images:
                self.logger.debug(
                    "Skipping person - not enough images",
                    person=person_dir.name,
                    num_images=len(image_files),
                    min_required=min_images
                )
                continue
            
            # Sanitize identity name
            identity = recognizer.sanitize_identity(person_dir.name)
            
            self.logger.info(
                "Processing person",
                identity=identity,
                num_images=len(image_files)
            )
            
            # Xử lý từng ảnh
            for image_path in image_files:
                try:
                    # Extract embedding từ ảnh gốc
                    embedding = recognizer.extract_features(image_path, tta=tta).squeeze(0)
                    recognizer.save_embedding(embedding_root, identity, embedding)
                    stats[identity] = stats.get(identity, 0) + 1
                    
                except Exception as e:
                    self.logger.warning(
                        "Failed to process image",
                        image=str(image_path),
                        error=str(e)
                    )
                    continue
                
                # Generate augmented images
                if augmentations > 0:
                    try:
                        with Image.open(image_path) as img:
                            base_np = np.array(img.convert('RGB'))
                    except Exception as e:
                        self.logger.warning(
                            "Failed to load image for augmentation",
                            image=str(image_path),
                            error=str(e)
                        )
                        continue
                    
                    aug_images = self.generate_augmented_images(base_np, augmentations)
                    
                    for aug_idx, aug_img in enumerate(aug_images, start=1):
                        try:
                            aug_embedding = recognizer.extract_features(aug_img, tta=tta).squeeze(0)
                            recognizer.save_embedding(embedding_root, identity, aug_embedding)
                            stats[identity] = stats.get(identity, 0) + 1
                            
                        except Exception as e:
                            self.logger.warning(
                                "Failed to process augmented image",
                                image=str(image_path),
                                aug_index=aug_idx,
                                error=str(e)
                            )
                            continue
        
        if not stats:
            raise ValueError(f"Không tìm thấy ảnh hợp lệ trong {source_dir}")
        
        total_saved = sum(stats.values())
        self.logger.info(
            "Folder ingestion completed",
            num_people=len(stats),
            total_embeddings=total_saved
        )
        
        return stats
    
    def refresh_database(
        self,
        embedding_root: Path,
        recognizer
    ) -> Dict[str, int]:
        """
        Refresh database từ thư mục embeddings
        
        Args:
            embedding_root: Thư mục chứa embeddings
            recognizer: FaceRecognitionService instance
            
        Returns:
            Database stats
        """
        try:
            database = recognizer.load_embedding_directory(embedding_root)
            
            total_vectors = sum(t.shape[0] for t in database.values())
            num_people = len(database)
            
            self.logger.info(
                "Database refreshed",
                embedding_root=str(embedding_root),
                num_people=num_people,
                total_vectors=total_vectors
            )
            
            return {
                "num_people": num_people,
                "total_vectors": total_vectors
            }
            
        except Exception as e:
            self.logger.error(
                "Failed to refresh database",
                embedding_root=str(embedding_root),
                error=str(e)
            )
            raise
    
    def export_embeddings(
        self,
        output_path: Path,
        database: Dict[str, np.ndarray],
        format: str = "npz"
    ) -> None:
        """
        Export embeddings ra file
        
        Args:
            output_path: Đường dẫn file output
            database: Database embeddings
            format: Format file (npz, pickle)
        """
        try:
            if format == "npz":
                np.savez(output_path, **database)
            elif format == "pickle":
                import pickle
                with open(output_path, 'wb') as f:
                    pickle.dump(database, f)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(
                "Embeddings exported",
                output_path=str(output_path),
                format=format,
                num_people=len(database)
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to export embeddings",
                output_path=str(output_path),
                error=str(e)
            )
            raise
    
    def import_embeddings(
        self,
        input_path: Path,
        format: str = "npz"
    ) -> Dict[str, np.ndarray]:
        """
        Import embeddings từ file
        
        Args:
            input_path: Đường dẫn file input
            format: Format file (npz, pickle)
            
        Returns:
            Database embeddings
        """
        try:
            if format == "npz":
                data = np.load(input_path)
                database = {key: data[key] for key in data.keys()}
            elif format == "pickle":
                import pickle
                with open(input_path, 'rb') as f:
                    database = pickle.load(f)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(
                "Embeddings imported",
                input_path=str(input_path),
                format=format,
                num_people=len(database)
            )
            
            return database
            
        except Exception as e:
            self.logger.error(
                "Failed to import embeddings",
                input_path=str(input_path),
                error=str(e)
            )
            raise


# Global embedding manager instance
embedding_manager = EmbeddingManager()

