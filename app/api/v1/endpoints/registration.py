"""
Face Registration Endpoint - Extract embeddings from registered faces
"""
from fastapi import APIRouter, HTTPException, status, Depends
from typing import List, Dict, Any
from datetime import datetime
import base64
import numpy as np
from PIL import Image
import io

from app.models.schemas import ErrorResponse
from app.services.face_recognition_service import FaceRecognitionService
from app.services.embedding_manager import EmbeddingManager
from app.core.logging import LoggerMixin
from app.core.config import settings
from pydantic import BaseModel, Field


router = APIRouter()


class FaceImageData(BaseModel):
    """Single face image data"""
    image_base64: str = Field(..., description="Base64 encoded cropped face image")
    step_name: str = Field(..., description="Step name (e.g., 'face_front')")
    step_number: int = Field(..., description="Step number (1-14)")


class RegisterFaceRequest(BaseModel):
    """Request to register student face embeddings"""
    student_code: str = Field(..., description="Student code (e.g., 'SV862155')")
    student_id: int = Field(..., description="Student ID from database")
    face_images: List[FaceImageData] = Field(..., description="List of cropped face images (14 images)")
    use_augmentation: bool = Field(default=True, description="Whether to use data augmentation")
    augmentation_count: int = Field(default=5, description="Number of augmented images per original (if enabled)")


class EmbeddingResult(BaseModel):
    """Result for a single embedding"""
    step_name: str
    step_number: int
    embedding: List[float]  # 512-dim vector
    augmented_count: int  # Number of augmentations created


class RegisterFaceResponse(BaseModel):
    """Response after registering face embeddings"""
    success: bool
    student_code: str
    student_id: int
    total_original_images: int
    total_embeddings_created: int
    embeddings: List[EmbeddingResult]
    processing_time_seconds: float
    message: str


class RegistrationService(LoggerMixin):
    """Service for handling face registration"""
    
    def __init__(self):
        super().__init__()
        self.face_recognition_service = FaceRecognitionService()
        self.embedding_manager = EmbeddingManager()
    
    def decode_base64_image(self, image_base64: str) -> np.ndarray:
        """
        Decode base64 string to numpy array (RGB)
        
        Args:
            image_base64: Base64 encoded image string
            
        Returns:
            RGB numpy array
        """
        try:
            # Remove data URL prefix if exists
            if ',' in image_base64:
                image_base64 = image_base64.split(',')[1]
            
            # Decode base64
            image_bytes = base64.b64decode(image_base64)
            
            # Convert to PIL Image then to RGB numpy array
            image = Image.open(io.BytesIO(image_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return np.array(image)
            
        except Exception as e:
            self.logger.error("Failed to decode base64 image", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid image data: {str(e)}"
            )
    
    def process_registration(
        self,
        request: RegisterFaceRequest
    ) -> RegisterFaceResponse:
        """
        Process face registration: extract embeddings from all images
        
        Args:
            request: Registration request with face images
            
        Returns:
            RegisterFaceResponse with all embeddings
        """
        start_time = datetime.now()
        
        self.logger.info(
            "Processing face registration",
            student_code=request.student_code,
            student_id=request.student_id,
            num_images=len(request.face_images),
            use_augmentation=request.use_augmentation,
            aug_count=request.augmentation_count if request.use_augmentation else 0
        )
        
        # Validate number of images
        if len(request.face_images) != 12:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Expected 12 face images, got {len(request.face_images)}"
            )
        
        embeddings_results = []
        total_embeddings_created = 0
        
        # Process each image
        for idx, face_data in enumerate(request.face_images):
            try:
                # Decode image
                face_image = self.decode_base64_image(face_data.image_base64)
                
                # Extract embedding from original image
                original_embedding = self.face_recognition_service.extract_features(
                    face_crop=face_image,
                    tta=settings.TTA_ENABLED  # Use TTA for better quality
                )
                
                # Collect all embeddings (original + augmented)
                all_embeddings = [original_embedding]
                
                # Generate augmented images if enabled
                if request.use_augmentation and request.augmentation_count > 0:
                    augmented_images = self.embedding_manager.generate_augmented_images(
                        image=face_image,
                        count=request.augmentation_count,
                        seed=idx  # Use step number as seed for reproducibility
                    )
                    
                    # Extract embeddings from augmented images
                    for aug_image in augmented_images:
                        aug_embedding = self.face_recognition_service.extract_features(
                            face_crop=aug_image,
                            tta=False  # Don't use TTA for augmented images (faster)
                        )
                        all_embeddings.append(aug_embedding)
                
                # Average all embeddings for this step
                avg_embedding = np.mean(all_embeddings, axis=0)
                
                # Normalize the averaged embedding
                avg_embedding = avg_embedding / (np.linalg.norm(avg_embedding) + 1e-8)
                
                embeddings_results.append(EmbeddingResult(
                    step_name=face_data.step_name,
                    step_number=face_data.step_number,
                    embedding=avg_embedding.tolist(),
                    augmented_count=len(all_embeddings) - 1
                ))
                
                total_embeddings_created += len(all_embeddings)
                
                self.logger.debug(
                    "Processed face image",
                    step=face_data.step_number,
                    step_name=face_data.step_name,
                    original_embeddings=1,
                    augmented_embeddings=len(all_embeddings) - 1,
                    total_for_step=len(all_embeddings)
                )
                
            except Exception as e:
                self.logger.error(
                    "Failed to process face image",
                    step=face_data.step_number,
                    step_name=face_data.step_name,
                    error=str(e)
                )
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to process image {face_data.step_name}: {str(e)}"
                )
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        self.logger.info(
            "Face registration completed",
            student_code=request.student_code,
            total_original=len(request.face_images),
            total_embeddings=total_embeddings_created,
            processing_time=processing_time
        )
        
        return RegisterFaceResponse(
            success=True,
            student_code=request.student_code,
            student_id=request.student_id,
            total_original_images=len(request.face_images),
            total_embeddings_created=total_embeddings_created,
            embeddings=embeddings_results,
            processing_time_seconds=round(processing_time, 2),
            message=f"Successfully registered {len(request.face_images)} face images with {total_embeddings_created} total embeddings"
        )


def get_registration_service() -> RegistrationService:
    return RegistrationService()


@router.post(
    "/register-face",
    response_model=RegisterFaceResponse,
    summary="Register student face embeddings",
    description="Extract embeddings from student face images for attendance recognition"
)
async def register_student_face(
    request: RegisterFaceRequest,
    service: RegistrationService = Depends(get_registration_service)
):
    """
    Register student face by extracting embeddings from 14 face images.
    
    This endpoint:
    1. Receives 14 cropped face images (base64 encoded)
    2. Optionally applies data augmentation to increase robustness
    3. Extracts embeddings using the face recognition model
    4. Returns all embeddings to be stored in the database
    
    **Process:**
    - Each original image → 1 embedding (with TTA)
    - Each augmented image → 1 embedding (without TTA)
    - Final embedding per step = average of all embeddings
    
    **Parameters:**
    - `student_code`: Student code (e.g., 'SV862155')
    - `student_id`: Student database ID
    - `face_images`: List of 14 cropped face images (base64)
    - `use_augmentation`: Enable data augmentation (default: True)
    - `augmentation_count`: Number of augmented versions per image (default: 5)
    
    **Returns:**
    - List of 14 averaged embeddings (512-dim vectors)
    - Processing statistics
    """
    return service.process_registration(request)


@router.get(
    "/health",
    summary="Check registration service health"
)
async def registration_health():
    """Check if registration service is ready"""
    return {
        "status": "healthy",
        "service": "face-registration",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": True
    }
