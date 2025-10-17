"""
Registration endpoints - Đăng ký người dùng mới
"""
import base64
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any
import numpy as np
import cv2

from fastapi import APIRouter, HTTPException, status, Depends

from app.models.schemas import (
    RegistrationRequest,
    RegistrationResponse,
    BatchRegistrationRequest,
    BatchRegistrationResponse,
    EmbeddingStats,
    RefreshDatabaseRequest,
    RefreshDatabaseResponse
)
from app.core.security import get_current_user
from app.core.config import settings
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/register", response_model=RegistrationResponse)
async def register_user(
    request: RegistrationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Đăng ký người dùng mới từ ảnh
    
    Args:
        request: Thông tin đăng ký (tên, ảnh base64)
        current_user: Thông tin user từ JWT
        
    Returns:
        Kết quả đăng ký
    """
    request_logger = logger.bind(
        person_name=request.person_name,
        user_id=current_user.get("sub")
    )
    
    try:
        # Import services
        from app.services.face_detection_service import get_face_detection_service
        from app.services.face_recognition_service import get_face_recognition_service
        from app.services.embedding_manager import embedding_manager
        from app.services.registration_service import create_registration_service
        
        # Get services
        detector = get_face_detection_service()
        recognizer = get_face_recognition_service()
        registration_service = create_registration_service(
            detector, recognizer, embedding_manager
        )
        
        # Decode image
        try:
            image_data = base64.b64decode(request.image_base64)
            image_array = np.frombuffer(image_data, dtype=np.uint8)
            image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if image_bgr is None:
                raise ValueError("Không thể decode ảnh")
            
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            request_logger.warning("Failed to decode image", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Ảnh không hợp lệ: {str(e)}"
            )
        
        # Get embedding directory from config
        if not settings.EMBEDDING_DIR:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="EMBEDDING_DIR chưa được cấu hình"
            )
        
        embedding_dir = Path(settings.EMBEDDING_DIR)
        
        request_logger.info("Starting user registration")
        
        # Register user
        result = registration_service.register_from_image(
            image_rgb=image_rgb,
            person_name=request.person_name,
            save_dir=embedding_dir,
            min_confidence=request.min_confidence,
            save_image=request.save_image or True,
            augmentations=request.augmentations
        )
        
        if result["success"]:
            # Refresh database
            embedding_manager.refresh_database(embedding_dir, recognizer)
            request_logger.info(
                "User registration successful",
                identity=result["identity"],
                embeddings_saved=result["embeddings_saved"]
            )
        else:
            request_logger.warning(
                "User registration failed",
                message=result["message"]
            )
        
        return RegistrationResponse(
            success=result["success"],
            message=result["message"],
            identity=result["identity"],
            embeddings_saved=result["embeddings_saved"],
            detection_confidence=result.get("detection_confidence"),
            timestamp=datetime.now(timezone.utc)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        request_logger.error("Registration error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi đăng ký: {str(e)}"
        )


@router.post("/register/batch", response_model=BatchRegistrationResponse)
async def register_batch(
    request: BatchRegistrationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Đăng ký hàng loạt người dùng từ thư mục
    
    Args:
        request: Thông tin thư mục nguồn
        current_user: Thông tin user từ JWT
        
    Returns:
        Kết quả đăng ký hàng loạt
    """
    request_logger = logger.bind(
        source_dir=request.source_dir,
        user_id=current_user.get("sub")
    )
    
    try:
        # Import services
        from app.services.face_detection_service import get_face_detection_service
        from app.services.face_recognition_service import get_face_recognition_service
        from app.services.embedding_manager import embedding_manager
        from app.services.registration_service import create_registration_service
        
        # Get services
        detector = get_face_detection_service()
        recognizer = get_face_recognition_service()
        registration_service = create_registration_service(
            detector, recognizer, embedding_manager
        )
        
        # Get embedding directory
        if not settings.EMBEDDING_DIR:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="EMBEDDING_DIR chưa được cấu hình"
            )
        
        embedding_dir = Path(settings.EMBEDDING_DIR)
        source_path = Path(request.source_dir)
        
        if not source_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Thư mục không tồn tại: {request.source_dir}"
            )
        
        request_logger.info("Starting batch registration")
        
        # Register batch
        stats = registration_service.register_batch_from_folder(
            source_dir=source_path,
            save_dir=embedding_dir,
            min_images=request.min_images or 1,
            augmentations=request.augmentations
        )
        
        total_people = len(stats)
        total_embeddings = sum(stats.values())
        
        request_logger.info(
            "Batch registration completed",
            total_people=total_people,
            total_embeddings=total_embeddings
        )
        
        return BatchRegistrationResponse(
            success=True,
            message=f"Đã đăng ký {total_people} người với {total_embeddings} embeddings",
            stats=stats,
            total_people=total_people,
            total_embeddings=total_embeddings,
            timestamp=datetime.now(timezone.utc)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        request_logger.error("Batch registration error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi đăng ký hàng loạt: {str(e)}"
        )


@router.get("/embeddings/stats", response_model=EmbeddingStats)
async def get_embedding_stats(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Lấy thống kê embeddings database
    
    Args:
        current_user: Thông tin user từ JWT
        
    Returns:
        Thống kê database
    """
    try:
        from app.services.face_engine import get_face_engine
        
        face_engine = get_face_engine()
        stats = face_engine.get_database_stats()
        
        return EmbeddingStats(
            num_people=stats["num_people"],
            total_vectors=stats["total_vectors"],
            timestamp=datetime.now(timezone.utc)
        )
        
    except Exception as e:
        logger.error("Failed to get embedding stats", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi lấy thống kê: {str(e)}"
        )


@router.post("/embeddings/refresh", response_model=RefreshDatabaseResponse)
async def refresh_database(
    request: RefreshDatabaseRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Refresh embeddings database từ thư mục
    
    Args:
        request: Đường dẫn thư mục (optional, dùng config nếu không có)
        current_user: Thông tin user từ JWT
        
    Returns:
        Kết quả refresh
    """
    try:
        from app.services.face_recognition_service import get_face_recognition_service
        from app.services.embedding_manager import embedding_manager
        
        recognizer = get_face_recognition_service()
        
        # Get embedding directory
        embedding_dir_str = request.embedding_dir or settings.EMBEDDING_DIR
        
        if not embedding_dir_str:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Embedding directory không được cung cấp"
            )
        
        embedding_dir = Path(embedding_dir_str)
        
        if not embedding_dir.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Thư mục không tồn tại: {embedding_dir}"
            )
        
        logger.info("Refreshing database", embedding_dir=str(embedding_dir))
        
        # Refresh database
        stats = embedding_manager.refresh_database(embedding_dir, recognizer)
        
        logger.info(
            "Database refreshed",
            num_people=stats["num_people"],
            total_vectors=stats["total_vectors"]
        )
        
        return RefreshDatabaseResponse(
            success=True,
            message=f"Đã refresh database với {stats['num_people']} người",
            num_people=stats["num_people"],
            total_vectors=stats["total_vectors"],
            timestamp=datetime.now(timezone.utc)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to refresh database", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Lỗi refresh database: {str(e)}"
        )

