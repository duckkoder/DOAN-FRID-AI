"""
Cấu hình ứng dụng sử dụng pydantic-settings
"""
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Cấu hình chính của ứng dụng"""
    
    # App settings
    APP_NAME: str = "AI-Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # JWT settings
    JWT_ALGORITHM: str = "HS256"
    JWT_SECRET: Optional[str] = None
    JWT_PUBLIC_KEY_BASE64: Optional[str] = None
    JWT_REQUIRED_CLAIMS: List[str] = ["exp", "iat", "aud", "iss"]
    JWT_AUDIENCE: Optional[str] = None
    JWT_ISSUER: Optional[str] = None
    
    # Callback settings
    CALLBACK_MAX_RETRIES: int = 3
    CALLBACK_RETRY_DELAY: float = 1.0
    
    # S3/MinIO settings (stub)
    S3_ENDPOINT: Optional[str] = None
    S3_ACCESS_KEY: Optional[str] = None
    S3_SECRET_KEY: Optional[str] = None
    S3_REGION: str = "us-east-1"
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Face Detection settings
    DETECTOR_CHECKPOINT: Optional[str] = None
    DETECTOR_CONF_THRESHOLD: float = 0.75
    DETECTOR_NMS_THRESHOLD: float = 0.4
    DETECTOR_DEVICE: str = "cuda"
    DETECTOR_PAD: int = 10
    
    # Face Recognition settings
    RECOGNIZER_CHECKPOINT: Optional[str] = None
    RECOGNIZER_DEVICE: str = "cuda"
    RECOGNIZER_THRESHOLD: float = 1.5
    RECOGNIZER_KNN_K: int = 5
    RECOGNIZER_MIN_CONFIDENCE: float = 0.5
    RECOGNIZER_MIN_VOTE_RATIO: float = 0.5
    RECOGNIZER_REQUIRE_STABLE: bool = False
    
    # Dynamic threshold settings
    REC_ENABLE_DYNAMIC_THRESHOLD: bool = True
    REC_IDENTITY_QUANTILE: float = 0.9
    REC_IDENTITY_MARGIN: float = 0.15
    REC_IDENTITY_MIN_SCALE: float = 0.7
    REC_IDENTITY_MAX_SCALE: float = 2.01
    
    # Quality filtering
    ENABLE_QUALITY_FILTER: bool = False
    QUALITY_THRESHOLD: float = 0.35
    
    # Temporal smoothing
    ENABLE_TEMPORAL_SMOOTHING: bool = True
    TEMPORAL_WINDOW_SIZE: int = 5
    
    # TTA (Test Time Augmentation)
    TTA_ENABLED: bool = False
    TTA_MODE: str = "basic"  # "basic" or "advanced"
    
    # Recognition Validation Settings (Anti-premature detection)
    # Số lần nhận diện tối thiểu để xác nhận (VD: 3 lần trong 5 frame)
    RECOGNITION_CONFIRMATION_THRESHOLD: int = 3
    # Số frame gần nhất để xem xét
    RECOGNITION_WINDOW_SIZE: int = 5
    # Confidence score trung bình tối thiểu để xác nhận
    RECOGNITION_MIN_AVG_CONFIDENCE: float = 0.55
    # Tỷ lệ nhận diện thành công tối thiểu (3/5 = 0.6)
    RECOGNITION_MIN_SUCCESS_RATE: float = 0.60
    # Thời gian debounce để tránh gửi callback lặp lại (giây)
    RECOGNITION_DEBOUNCE_SECONDS: int = 30
    
    # Embedding storage
    EMBEDDING_DIR: Optional[str] = None
    DATABASE_DIR: Optional[str] = None
    
    # Registration settings
    REGISTRATION_MIN_CONFIDENCE: float = 0.55
    REGISTRATION_AUGMENTATIONS: int = 5
    SAVE_REGISTRATION_IMAGES: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
