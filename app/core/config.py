"""
Cấu hình ứng dụng sử dụng pydantic-settings
"""
import urllib.parse
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Cấu hình chính của ứng dụng"""
    
    # App settings
    APP_NAME: str = "AI-Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "development"  # development, staging, production
    
    # Backend Integration - BẮT BUỘC phải set qua ENV, không có default
    BACKEND_JWT_SECRET: str  # Must match Backend SECRET_KEY
    BACKEND_CALLBACK_SECRET: str  # Must match Backend AI_SERVICE_SECRET
    
    # JWT settings
    JWT_ALGORITHM: str = "HS256"

    # Callback settings
    CALLBACK_MAX_RETRIES: int = 3
    CALLBACK_RETRY_DELAY: float = 1.0
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8069
    
    # Model Paths - Quan trọng cho AWS
    EMBEDDING_DIR: str = ""
    DETECTOR_CHECKPOINT: Optional[str] = None
    RECOGNIZER_CHECKPOINT: Optional[str] = None
    ANTISPOOFING_CHECKPOINT: Optional[str] = None
    
    # Model Device settings
    MODEL_DEVICE: str = "cuda"
    ANTISPOOFING_DEVICE: str = "cuda"
    
    # Face Detection settings
    DETECTOR_CONF_THRESHOLD: float = 0.75
    DETECTOR_NMS_THRESHOLD: float = 0.4
    DETECTOR_PAD: int = 10

    # Face Recognition settings
    RECOGNIZER_THRESHOLD: float = 1.2
    RECOGNIZER_KNN_K: int = 5
    RECOGNIZER_KNN_VOTING_THRESHOLD: float = 1.2

    # Anti-spoofing settings
    ANTISPOOFING_THRESHOLD: float = 0.55
    ANTISPOOFING_BLOCK_RECOGNITION: bool = True
    
    # Dynamic threshold settings
    REC_ENABLE_DYNAMIC_THRESHOLD: bool = True
    REC_IDENTITY_QUANTILE: float = 0.8
    REC_IDENTITY_MARGIN: float = 0.20
    REC_IDENTITY_MIN_SCALE: float = 0.7

    # TTA (Test Time Augmentation)
    TTA_ENABLED: bool = False
    
    # Calibrated Confidence Settings
    REC_CONFIDENCE_DISTANCE_WEIGHT: float = 0.3
    REC_CONFIDENCE_VOTE_WEIGHT: float = 0.7
    
    # Recognition Filtering Settings
    REC_MIN_CONFIDENCE: float = 0.5
    REC_MIN_VOTE_RATIO: float = 0.7
    REC_MIN_VALID_NEIGHBORS_RATIO: float = 0.7
    REC_REQUIRE_STABLE: bool = False
    REC_MAX_DISTANCE_RATIO: float = 0.95
    
    # Recognition Validation Settings
    RECOGNITION_CONFIRMATION_THRESHOLD: int = 3
    RECOGNITION_WINDOW_SIZE: int = 5
    RECOGNITION_MIN_FRAME_SUCCESS_RATE: float = 0.60
    RECOGNITION_DEBOUNCE_SECONDS: int = 30
    
    # ✅ MEMORY OPTIMIZATION SETTINGS
    MEMORY_GPU_THRESHOLD: float = 0.85  # Cleanup khi GPU usage > 85%
    MEMORY_CLEANUP_INTERVAL: int = 50   # Cleanup sau mỗi N frames
    MEMORY_MAX_FACES_PER_FRAME: int = 10  # Max faces xử lý mỗi frame
    MEMORY_MAX_IMAGE_SIZE: int = 1280   # Max dimension cho input image
    MEMORY_MAX_SPOOF_CROPS: int = 50    # Max spoof crops lưu mỗi session
    MEMORY_AGGRESSIVE_GC: bool = True   # Bật aggressive garbage collection
    
    # PostgreSQL pgvector connection - BẮT BUỘC qua ENV
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "ai_attendance"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str  # BẮT BUỘC - set qua ENV, không có default
    
    # Cho phép ghi đè trực tiếp nguyên cả URL nếu muốn
    DATABASE_URL_OVERRIDE: Optional[str] = None
    
    @property
    def DATABASE_URL(self) -> str:
        """Get PostgreSQL connection URL."""
        if self.DATABASE_URL_OVERRIDE:
            return self.DATABASE_URL_OVERRIDE
        
        # Mã hóa mật khẩu để xử lý ký tự đặc biệt (ví dụ dấu @)
        safe_password = urllib.parse.quote_plus(self.POSTGRES_PASSWORD)
        return f"postgresql://{self.POSTGRES_USER}:{safe_password}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"



# Global settings instance
settings = Settings()