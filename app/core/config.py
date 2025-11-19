"""
Cấu hình ứng dụng sử dụng pydantic-settings
"""
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Cấu hình chính của ứng dụng"""
    EMBEDDING_DIR: str = ""
    # App settings
    APP_NAME: str = "AI-Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # Backend Integration
    BACKEND_JWT_SECRET: str = "jB9gwgsbOxaZXKWCTF8BsgYCgLOYROrnwbI4vJWa1T1zG4x0sFG63swllVES3yoj"  # Must match Backend SECRET_KEY
    BACKEND_CALLBACK_SECRET: str = "jB9gwgsbOxaZXKWCTF8BsgYCgLOYROrnwbI4vJWa1T1zG4x0sFG63swllVES3yoj"  # Must match Backend AI_SERVICE_SECRET
    
    # JWT settings
    JWT_ALGORITHM: str = "HS256"

    # Callback settings
    CALLBACK_MAX_RETRIES: int = 3
    CALLBACK_RETRY_DELAY: float = 1.0
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Face Detection settings
    DETECTOR_CHECKPOINT: Optional[str] = None
    DETECTOR_CONF_THRESHOLD: float = 0.75
    DETECTOR_NMS_THRESHOLD: float = 0.4
    DETECTOR_PAD: int = 10

    MODEL_DEVICE: str = "cuda"

    # Face Recognition settings
    RECOGNIZER_CHECKPOINT: Optional[str] = None
    RECOGNIZER_THRESHOLD: float = 1.4
    RECOGNIZER_KNN_K: int = 5

    # Anti-spoofing settings
    ANTISPOOFING_CHECKPOINT: Optional[str] = None
    ANTISPOOFING_THRESHOLD: float = 0.55
    ANTISPOOFING_DEVICE: str = "cuda"
    ANTISPOOFING_BLOCK_RECOGNITION: bool = True  # ✅ Block recognition nếu anti-spoofing fail
    
    # Dynamic threshold settings
    REC_ENABLE_DYNAMIC_THRESHOLD: bool = True
    REC_IDENTITY_QUANTILE: float = 0.8  # ✅ Giảm từ 0.9 xuống 0.75 để tránh outlier
    REC_IDENTITY_MARGIN: float = 0.20    # ✅ Tăng từ 0.15 lên 0.20 (margin_enlarged = 0.20 × 2.5 = 0.50)
    REC_IDENTITY_MIN_SCALE: float = 0.7  # ✅ Giảm từ 0.7 xuống 0.6 (lower_bound = 60% global)

    # TTA (Test Time Augmentation) - Used in registration
    TTA_ENABLED: bool = False
    
    # Calibrated Confidence Settings (weights cho tính toán confidence mới)
    REC_CONFIDENCE_DISTANCE_WEIGHT: float = 0.3  # 30% từ distance margin
    REC_CONFIDENCE_VOTE_WEIGHT: float = 0.7      # 70% từ vote consensus
    
    # Recognition Filtering Settings (để tránh nhận nhầm người lạ)
    REC_MIN_CONFIDENCE: float = 0.5      # Min calibrated confidence (cân bằng giữa strict và lenient)
    REC_MIN_VOTE_RATIO: float = 0.7       # Min vote ratio từ KNN (chặt để tránh false positive)
    REC_REQUIRE_STABLE: bool = False      # Yêu cầu stable qua temporal smoothing (để False cho đơn giản)
    REC_MAX_DISTANCE_RATIO: float = 0.95  # ✅ Distance phải < 90% threshold (chặt hơn để an toàn)
    
    # Recognition Validation Settings (Anti-premature detection)
    RECOGNITION_CONFIRMATION_THRESHOLD: int = 3  # Min recognition count in window (3/5 = 60%)
    RECOGNITION_WINDOW_SIZE: int = 5  # Number of recent frames to consider
    RECOGNITION_MIN_FRAME_SUCCESS_RATE: float = 0.60  # Min success rate (3/5 = 60%)
    RECOGNITION_DEBOUNCE_SECONDS: int = 30  # Cooldown before re-sending callback
    
    # PostgreSQL pgvector connection
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "attendance_db"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "123qwe!%40%23"
    
    @property
    def DATABASE_URL(self) -> str:
        """Get PostgreSQL connection URL."""
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        # return f"postgresql://bietto:123qwe!%40%23@frid-db.postgres.database.azure.com:5432/attendance_db"
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
