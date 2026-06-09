"""
Cấu hình ứng dụng sử dụng pydantic-settings
"""
from typing import Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Cấu hình chính của ứng dụng"""
    
    # App settings
    APP_NAME: str = "AI-Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    ACCESS_LOG: bool = False
    ENVIRONMENT: str = "development"
    EMBEDDING_DIR: str = ""
    
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
    DETECTOR_CONF_THRESHOLD: float = 0.8
    DETECTOR_NMS_THRESHOLD: float = 0.4
    DETECTOR_PAD: int = 10

    MODEL_DEVICE: str = "cuda"

    # Face Recognition settings
    RECOGNIZER_CHECKPOINT: Optional[str] = None
    RECOGNIZER_THRESHOLD: float = 1.1
    RECOGNIZER_KNN_K: int = 5
    RECOGNIZER_KNN_VOTING_THRESHOLD: float = 1.2

    # Anti-spoofing settings
    ANTISPOOFING_CHECKPOINT: Optional[str] = None
    ANTISPOOFING_THRESHOLD: float = 0.55
    ANTISPOOFING_DEVICE: str = "cuda"
    
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
    REC_MIN_CONFIDENCE: float = 0.45      # Min calibrated confidence (cân bằng giữa strict và lenient)
    REC_MIN_VOTE_RATIO: float = 0.7       # Min vote ratio từ KNN (chặt để tránh false positive)
    REC_REQUIRE_STABLE: bool = False      # Yêu cầu stable qua temporal smoothing (để False cho đơn giản)
    REC_MAX_DISTANCE_RATIO: float = 0.85  # ✅ Distance phải < 90% threshold (chặt hơn để an toàn)
    
    REC_MIN_VALID_NEIGHBORS_RATIO: float = 0.7

    # Recognition Validation Settings (Anti-premature detection)
    RECOGNITION_CONFIRMATION_THRESHOLD: int = 3  # Min recognition count in window (3/5 = 60%)
    RECOGNITION_WINDOW_SIZE: int = 5  # Number of recent frames to consider
    RECOGNITION_MIN_FRAME_SUCCESS_RATE: float = 0.60  # Min success rate (3/5 = 60%)
    RECOGNITION_DEBOUNCE_SECONDS: int = 30  # Cooldown before re-sending callback
    RECOGNITION_AUTO_ADJUST_TO_FPS: bool = False
    RECOGNITION_TARGET_FPS: float = 5.0

    # Worker / resource settings
    AI_FACE_WORKERS: int = 4
    AI_RAG_WORKERS: int = 1
    AI_IO_WORKERS: int = 2
    RAG_EMBEDDING_DEVICE: str = "cpu"
    RAG_EMBEDDING_BATCH_SIZE: int = 16
    RAG_BI_ENCODER_MODEL: str = "bkai-foundation-models/vietnamese-bi-encoder"

    # Memory safety settings
    MEMORY_GPU_THRESHOLD: float = 0.85
    MEMORY_CLEANUP_INTERVAL: int = 50
    MEMORY_MAX_FACES_PER_FRAME: int = 10
    MEMORY_MAX_IMAGE_SIZE: int = 1280
    MEMORY_MAX_SPOOF_CROPS: int = 50
    MEMORY_AGGRESSIVE_GC: bool = True

    # Attendance realtime tuning
    ATTENDANCE_DETECTION_INTERVAL: int = 1
    ATTENDANCE_HEAVY_PROCESS_FACE_THRESHOLD: int = 8
    ATTENDANCE_HEAVY_PROCESS_INTERVAL: int = 2
    ATTENDANCE_RECOGNITION_CACHE_TTL_FRAMES: int = 8

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, value):
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "production", "prod"}:
                return False
            if normalized in {"development", "dev", "debug"}:
                return True
        return value
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


# Global settings instance
settings = Settings()
