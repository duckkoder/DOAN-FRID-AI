"""
Cáº¥u hÃ¬nh á»©ng dá»¥ng sá»­ dá»¥ng pydantic-settings
"""
from typing import Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Cáº¥u hÃ¬nh chÃ­nh cá»§a á»©ng dá»¥ng"""
    
    # App settings
    APP_NAME: str = "AI-Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    ACCESS_LOG: bool = False
    ENVIRONMENT: str = "development"  # development, staging, production
    
    # Backend Integration - Báº®T BUá»˜C pháº£i set qua ENV, khÃ´ng cÃ³ default
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
    
    # RAG embedding model (default: vietnamese-bi-encoder)
    RAG_BI_ENCODER_MODEL: str = "bkai-foundation-models/vietnamese-bi-encoder"

    # Model Paths - Quan trá»ng cho AWS
    EMBEDDING_DIR: str = ""
    DETECTOR_CHECKPOINT: Optional[str] = None
    RECOGNIZER_CHECKPOINT: Optional[str] = None
    ANTISPOOFING_CHECKPOINT: Optional[str] = None
    
    # Model Device settings
    MODEL_DEVICE: str = "cuda"
    ANTISPOOFING_DEVICE: str = "cuda"
    
    # Face Detection settings
    DETECTOR_CONF_THRESHOLD: float
    DETECTOR_NMS_THRESHOLD: float
    DETECTOR_PAD: int

    # Face Recognition settings
    RECOGNIZER_THRESHOLD: float
    RECOGNIZER_KNN_K: int
    RECOGNIZER_KNN_VOTING_THRESHOLD: float

    # Anti-spoofing settings
    ANTISPOOFING_THRESHOLD: float
    ANTISPOOFING_BLOCK_RECOGNITION: bool
    
    # Dynamic threshold settings
    REC_ENABLE_DYNAMIC_THRESHOLD: bool
    REC_IDENTITY_QUANTILE: float
    REC_IDENTITY_MARGIN: float
    REC_IDENTITY_MIN_SCALE: float

    # TTA (Test Time Augmentation)
    TTA_ENABLED: bool
    
    # Calibrated Confidence Settings
    REC_CONFIDENCE_DISTANCE_WEIGHT: float
    REC_CONFIDENCE_VOTE_WEIGHT: float
    
    # Recognition Filtering Settings
    REC_MIN_CONFIDENCE: float
    REC_MIN_VOTE_RATIO: float
    REC_MIN_VALID_NEIGHBORS_RATIO: float
    REC_REQUIRE_STABLE: bool
    REC_MAX_DISTANCE_RATIO: float
    
    # Recognition Validation Settings
    RECOGNITION_CONFIRMATION_THRESHOLD: int
    RECOGNITION_WINDOW_SIZE: int
    RECOGNITION_MIN_FRAME_SUCCESS_RATE: float
    RECOGNITION_DEBOUNCE_SECONDS: int = 30

    # Attendance stream throttling/cache settings
    ATTENDANCE_HEAVY_PROCESS_FACE_THRESHOLD: int = 8
    ATTENDANCE_HEAVY_PROCESS_INTERVAL: int = 3
    ATTENDANCE_RECOGNITION_CACHE_TTL_FRAMES: int = 8
    
    # âœ… MEMORY OPTIMIZATION SETTINGS (Tuned for AWS g4dn.xlarge - T4 16GB)
    MEMORY_GPU_THRESHOLD: float = 0.88  # Cleanup khi GPU usage > 88% (T4 dÆ° sá»©c)
    MEMORY_CLEANUP_INTERVAL: int = 100  # Cleanup sau má»—i 100 frames (giáº£m overhead)
    MEMORY_MAX_FACES_PER_FRAME: int = 50  # TÄƒng lÃªn 50 (T4 16GB xá»­ lÃ½ batch lá»›n thoáº£i mÃ¡i)
    MEMORY_MAX_IMAGE_SIZE: int = 1280   # Max dimension cho input image
    MEMORY_MAX_SPOOF_CROPS: int = 200   # TÄƒng lÃªn 200 spoof crops má»—i session
    MEMORY_AGGRESSIVE_GC: bool = True   # Báº­t aggressive garbage collection
    
    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug(cls, value):
        if isinstance(value, str) and value.lower() in {"release", "production", "prod"}:
            return False
        return value
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"



# Global settings instance
settings = Settings()
