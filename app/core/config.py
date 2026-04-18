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
    
    # Gemini API (RAG)
    GEMINI_API_KEY: str # Set via ENV: GEMINI_API_KEY

    # RAG embedding model (default: vietnamese-bi-encoder)
    RAG_BI_ENCODER_MODEL: str = "bkai-foundation-models/vietnamese-bi-encoder"

    # AWS S3 (for downloading PDFs during RAG ingestion)
    AWS_ACCESS_KEY_ID: str = ""
    AWS_SECRET_ACCESS_KEY: str = ""
    AWS_REGION: str = "ap-southeast-1"
    S3_MODEL_BUCKET: str = ""

    # Model Paths - Quan trọng cho AWS
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
    
    # ✅ MEMORY OPTIMIZATION SETTINGS (Tuned for AWS g4dn.xlarge - T4 16GB)
    MEMORY_GPU_THRESHOLD: float = 0.88  # Cleanup khi GPU usage > 88% (T4 dư sức)
    MEMORY_CLEANUP_INTERVAL: int = 100  # Cleanup sau mỗi 100 frames (giảm overhead)
    MEMORY_MAX_FACES_PER_FRAME: int = 50  # Tăng lên 50 (T4 16GB xử lý batch lớn thoải mái)
    MEMORY_MAX_IMAGE_SIZE: int = 1280   # Max dimension cho input image
    MEMORY_MAX_SPOOF_CROPS: int = 200   # Tăng lên 200 spoof crops mỗi session
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