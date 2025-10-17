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
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()
