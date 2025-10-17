"""
JWT Authentication và Security utilities
"""
import base64
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from jwt.exceptions import InvalidTokenError
import structlog

from app.core.config import settings

logger = structlog.get_logger(__name__)
security = HTTPBearer()


class JWTHandler:
    """Xử lý JWT token validation"""
    
    def __init__(self):
        self.algorithm = settings.JWT_ALGORITHM
        self.secret_key = self._get_secret_key()
    
    def _get_secret_key(self) -> str:
        """Lấy secret key để verify JWT"""
        if self.algorithm.startswith("HS"):
            if not settings.JWT_SECRET:
                # TODO: Trong production cần secret key thật
                logger.warning("Using default JWT secret - NOT for production!")
                return "default-secret-key-change-in-production"
            return settings.JWT_SECRET
        elif self.algorithm.startswith("RS"):
            if not settings.JWT_PUBLIC_KEY_BASE64:
                raise ValueError("RS256 requires JWT_PUBLIC_KEY_BASE64")
            return base64.b64decode(settings.JWT_PUBLIC_KEY_BASE64).decode()
        else:
            raise ValueError(f"Unsupported JWT algorithm: {self.algorithm}")
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify và decode JWT token"""
        try:
            payload = jwt.decode(
                token, 
                self.secret_key, 
                algorithms=[self.algorithm]
            )
            
            # Kiểm tra required claims
            for claim in settings.JWT_REQUIRED_CLAIMS:
                if claim not in payload:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail=f"Missing required claim: {claim}"
                    )
            
            # Kiểm tra audience và issuer nếu được cấu hình
            if settings.JWT_AUDIENCE and payload.get("aud") != settings.JWT_AUDIENCE:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid audience"
                )
            
            if settings.JWT_ISSUER and payload.get("iss") != settings.JWT_ISSUER:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid issuer"
                )
            
            return payload
            
        except InvalidTokenError as e:
            logger.error("JWT verification failed", error=str(e))
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )


jwt_handler = JWTHandler()


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """Dependency để lấy thông tin user từ JWT token"""
    return jwt_handler.verify_token(credentials.credentials)


def create_stub_token() -> str:
    """Tạo stub token cho testing - TODO: Remove in production"""
    payload = {
        "sub": "test-user",
        "exp": int((datetime.now(timezone.utc).timestamp())) + 3600,
        "iat": int(datetime.now(timezone.utc).timestamp()),
        "aud": settings.JWT_AUDIENCE or "ai-service",
        "iss": settings.JWT_ISSUER or "test-issuer"
    }
    
    return jwt.encode(payload, jwt_handler.secret_key, algorithm=jwt_handler.algorithm)
