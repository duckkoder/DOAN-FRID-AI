"""JWT utilities for WebSocket authentication."""
from typing import Optional, Dict
from jose import jwt, JWTError
from fastapi import HTTPException, status
from datetime import datetime

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def verify_websocket_token(token: str) -> Dict:
    """
    Verify JWT token từ Backend cho WebSocket authentication.
    
    Token structure from Backend:
    {
        "user_id": int,
        "session_id": int,  # Backend session ID
        "role": str,  # "teacher" or "student"
        "type": "websocket",
        "exp": timestamp
    }
    
    Args:
        token: JWT token string
        
    Returns:
        Dict chứa payload nếu valid
        
    Raises:
        HTTPException: Nếu token invalid hoặc expired
    """
    try:
        # Decode token với Backend's secret key
        payload = jwt.decode(
            token,
            settings.BACKEND_JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM]
        )
        
        # Verify token type
        if payload.get("type") != "websocket":
            logger.warning("Invalid token type", token_type=payload.get("type"))
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        # Verify expiry
        exp = payload.get("exp")
        if exp and datetime.utcnow().timestamp() > exp:
            logger.warning("Token expired", exp=exp)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired"
            )
        
        # Extract required fields
        user_id = payload.get("user_id")
        session_id = payload.get("session_id")
        role = payload.get("role")
        
        if not all([user_id, session_id, role]):
            logger.warning(
                "Missing required fields in token",
                user_id=user_id,
                session_id=session_id,
                role=role
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload"
            )
        
        logger.debug(
            "Token verified successfully",
            user_id=user_id,
            session_id=session_id,
            role=role
        )
        
        return payload
        
    except JWTError as e:
        logger.error("JWT verification failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}"
        )


def verify_user_permission(
    token_payload: Dict,
    session_data: "SessionData",
    backend_session_id: int
) -> bool:
    """
    Verify user có quyền truy cập session không.
    
    Args:
        token_payload: Payload từ JWT token
        session_data: SessionData object
        backend_session_id: Backend session ID từ URL
        
    Returns:
        True nếu có quyền, False nếu không
    """
    # Check session_id match
    if token_payload.get("session_id") != backend_session_id:
        logger.warning(
            "Session ID mismatch",
            token_session_id=token_payload.get("session_id"),
            backend_session_id=backend_session_id
        )
        return False
    
    # Check user in allowed_users (if specified)
    user_id = str(token_payload.get("user_id"))
    if session_data.allowed_users and user_id not in session_data.allowed_users:
        logger.warning(
            "User not in allowed list",
            user_id=user_id,
            allowed_users=session_data.allowed_users
        )
        return False
    
    logger.debug(
        "User permission verified",
        user_id=user_id,
        session_id=backend_session_id,
        role=token_payload.get("role")
    )
    
    return True
