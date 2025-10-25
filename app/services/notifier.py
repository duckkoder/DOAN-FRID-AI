"""
Notifier Service - Gửi callback về backend với HMAC signature
"""
import asyncio
import hmac
import hashlib
import json
from typing import Optional, Dict, Any
from datetime import datetime, timezone

import httpx
from httpx import AsyncClient, Response

from app.models.schemas import AttendanceUpdate
from app.core.config import settings
from app.core.logging import LoggerMixin


class NotificationError(Exception):
    """Exception cho notification errors"""
    pass


class BackendNotifier(LoggerMixin):
    """
    Service gửi callback về backend
    TODO: Implement retry logic, circuit breaker, dead letter queue
    """
    
    def __init__(self):
        super().__init__()
        self.client: Optional[AsyncClient] = None
        self.max_retries = settings.CALLBACK_MAX_RETRIES
        self.retry_delay = settings.CALLBACK_RETRY_DELAY
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.client = AsyncClient(
            timeout=httpx.Timeout(30.0),
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.client:
            await self.client.aclose()
    
    async def send_attendance_update(
        self, 
        callback_url: str, 
        attendance_data: AttendanceUpdate,
        session_id: str
    ) -> bool:
        """
        Gửi cập nhật điểm danh về backend
        
        Args:
            callback_url: URL callback của backend
            attendance_data: Dữ liệu điểm danh
            session_id: ID session để logging
            
        Returns:
            True nếu gửi thành công, False nếu thất bại
        """
        logger = self.get_contextual_logger(session_id=session_id)
        
        if not self.client:
            logger.error("HTTP client not initialized")
            return False
        
        try:
            logger.info("Sending attendance update to backend", callback_url=callback_url)
            
            # Prepare payload
            payload = {
                "session_id": attendance_data.session_id,
                "class_id": attendance_data.class_id,
                "recognized_students": attendance_data.recognized_students,
                "timestamp": attendance_data.timestamp.isoformat() if isinstance(attendance_data.timestamp, datetime) else attendance_data.timestamp,
                "total_faces_detected": attendance_data.total_faces_detected
            }
            
            # Generate HMAC-SHA256 signature
            payload_str = json.dumps(payload, separators=(',', ':'))  # No spaces
            signature = hmac.new(
                settings.BACKEND_CALLBACK_SECRET.encode(),
                payload_str.encode(),
                hashlib.sha256
            ).hexdigest()
            
            logger.debug("Generated HMAC signature", signature_preview=signature[:16])
            
            # Send POST request to backend webhook với signature header
            response = await self.client.post(
                callback_url,
                json=payload,
                headers={
                    "X-AI-Signature": signature,
                    "Content-Type": "application/json"
                },
                timeout=10.0
            )
            
            if response.status_code == 200:
                logger.info("Attendance update sent successfully", 
                           status_code=response.status_code,
                           response_text=response.text)
                return True
            else:
                logger.error("Failed to send attendance update", 
                           status_code=response.status_code,
                           response_text=response.text)
                return False
                
        except httpx.TimeoutException as e:
            logger.error("Timeout sending attendance update", error=str(e))
            return False
        except httpx.RequestError as e:
            logger.error("Request error sending attendance update", error=str(e))
            return False
        except Exception as e:
            logger.error("Unexpected error sending attendance update", error=str(e))
            return False
    
    async def send_attendance_update_with_retry(
        self,
        callback_url: str,
        attendance_data: AttendanceUpdate,
        session_id: str
    ) -> bool:
        """
        Gửi attendance update với retry logic
        
        TODO: Implement exponential backoff, jitter
        """
        logger = self.get_contextual_logger(session_id=session_id)
        
        for attempt in range(self.max_retries + 1):
            try:
                success = await self.send_attendance_update(
                    callback_url, attendance_data, session_id
                )
                
                if success:
                    if attempt > 0:
                        logger.info(f"Attendance update succeeded on attempt {attempt + 1}")
                    return True
                
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(
                        f"Attendance update failed, retrying in {delay}s",
                        attempt=attempt + 1,
                        max_retries=self.max_retries
                    )
                    await asyncio.sleep(delay)
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed", error=str(e))
                
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
        
        logger.error("All retry attempts failed")
        return False
    
    async def health_check(self, callback_url: str) -> bool:
        """
        Kiểm tra health của backend callback endpoint
        
        TODO: Implement thật
        """
        # Stub: Always return healthy
        self.logger.debug("Backend health check (STUB)", callback_url=callback_url)
        return True


# Global notifier instance
backend_notifier = BackendNotifier()
