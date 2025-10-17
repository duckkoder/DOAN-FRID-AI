"""
Notifier Service - Gửi callback về backend
"""
import asyncio
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
            
        TODO: Implement retry với exponential backoff
        """
        logger = self.get_contextual_logger(session_id=session_id)
        
        if not self.client:
            logger.error("HTTP client not initialized")
            return False
        
        # TODO: Implement thật - hiện tại chỉ là stub
        try:
            logger.info("Sending attendance update (STUB)", callback_url=callback_url)
            
            # Stub: Giả lập gửi request
            await asyncio.sleep(0.1)  # Simulate network delay
            
            # Stub: Random success/failure
            import random
            success = random.random() > 0.1  # 90% success rate
            
            if success:
                logger.info("Attendance update sent successfully (STUB)")
                return True
            else:
                logger.error("Failed to send attendance update (STUB)")
                return False
                
        except Exception as e:
            logger.error("Error sending attendance update", error=str(e))
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
