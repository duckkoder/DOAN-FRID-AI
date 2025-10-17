"""
Session Manager - Quản lý sessions in-memory
"""
import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass

from app.models.schemas import SessionCreateRequest, SessionResponse, S3Config
from app.core.logging import LoggerMixin


@dataclass
class SessionData:
    """Dữ liệu session lưu trong memory"""
    session_id: str
    class_id: str
    backend_callback_url: str
    s3_config: S3Config
    status: str
    created_at: datetime
    embeddings_loaded: bool = False
    total_frames_processed: int = 0
    max_duration_minutes: int = 60


class SessionManager(LoggerMixin):
    """Quản lý sessions in-memory với thread safety"""
    
    def __init__(self):
        super().__init__()
        self._sessions: Dict[str, SessionData] = {}
        self._lock = asyncio.Lock()
    
    async def create_session(self, request: SessionCreateRequest) -> SessionResponse:
        """
        Tạo session mới
        
        Args:
            request: Thông tin tạo session
            
        Returns:
            Thông tin session đã tạo
        """
        session_id = str(uuid.uuid4())
        
        async with self._lock:
            session_data = SessionData(
                session_id=session_id,
                class_id=request.class_id,
                backend_callback_url=request.backend_callback_url,
                s3_config=request.s3,
                status="active",
                created_at=datetime.now(timezone.utc),
                max_duration_minutes=request.max_duration_minutes or 60
            )
            
            self._sessions[session_id] = session_data
            
            # TODO: Trigger embeddings loading từ S3
            # Stub: Giả lập load embeddings thành công
            session_data.embeddings_loaded = True
            
            self.logger.info(
                "Session created",
                session_id=session_id,
                class_id=request.class_id,
                s3_bucket=request.s3.bucket,
                s3_key=request.s3.key
            )
            
            return self._session_data_to_response(session_data)
    
    async def get_session(self, session_id: str) -> Optional[SessionResponse]:
        """
        Lấy thông tin session
        
        Args:
            session_id: ID của session
            
        Returns:
            Thông tin session hoặc None nếu không tồn tại
        """
        async with self._lock:
            session_data = self._sessions.get(session_id)
            if not session_data:
                return None
            
            # Kiểm tra session có hết hạn không
            if self._is_session_expired(session_data):
                session_data.status = "expired"
            
            return self._session_data_to_response(session_data)
    
    async def delete_session(self, session_id: str) -> bool:
        """
        Xóa session
        
        Args:
            session_id: ID của session
            
        Returns:
            True nếu xóa thành công, False nếu không tồn tại
        """
        async with self._lock:
            session_data = self._sessions.get(session_id)
            if not session_data:
                return False
            
            session_data.status = "ended"
            del self._sessions[session_id]
            
            self.logger.info(
                "Session deleted",
                session_id=session_id,
                class_id=session_data.class_id,
                total_frames=session_data.total_frames_processed
            )
            
            return True
    
    async def increment_frame_count(self, session_id: str) -> bool:
        """
        Tăng số lượng frame đã xử lý
        
        Args:
            session_id: ID của session
            
        Returns:
            True nếu thành công, False nếu session không tồn tại
        """
        async with self._lock:
            session_data = self._sessions.get(session_id)
            if not session_data:
                return False
            
            session_data.total_frames_processed += 1
            return True
    
    async def get_active_sessions_count(self) -> int:
        """Lấy số lượng session đang hoạt động"""
        async with self._lock:
            active_count = 0
            for session_data in self._sessions.values():
                if session_data.status == "active" and not self._is_session_expired(session_data):
                    active_count += 1
            return active_count
    
    async def cleanup_expired_sessions(self) -> int:
        """
        Dọn dẹp các session đã hết hạn
        
        Returns:
            Số lượng session đã được dọn dẹp
        """
        async with self._lock:
            expired_sessions = []
            
            for session_id, session_data in self._sessions.items():
                if self._is_session_expired(session_data):
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self._sessions[session_id]
                self.logger.info("Expired session cleaned up", session_id=session_id)
            
            return len(expired_sessions)
    
    def _session_data_to_response(self, session_data: SessionData) -> SessionResponse:
        """Chuyển đổi SessionData thành SessionResponse"""
        return SessionResponse(
            session_id=session_data.session_id,
            class_id=session_data.class_id,
            status=session_data.status,
            created_at=session_data.created_at,
            backend_callback_url=session_data.backend_callback_url,
            s3_config=session_data.s3_config,
            embeddings_loaded=session_data.embeddings_loaded,
            total_frames_processed=session_data.total_frames_processed
        )
    
    def _is_session_expired(self, session_data: SessionData) -> bool:
        """Kiểm tra session có hết hạn không"""
        if session_data.status != "active":
            return True
        
        expiry_time = session_data.created_at + timedelta(minutes=session_data.max_duration_minutes)
        return datetime.now(timezone.utc) > expiry_time


# Global session manager instance
session_manager = SessionManager()
