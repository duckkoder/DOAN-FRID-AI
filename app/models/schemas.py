"""
Pydantic models cho request/response schemas
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


# Session related schemas
class S3Config(BaseModel):
    """Cấu hình S3 cho embeddings"""
    bucket: str = Field(..., description="S3 bucket name")
    key: str = Field(..., description="S3 object key")
    region: Optional[str] = Field(None, description="S3 region")


class SessionCreateRequest(BaseModel):
    """Request để tạo session mới"""
    class_id: str = Field(..., description="ID của lớp học")
    backend_callback_url: str = Field(..., description="URL callback để thông báo backend")
    s3: S3Config = Field(..., description="Cấu hình S3 cho embeddings")
    max_duration_minutes: Optional[int] = Field(60, description="Thời gian tối đa của session (phút)")


class SessionResponse(BaseModel):
    """Response khi tạo hoặc lấy thông tin session"""
    session_id: str = Field(..., description="ID của session")
    class_id: str = Field(..., description="ID của lớp học")
    status: str = Field(..., description="Trạng thái session: active, ended")
    created_at: datetime = Field(..., description="Thời gian tạo session")
    backend_callback_url: str = Field(..., description="URL callback")
    s3_config: S3Config = Field(..., description="Cấu hình S3")
    embeddings_loaded: bool = Field(..., description="Trạng thái load embeddings")
    total_frames_processed: int = Field(0, description="Tổng số frame đã xử lý")


# Frame processing schemas
class FrameRequest(BaseModel):
    """Request để xử lý frame"""
    frame_base64: Optional[str] = Field(None, description="Frame ảnh encoded base64 (optional)")
    timestamp: datetime = Field(..., description="Timestamp của frame")
    client_seq: Optional[int] = Field(None, description="Sequence number từ client")


class Detection(BaseModel):
    """Thông tin detection của một khuôn mặt"""
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    confidence: float = Field(..., description="Confidence score của detection")
    track_id: Optional[int] = Field(None, description="ID tracking của khuôn mặt")
    student_id: Optional[str] = Field(None, description="ID sinh viên nếu được nhận diện")
    recognition_confidence: Optional[float] = Field(None, description="Confidence của recognition")


class FrameResponse(BaseModel):
    """Response sau khi xử lý frame"""
    session_id: str = Field(..., description="ID của session")
    timestamp: datetime = Field(..., description="Timestamp của frame")
    client_seq: Optional[int] = Field(None, description="Echo client sequence number")
    processed_at: datetime = Field(..., description="Thời gian xử lý")
    detections: List[Detection] = Field(..., description="Danh sách detections")
    recognized_student_ids: List[str] = Field(..., description="Danh sách ID sinh viên được nhận diện")
    total_faces: int = Field(..., description="Tổng số khuôn mặt detected")
    callback_sent: bool = Field(False, description="Trạng thái gửi callback")


# Callback schemas
class AttendanceUpdate(BaseModel):
    """Thông tin cập nhật điểm danh gửi về backend"""
    session_id: str = Field(..., description="ID của session")
    class_id: str = Field(..., description="ID của lớp học")
    timestamp: datetime = Field(..., description="Thời gian")
    recognized_students: List[str] = Field(..., description="Danh sách sinh viên được nhận diện")
    total_faces_detected: int = Field(..., description="Tổng số khuôn mặt detected")


# Health check schema
class HealthResponse(BaseModel):
    """Response cho health check"""
    status: str = Field("healthy", description="Trạng thái service")
    timestamp: datetime = Field(..., description="Thời gian check")
    version: str = Field(..., description="Version của service")
    active_sessions: int = Field(..., description="Số session đang hoạt động")


# Error schemas
class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str = Field(..., description="Mã lỗi")
    message: str = Field(..., description="Thông báo lỗi")
    details: Optional[Dict[str, Any]] = Field(None, description="Chi tiết lỗi")
    timestamp: datetime = Field(..., description="Thời gian xảy ra lỗi")
