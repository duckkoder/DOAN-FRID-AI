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
    backend_session_id: int = Field(..., description="Backend session ID để mapping")
    class_id: str = Field(..., description="ID của lớp học")
    student_codes: List[str] = Field(..., description="Danh sách student codes trong lớp (100 students)")
    backend_callback_url: str = Field(..., description="URL callback để thông báo backend")
    ws_token: str = Field(..., description="JWT token mẫu để verify (optional, for reference)")
    allowed_users: List[str] = Field(default_factory=list, description="Danh sách user_ids được phép (RBAC)")
    s3: Optional[S3Config] = Field(None, description="Cấu hình S3 cho embeddings (optional, deprecated)")
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
    student_code: Optional[str] = Field(None, description="Mã sinh viên nếu được nhận diện (student_code)")
    student_name: Optional[str] = Field(None, description="Tên sinh viên nếu được nhận diện")
    recognition_confidence: Optional[float] = Field(None, description="Confidence của recognition")
    
    # ✅ Backward compatibility alias - for gradual migration
    @property
    def student_id(self) -> Optional[str]:
        """Alias for student_code (backward compatibility)"""
        return self.student_code


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
class ValidatedStudent(BaseModel):
    """Thông tin sinh viên đã được validate (match Backend's AIValidatedStudent)"""
    student_code: str = Field(..., description="Mã sinh viên")
    student_name: str = Field(..., description="Tên sinh viên")
    track_id: int = Field(..., description="Tracking ID")
    avg_confidence: float = Field(..., description="Độ tin cậy trung bình")
    frame_count: int = Field(..., description="Số frame đã xử lý")
    recognition_count: int = Field(..., description="Số lần nhận diện thành công")
    validation_passed_at: datetime = Field(..., description="Thời điểm pass validation")


class AttendanceUpdate(BaseModel):
    """Thông tin cập nhật điểm danh gửi về backend (match Backend's AICallbackPayload)"""
    session_id: str = Field(..., description="AI session ID")
    validated_students: List[ValidatedStudent] = Field(..., description="Danh sách sinh viên đã validate")
    timestamp: datetime = Field(..., description="Thời gian callback")


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


# Registration schemas
class RegistrationRequest(BaseModel):
    """Request để đăng ký người dùng mới"""
    person_name: str = Field(..., description="Tên người cần đăng ký")
    image_base64: str = Field(..., description="Ảnh encoded base64")
    min_confidence: Optional[float] = Field(None, description="Ngưỡng confidence tối thiểu")
    augmentations: Optional[int] = Field(None, description="Số lượng ảnh augmented")
    save_image: Optional[bool] = Field(True, description="Có lưu ảnh crop không")


class RegistrationResponse(BaseModel):
    """Response sau khi đăng ký người dùng"""
    success: bool = Field(..., description="Trạng thái đăng ký")
    message: str = Field(..., description="Thông báo kết quả")
    identity: str = Field(..., description="Tên đã được làm sạch")
    embeddings_saved: int = Field(..., description="Số embeddings đã lưu")
    detection_confidence: Optional[float] = Field(None, description="Độ tin cậy detection")
    timestamp: datetime = Field(..., description="Thời gian đăng ký")


class BatchRegistrationRequest(BaseModel):
    """Request để đăng ký hàng loạt từ thư mục"""
    source_dir: str = Field(..., description="Đường dẫn thư mục nguồn")
    min_images: Optional[int] = Field(1, description="Số ảnh tối thiểu cho mỗi người")
    augmentations: Optional[int] = Field(None, description="Số ảnh augmented")


class BatchRegistrationResponse(BaseModel):
    """Response sau khi đăng ký hàng loạt"""
    success: bool = Field(..., description="Trạng thái")
    message: str = Field(..., description="Thông báo")
    stats: Dict[str, int] = Field(..., description="Thống kê {identity: num_embeddings}")
    total_people: int = Field(..., description="Tổng số người đã đăng ký")
    total_embeddings: int = Field(..., description="Tổng số embeddings")
    timestamp: datetime = Field(..., description="Thời gian hoàn thành")


# Embedding management schemas
class EmbeddingStats(BaseModel):
    """Thống kê embeddings database"""
    num_people: int = Field(..., description="Số lượng người trong database")
    total_vectors: int = Field(..., description="Tổng số vectors")
    timestamp: datetime = Field(..., description="Thời gian lấy thống kê")


class RefreshDatabaseRequest(BaseModel):
    """Request để refresh database"""
    embedding_dir: Optional[str] = Field(None, description="Đường dẫn thư mục embeddings")


class RefreshDatabaseResponse(BaseModel):
    """Response sau khi refresh database"""
    success: bool = Field(..., description="Trạng thái")
    message: str = Field(..., description="Thông báo")
    num_people: int = Field(..., description="Số người trong database")
    total_vectors: int = Field(..., description="Tổng số vectors")
    timestamp: datetime = Field(..., description="Thời gian refresh")