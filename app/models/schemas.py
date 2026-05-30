"""
Pydantic models cho request/response schemas
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


# Session related schemas
class SessionCreateRequest(BaseModel):
    """Request Ä‘á»ƒ táº¡o session má»›i"""
    backend_session_id: int = Field(..., description="Backend session ID Ä‘á»ƒ mapping")
    class_id: str = Field(..., description="ID cá»§a lá»›p há»c")
    student_codes: List[str] = Field(..., description="Danh sÃ¡ch student codes trong lá»›p (100 students)")
    face_embeddings: List[Dict[str, Any]] = Field(default_factory=list, description="Approved face embeddings supplied by Backend")
    backend_callback_url: str = Field(..., description="URL callback Ä‘á»ƒ thÃ´ng bÃ¡o backend")
    ws_token: str = Field(..., description="JWT token máº«u Ä‘á»ƒ verify (optional, for reference)")
    allowed_users: List[str] = Field(default_factory=list, description="Danh sÃ¡ch user_ids Ä‘Æ°á»£c phÃ©p (RBAC)")
    max_duration_minutes: Optional[int] = Field(60, description="Thá»i gian tá»‘i Ä‘a cá»§a session (phÃºt)")


class SessionResponse(BaseModel):
    """Response khi táº¡o hoáº·c láº¥y thÃ´ng tin session"""
    session_id: str = Field(..., description="ID cá»§a session")
    class_id: str = Field(..., description="ID cá»§a lá»›p há»c")
    status: str = Field(..., description="Tráº¡ng thÃ¡i session: active, ended")
    created_at: datetime = Field(..., description="Thá»i gian táº¡o session")
    backend_callback_url: str = Field(..., description="URL callback")
    embeddings_loaded: bool = Field(..., description="Tráº¡ng thÃ¡i load embeddings")
    total_frames_processed: int = Field(0, description="Tá»•ng sá»‘ frame Ä‘Ã£ xá»­ lÃ½")


class Detection(BaseModel):
    """ThÃ´ng tin detection cá»§a má»™t khuÃ´n máº·t"""
    bbox: List[float] = Field(..., description="Bounding box [x1, y1, x2, y2]")
    confidence: float = Field(..., description="Confidence score cá»§a detection")
    track_id: Optional[int] = Field(None, description="ID tracking cá»§a khuÃ´n máº·t")
    student_code: Optional[str] = Field(None, description="MÃ£ sinh viÃªn náº¿u Ä‘Æ°á»£c nháº­n diá»‡n (student_code)")
    student_name: Optional[str] = Field(None, description="TÃªn sinh viÃªn náº¿u Ä‘Æ°á»£c nháº­n diá»‡n")
    recognition_confidence: Optional[float] = Field(None, description="Confidence cá»§a recognition")
    
    # âœ… Anti-spoofing fields
    is_live: Optional[bool] = Field(None, description="True náº¿u lÃ  live face, False náº¿u print/replay")
    spoofing_type: Optional[str] = Field(None, description="Loáº¡i: 'live', 'print', 'replay'")
    spoofing_confidence: Optional[float] = Field(None, description="Äá»™ tin cáº­y cá»§a anti-spoofing prediction (0.0-1.0)")


    # âœ… Backward compatibility alias - for gradual migration
    @property
    def student_id(self) -> Optional[str]:
        """Alias for student_code (backward compatibility)"""
        return self.student_code
# Callback schemas
class ValidatedStudent(BaseModel):
    """ThÃ´ng tin sinh viÃªn Ä‘Ã£ Ä‘Æ°á»£c validate (match Backend's AIValidatedStudent)"""
    student_code: str = Field(..., description="MÃ£ sinh viÃªn")
    student_name: str = Field(..., description="TÃªn sinh viÃªn")
    track_id: int = Field(..., description="Tracking ID")
    avg_confidence: float = Field(..., description="Äá»™ tin cáº­y trung bÃ¬nh")
    frame_count: int = Field(..., description="Sá»‘ frame Ä‘Ã£ xá»­ lÃ½")
    recognition_count: int = Field(..., description="Sá»‘ láº§n nháº­n diá»‡n thÃ nh cÃ´ng")
    validation_passed_at: datetime = Field(..., description="Thá»i Ä‘iá»ƒm pass validation")


class AttendanceUpdate(BaseModel):
    """ThÃ´ng tin cáº­p nháº­t Ä‘iá»ƒm danh gá»­i vá» backend (match Backend's AICallbackPayload)"""
    session_id: str = Field(..., description="AI session ID")
    validated_students: List[ValidatedStudent] = Field(..., description="Danh sÃ¡ch sinh viÃªn Ä‘Ã£ validate")
    timestamp: datetime = Field(..., description="Thá»i gian callback")


# Health check schema
class HealthResponse(BaseModel):
    """Response cho health check"""
    status: str = Field("healthy", description="Tráº¡ng thÃ¡i service")
    timestamp: datetime = Field(..., description="Thá»i gian check")
    version: str = Field(..., description="Version cá»§a service")
    active_sessions: int = Field(..., description="Sá»‘ session Ä‘ang hoáº¡t Ä‘á»™ng")


# Registration schemas
class RegistrationRequest(BaseModel):
    """Request Ä‘á»ƒ Ä‘Äƒng kÃ½ ngÆ°á»i dÃ¹ng má»›i"""
    person_name: str = Field(..., description="TÃªn ngÆ°á»i cáº§n Ä‘Äƒng kÃ½")
    image_base64: str = Field(..., description="áº¢nh encoded base64")
    min_confidence: Optional[float] = Field(None, description="NgÆ°á»¡ng confidence tá»‘i thiá»ƒu")
    augmentations: Optional[int] = Field(None, description="Sá»‘ lÆ°á»£ng áº£nh augmented")
    save_image: Optional[bool] = Field(True, description="CÃ³ lÆ°u áº£nh crop khÃ´ng")


class RegistrationResponse(BaseModel):
    """Response sau khi Ä‘Äƒng kÃ½ ngÆ°á»i dÃ¹ng"""
    success: bool = Field(..., description="Tráº¡ng thÃ¡i Ä‘Äƒng kÃ½")
    message: str = Field(..., description="ThÃ´ng bÃ¡o káº¿t quáº£")
    identity: str = Field(..., description="TÃªn Ä‘Ã£ Ä‘Æ°á»£c lÃ m sáº¡ch")
    embeddings_saved: int = Field(..., description="Sá»‘ embeddings Ä‘Ã£ lÆ°u")
    detection_confidence: Optional[float] = Field(None, description="Äá»™ tin cáº­y detection")
    timestamp: datetime = Field(..., description="Thá»i gian Ä‘Äƒng kÃ½")


class BatchRegistrationRequest(BaseModel):
    """Request Ä‘á»ƒ Ä‘Äƒng kÃ½ hÃ ng loáº¡t tá»« thÆ° má»¥c"""
    source_dir: str = Field(..., description="ÄÆ°á»ng dáº«n thÆ° má»¥c nguá»“n")
    min_images: Optional[int] = Field(1, description="Sá»‘ áº£nh tá»‘i thiá»ƒu cho má»—i ngÆ°á»i")
    augmentations: Optional[int] = Field(None, description="Sá»‘ áº£nh augmented")


class BatchRegistrationResponse(BaseModel):
    """Response sau khi Ä‘Äƒng kÃ½ hÃ ng loáº¡t"""
    success: bool = Field(..., description="Tráº¡ng thÃ¡i")
    message: str = Field(..., description="ThÃ´ng bÃ¡o")
    stats: Dict[str, int] = Field(..., description="Thá»‘ng kÃª {identity: num_embeddings}")
    total_people: int = Field(..., description="Tá»•ng sá»‘ ngÆ°á»i Ä‘Ã£ Ä‘Äƒng kÃ½")
    total_embeddings: int = Field(..., description="Tá»•ng sá»‘ embeddings")
    timestamp: datetime = Field(..., description="Thá»i gian hoÃ n thÃ nh")


# Embedding management schemas
class EmbeddingStats(BaseModel):
    """Thá»‘ng kÃª embeddings database"""
    num_people: int = Field(..., description="Sá»‘ lÆ°á»£ng ngÆ°á»i trong database")
    total_vectors: int = Field(..., description="Tá»•ng sá»‘ vectors")
    timestamp: datetime = Field(..., description="Thá»i gian láº¥y thá»‘ng kÃª")


class RefreshDatabaseRequest(BaseModel):
    """Request Ä‘á»ƒ refresh database"""
    embedding_dir: Optional[str] = Field(None, description="ÄÆ°á»ng dáº«n thÆ° má»¥c embeddings")


class RefreshDatabaseResponse(BaseModel):
    """Response sau khi refresh database"""
    success: bool = Field(..., description="Tráº¡ng thÃ¡i")
    message: str = Field(..., description="ThÃ´ng bÃ¡o")
    num_people: int = Field(..., description="Sá»‘ ngÆ°á»i trong database")
    total_vectors: int = Field(..., description="Tá»•ng sá»‘ vectors")
    timestamp: datetime = Field(..., description="Thá»i gian refresh")
