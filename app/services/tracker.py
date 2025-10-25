"""
Object Tracker - Interface và stub cho face tracking
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import deque

from app.models.schemas import Detection
from app.core.logging import LoggerMixin


@dataclass
class RecognitionRecord:
    """Bản ghi nhận diện cho một frame"""
    student_id: Optional[str]
    confidence: float
    timestamp: datetime


@dataclass
class TrackState:
    """Trạng thái tracking của một object"""
    track_id: int
    last_bbox: List[float]
    last_seen: datetime
    student_id: Optional[str] = None
    confidence_history: List[float] = field(default_factory=list)
    
    # Recognition buffer - lưu lịch sử nhận diện
    recognition_history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    # Confirmed student ID sau validation
    confirmed_student_id: Optional[str] = None
    confirmed_at: Optional[datetime] = None
    
    # Metrics
    total_recognitions: int = 0
    successful_recognitions: int = 0
    
    def add_recognition(self, student_id: Optional[str], confidence: float, timestamp: datetime):
        """Thêm một bản ghi nhận diện vào history"""
        record = RecognitionRecord(
            student_id=student_id,
            confidence=confidence,
            timestamp=timestamp
        )
        self.recognition_history.append(record)
        self.total_recognitions += 1
        if student_id is not None:
            self.successful_recognitions += 1
    
    def get_recent_recognitions(self, window_size: int = 5) -> List[RecognitionRecord]:
        """Lấy N bản ghi nhận diện gần nhất"""
        history_list = list(self.recognition_history)
        return history_list[-window_size:] if history_list else []
    
    def get_recognition_stats(self, window_size: int = 5) -> Dict:
        """Tính toán thống kê nhận diện trong window"""
        recent = self.get_recent_recognitions(window_size)
        
        if not recent:
            return {
                "total_frames": 0,
                "successful_frames": 0,
                "success_rate": 0.0,
                "avg_confidence": 0.0,
                "dominant_student_id": None
            }
        
        successful = [r for r in recent if r.student_id is not None]
        
        # Tìm student_id xuất hiện nhiều nhất
        student_id_counts = {}
        for r in successful:
            student_id_counts[r.student_id] = student_id_counts.get(r.student_id, 0) + 1
        
        dominant_student_id = max(student_id_counts.items(), key=lambda x: x[1])[0] if student_id_counts else None
        
        # Tính confidence trung bình cho dominant student
        avg_confidence = 0.0
        if dominant_student_id:
            confidences = [r.confidence for r in successful if r.student_id == dominant_student_id]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            "total_frames": len(recent),
            "successful_frames": len(successful),
            "success_rate": len(successful) / len(recent) if recent else 0.0,
            "avg_confidence": avg_confidence,
            "dominant_student_id": dominant_student_id,
            "dominant_count": student_id_counts.get(dominant_student_id, 0) if dominant_student_id else 0
        }


class FaceTracker(LoggerMixin):
    """
    Multi-object tracker cho faces
    TODO: Thay thế bằng implementation thật (DeepSORT, ByteTrack, etc.)
    """
    
    def __init__(self, max_disappeared: int = 30):
        super().__init__()
        self.max_disappeared = max_disappeared
        self._tracks: Dict[int, TrackState] = {}
        self._next_track_id = 1
        self.logger.info("FaceTracker initialized (STUB)")
    
    async def update(self, detections: List[Detection]) -> List[Detection]:
        """
        Cập nhật tracker với detections mới
        
        Args:
            detections: Danh sách detections từ face engine
            
        Returns:
            Danh sách detections với track_id được gán
            
        TODO: Implement thật với tracking algorithm
        """
        current_time = datetime.now(timezone.utc)
        
        # Stub: Gán track_id đơn giản
        for detection in detections:
            if detection.track_id is None:
                # Tìm track gần nhất hoặc tạo mới
                track_id = self._find_or_create_track(detection, current_time)
                detection.track_id = track_id
            
            # Cập nhật track state
            self._update_track_state(detection, current_time)
        
        # Cleanup tracks cũ
        await self._cleanup_old_tracks(current_time)
        
        self.logger.debug(f"Updated tracker with {len(detections)} detections (STUB)")
        return detections
    
    def _find_or_create_track(self, detection: Detection, current_time: datetime) -> int:
        """
        Tìm track phù hợp hoặc tạo track mới
        
        TODO: Implement với IoU matching, Kalman filter, etc.
        """
        # Stub: Tìm track gần nhất dựa trên bbox
        best_track_id = None
        min_distance = float('inf')
        
        detection_center = self._get_bbox_center(detection.bbox)
        
        self.logger.info(f"Finding track for detection at {detection_center}, existing tracks: {len(self._tracks)}")
        
        for track_id, track_state in self._tracks.items():
            track_center = self._get_bbox_center(track_state.last_bbox)
            distance = self._calculate_distance(detection_center, track_center)
            
            self.logger.info(  # ✅ Changed from debug to info
                f"Checking track {track_id}: distance={distance:.1f}px, "
                f"detection_center={detection_center}, track_center={track_center}"
            )
            
            if distance < min_distance and distance < 200:  # ✅ Tăng threshold lên 200px
                min_distance = distance
                best_track_id = track_id
        
        if best_track_id is not None:
            self.logger.info(f"Matched detection to existing track {best_track_id} (distance={min_distance:.1f}px)")
            return best_track_id
        else:
            # Tạo track mới
            new_track_id = self._next_track_id
            self._next_track_id += 1
            self.logger.info(f"Created new track {new_track_id} (no match found, {len(self._tracks)} existing tracks)")
            return new_track_id
    
    def _update_track_state(self, detection: Detection, current_time: datetime):
        """Cập nhật trạng thái track"""
        track_id = detection.track_id
        
        if track_id not in self._tracks:
            self._tracks[track_id] = TrackState(
                track_id=track_id,
                last_bbox=detection.bbox,
                last_seen=current_time
            )
        
        track_state = self._tracks[track_id]
        track_state.last_bbox = detection.bbox
        track_state.last_seen = current_time
        track_state.confidence_history.append(detection.confidence)
        
        # Giữ lại 10 confidence scores gần nhất
        if len(track_state.confidence_history) > 10:
            track_state.confidence_history = track_state.confidence_history[-10:]
        
        # Cập nhật student_id nếu có
        if detection.student_id:
            track_state.student_id = detection.student_id
    
    async def _cleanup_old_tracks(self, current_time: datetime):
        """Dọn dẹp tracks cũ không còn xuất hiện"""
        tracks_to_remove = []
        
        for track_id, track_state in self._tracks.items():
            time_diff = (current_time - track_state.last_seen).total_seconds()
            if time_diff > self.max_disappeared:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self._tracks[track_id]
            self.logger.debug(f"Removed old track {track_id}")
    
    def _get_bbox_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Tính center của bounding box"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        return center_x, center_y
    
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Tính khoảng cách Euclidean"""
        x1, y1 = point1
        x2, y2 = point2
        return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    
    async def get_track_info(self, track_id: int) -> Optional[TrackState]:
        """Lấy thông tin track"""
        return self._tracks.get(track_id)
    
    async def get_active_tracks_count(self) -> int:
        """Lấy số lượng tracks đang hoạt động"""
        return len(self._tracks)


# Global tracker instance
face_tracker = FaceTracker()
