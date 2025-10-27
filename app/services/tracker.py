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
    student_code: Optional[str]  # Changed from student_id
    confidence: float
    timestamp: datetime


@dataclass
class TrackState:
    """Trạng thái tracking của một object"""
    track_id: int
    last_bbox: List[float]
    last_seen: datetime
    student_code: Optional[str] = None  # Changed from student_id
    confidence_history: List[float] = field(default_factory=list)
    
    # Recognition buffer - lưu lịch sử nhận diện
    recognition_history: deque = field(default_factory=lambda: deque(maxlen=10))
    
    # Confirmed student code sau validation
    confirmed_student_code: Optional[str] = None  # Changed from confirmed_student_id
    confirmed_at: Optional[datetime] = None
    
    # Metrics
    total_recognitions: int = 0
    successful_recognitions: int = 0
    
    def add_recognition(self, student_code: Optional[str], confidence: float, timestamp: datetime):
        """Thêm một bản ghi nhận diện vào history"""
        record = RecognitionRecord(
            student_code=student_code,
            confidence=confidence,
            timestamp=timestamp
        )
        self.recognition_history.append(record)
        self.total_recognitions += 1
        if student_code is not None:
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
                "dominant_student_code": None,  # Changed from dominant_student_id
                "dominant_count": 0
            }
        
        successful = [r for r in recent if r.student_code is not None]
        
        # Tìm student_code xuất hiện nhiều nhất
        student_code_counts = {}
        for r in successful:
            student_code_counts[r.student_code] = student_code_counts.get(r.student_code, 0) + 1
        
        dominant_student_code = max(student_code_counts.items(), key=lambda x: x[1])[0] if student_code_counts else None
        
        # Tính confidence trung bình cho dominant student
        avg_confidence = 0.0
        if dominant_student_code:
            confidences = [r.confidence for r in successful if r.student_code == dominant_student_code]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        
        return {
            "total_frames": len(recent),
            "successful_frames": len(successful),
            "success_rate": len(successful) / len(recent) if recent else 0.0,
            "avg_confidence": avg_confidence,
            "dominant_student_code": dominant_student_code,  # Changed from dominant_student_id
            "dominant_count": student_code_counts.get(dominant_student_code, 0) if dominant_student_code else 0
        }


class FaceTracker(LoggerMixin):
    """
    Multi-object tracker cho faces with IoU-based matching
    Improved tracking algorithm with configurable parameters
    """
    
    def __init__(
        self, 
        max_disappeared: int = 30,
        distance_threshold: int = 200,
        iou_threshold: float = 0.3,
        use_iou: bool = True
    ):
        """
        Initialize Face Tracker
        
        Args:
            max_disappeared: Số giây tối đa track có thể biến mất
            distance_threshold: Ngưỡng khoảng cách center (pixels) 
            iou_threshold: Ngưỡng IoU để match tracks
            use_iou: Sử dụng IoU thay vì distance
        """
        super().__init__()
        self.max_disappeared = max_disappeared
        self.distance_threshold = distance_threshold
        self.iou_threshold = iou_threshold
        self.use_iou = use_iou
        self._tracks: Dict[int, TrackState] = {}
        self._next_track_id = 1
        self.logger.info(
            "FaceTracker initialized",
            use_iou=use_iou,
            iou_threshold=iou_threshold if use_iou else None,
            distance_threshold=distance_threshold if not use_iou else None
        )
    
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
        Tìm track phù hợp hoặc tạo track mới với IoU hoặc distance matching
        """
        best_track_id = None
        best_score = float('-inf') if self.use_iou else float('inf')
        
        detection_bbox = detection.bbox
        
        self.logger.debug(f"Finding track for detection, existing tracks: {len(self._tracks)}")
        
        for track_id, track_state in self._tracks.items():
            track_bbox = track_state.last_bbox
            
            if self.use_iou:
                # Sử dụng IoU (Intersection over Union)
                iou = self._calculate_iou(detection_bbox, track_bbox)
                
                self.logger.debug(
                    f"Track {track_id}: IoU={iou:.3f}"
                )
                
                # IoU cao hơn = match tốt hơn
                if iou > self.iou_threshold and iou > best_score:
                    best_score = iou
                    best_track_id = track_id
            else:
                # Sử dụng khoảng cách center (legacy)
                detection_center = self._get_bbox_center(detection_bbox)
                track_center = self._get_bbox_center(track_bbox)
                distance = self._calculate_distance(detection_center, track_center)
                
                self.logger.debug(
                    f"Track {track_id}: distance={distance:.1f}px"
                )
                
                # Distance thấp hơn = match tốt hơn
                if distance < self.distance_threshold and distance < best_score:
                    best_score = distance
                    best_track_id = track_id
        
        if best_track_id is not None:
            self.logger.debug(
                f"Matched detection to existing track {best_track_id} "
                f"(score={'IoU=%.3f' % best_score if self.use_iou else 'dist=%.1fpx' % best_score})"
            )
            return best_track_id
        else:
            # Tạo track mới
            new_track_id = self._next_track_id
            self._next_track_id += 1
            self.logger.debug(f"Created new track {new_track_id} (no match found)")
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
        
        # Cập nhật student_code nếu có
        if detection.student_code:
            track_state.student_code = detection.student_code
    
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
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Tính IoU (Intersection over Union) giữa 2 bounding boxes
        
        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]
            
        Returns:
            IoU score (0-1)
        """
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Tính intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        
        # Tính union
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - inter_area
        
        # Tránh chia cho 0
        if union_area == 0:
            return 0.0
        
        iou = inter_area / union_area
        return iou
    
    async def get_track_info(self, track_id: int) -> Optional[TrackState]:
        """Lấy thông tin track"""
        return self._tracks.get(track_id)
    
    async def get_active_tracks_count(self) -> int:
        """Lấy số lượng tracks đang hoạt động"""
        return len(self._tracks)
    
    def reset(self):
        """Reset tracker state - để dùng khi bắt đầu session mới"""
        self._tracks.clear()
        self._next_track_id = 1
        self.logger.info("Tracker reset")


# ============= Per-Session Factory =============

def create_face_tracker(
    max_disappeared: int = 30,
    distance_threshold: int = 200,
    iou_threshold: float = 0.3,
    use_iou: bool = True
) -> FaceTracker:
    """
    Factory function to create a new FaceTracker instance for each session
    
    Args:
        max_disappeared: Số giây tối đa track có thể biến mất
        distance_threshold: Ngưỡng khoảng cách center (pixels)
        iou_threshold: Ngưỡng IoU để match tracks
        use_iou: Sử dụng IoU thay vì distance
        
    Returns:
        New FaceTracker instance
    """
    return FaceTracker(
        max_disappeared=max_disappeared,
        distance_threshold=distance_threshold,
        iou_threshold=iou_threshold,
        use_iou=use_iou
    )