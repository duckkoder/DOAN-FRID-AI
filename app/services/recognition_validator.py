"""
Recognition Validator Service - Xác thực nhận diện để giảm thiểu sai sót
"""
from typing import Optional, Dict, Set
from datetime import datetime, timezone, timedelta

from app.services.tracker import face_tracker, TrackState
from app.core.logging import LoggerMixin
from app.core.config import settings


class RecognitionValidator(LoggerMixin):
    """
    Service xác thực nhận diện dựa trên lịch sử tracking
    
    Cơ chế hoạt động:
    1. Tracking: Mỗi người có một track_id duy nhất khi di chuyển trong khung hình
    2. Recognition Buffer: Lưu lịch sử nhận diện của mỗi track (N frame gần nhất)
    3. Confirmation Threshold: Chỉ xác nhận khi được nhận diện đúng >= M lần trong N frame
    4. Confidence Averaging: Tính điểm tin cậy trung bình phải >= ngưỡng
    5. Debouncing: Tránh gửi thông báo nhiều lần cho cùng một người
    """
    
    def __init__(
        self,
        confirmation_threshold: int = 3,  # Số lần nhận diện tối thiểu
        window_size: int = 5,  # Số frame xem xét
        min_avg_confidence: float = 0.5,  # Confidence trung bình tối thiểu (LOWERED for KNN vote scaling)
        min_success_rate: float = 0.6,  # Tỷ lệ thành công tối thiểu (3/5 = 0.6)
        debounce_seconds: int = 30  # Thời gian debounce (giây)
    ):
        """
        Khởi tạo Recognition Validator
        
        Args:
            confirmation_threshold: Số lần nhận diện đúng tối thiểu để xác nhận
            window_size: Số frame gần nhất để xem xét
            min_avg_confidence: Confidence score trung bình tối thiểu
            min_success_rate: Tỷ lệ nhận diện thành công tối thiểu
            debounce_seconds: Thời gian debounce để tránh gửi callback lặp lại
        """
        super().__init__()
        
        self.confirmation_threshold = confirmation_threshold
        self.window_size = window_size
        self.min_avg_confidence = min_avg_confidence
        self.min_success_rate = min_success_rate
        self.debounce_seconds = debounce_seconds
        
        # Lưu các student_id đã được confirmed (để debouncing)
        self._confirmed_students: Dict[str, datetime] = {}
        
        self.logger.info(
            "RecognitionValidator initialized",
            confirmation_threshold=confirmation_threshold,
            window_size=window_size,
            min_avg_confidence=min_avg_confidence,
            min_success_rate=min_success_rate,
            debounce_seconds=debounce_seconds
        )
    
    async def add_recognition(
        self,
        track_id: int,
        student_id: Optional[str],
        confidence: float,
        timestamp: datetime
    ):
        """
        Thêm một bản ghi nhận diện vào track
        
        Args:
            track_id: ID của track
            student_id: ID sinh viên (None nếu không nhận diện được)
            confidence: Độ tin cậy của recognition
            timestamp: Thời điểm nhận diện
        """
        track_state = await face_tracker.get_track_info(track_id)
        
        if track_state is None:
            self.logger.warning(
                "Track not found when adding recognition",
                track_id=track_id
            )
            return
        
        # Thêm vào recognition history
        track_state.add_recognition(student_id, confidence, timestamp)
    
    async def validate_recognition(
        self,
        track_id: int,
        current_time: datetime
    ) -> Optional[Dict]:
        """
        Xác thực nhận diện cho một track
        
        Args:
            track_id: ID của track cần xác thực
            current_time: Thời điểm hiện tại
            
        Returns:
            Dictionary với thông tin validation nếu pass, None nếu không pass:
            {
                "student_id": str,
                "confidence": float,
                "validation_score": float,
                "stats": Dict
            }
        """
        track_state = await face_tracker.get_track_info(track_id)
        
        if track_state is None:
            return None
        
        # Lấy thống kê nhận diện
        stats = track_state.get_recognition_stats(self.window_size)
        
        dominant_student_id = stats["dominant_student_id"]
        dominant_count = stats["dominant_count"]
        avg_confidence = stats["avg_confidence"]
        success_rate = stats["success_rate"]
        
        # Kiểm tra các điều kiện validation
        
        # 1. Phải có student_id dominant
        if dominant_student_id is None:
            return None
        
        # 2. Số lần nhận diện phải >= confirmation_threshold
        if dominant_count < self.confirmation_threshold:
            return None
        
        # 3. Confidence trung bình phải >= min_avg_confidence
        if avg_confidence < self.min_avg_confidence:
            return None
        
        # 4. Success rate phải >= min_success_rate
        if success_rate < self.min_success_rate:
            return None
        
        # 5. Kiểm tra debouncing - đã confirm student này trong thời gian gần đây chưa?
        if dominant_student_id in self._confirmed_students:
            last_confirmed = self._confirmed_students[dominant_student_id]
            time_since_last = (current_time - last_confirmed).total_seconds()
            
            if time_since_last < self.debounce_seconds:
                return None
        
        # Tất cả điều kiện đều pass - Validation thành công!
        
        # Tính validation score (0-1) dựa trên các metrics
        validation_score = self._calculate_validation_score(
            dominant_count,
            avg_confidence,
            success_rate
        )
        
        # Cập nhật track state
        track_state.confirmed_student_id = dominant_student_id
        track_state.confirmed_at = current_time
        
        # Thêm vào debounce set
        self._confirmed_students[dominant_student_id] = current_time
        
        self.logger.info(
            "Recognition validated successfully",
            track_id=track_id,
            student_id=dominant_student_id,
            validation_score=validation_score,
            avg_confidence=avg_confidence,
            success_rate=success_rate,
            dominant_count=dominant_count
        )
        
        return {
            "student_id": dominant_student_id,
            "confidence": avg_confidence,
            "validation_score": validation_score,
            "stats": stats
        }
    
    def _calculate_validation_score(
        self,
        recognition_count: int,
        avg_confidence: float,
        success_rate: float
    ) -> float:
        """
        Tính validation score từ các metrics
        
        Score càng cao = chất lượng validation càng tốt
        
        Args:
            recognition_count: Số lần nhận diện
            avg_confidence: Confidence trung bình
            success_rate: Tỷ lệ thành công
            
        Returns:
            Score từ 0.0 đến 1.0
        """
        # Normalize recognition count (cap at window_size)
        count_score = min(recognition_count / self.window_size, 1.0)
        
        # Confidence already in [0, 1]
        confidence_score = avg_confidence
        
        # Success rate already in [0, 1]
        success_score = success_rate
        
        # Weighted average (có thể điều chỉnh weights)
        score = (
            0.3 * count_score +
            0.4 * confidence_score +
            0.3 * success_score
        )
        
        return round(score, 3)
    
    async def get_newly_confirmed_students(
        self,
        current_time: datetime
    ) -> Set[str]:
        """
        Lấy danh sách các track đã được validated và chưa gửi callback
        
        Args:
            current_time: Thời điểm hiện tại
            
        Returns:
            Set các student_id đã được validated
        """
        validated_students = set()
        
        # Lấy tất cả active tracks
        active_tracks_count = await face_tracker.get_active_tracks_count()
        
        self.logger.info(
            f"Checking validation for {active_tracks_count} active tracks",
            active_tracks=active_tracks_count
        )
        
        if active_tracks_count == 0:
            return validated_students
        
        # Duyệt qua tất cả tracks và validate
        # (Cần access internal tracks - có thể cần thêm API)
        for track_id, track_state in face_tracker._tracks.items():
            # Log track state
            stats = track_state.get_recognition_stats(self.window_size)
            self.logger.info(
                f"[Track {track_id}] Stats: history_size={len(track_state.recognition_history)}, "
                f"dominant_student={stats['dominant_student_id']}, "
                f"dominant_count={stats['dominant_count']}, "
                f"avg_conf={stats['avg_confidence']:.3f}, "
                f"success_rate={stats['success_rate']:.3f}, "
                f"already_confirmed={track_state.confirmed_student_id}"
            )
            
            # Bỏ qua nếu đã confirmed
            if track_state.confirmed_student_id is not None:
                # Already validated - return it only if not yet added to _confirmed_students (first validation)
                # This ensures we return the student_id on the FIRST frame they were validated
                if track_state.confirmed_student_id not in self._confirmed_students:
                    self.logger.info(
                        f"Found newly confirmed student from track state: {track_state.confirmed_student_id}"
                    )
                    validated_students.add(track_state.confirmed_student_id)
                    # Add to _confirmed_students to prevent returning again
                    self._confirmed_students[track_state.confirmed_student_id] = current_time
                continue
            
            # Thử validate
            validation_result = await self.validate_recognition(track_id, current_time)
            
            if validation_result is not None:
                validated_students.add(validation_result["student_id"])
        
        return validated_students
    
    async def get_validated_student_data(self, student_id: str) -> Optional[Dict]:
        """
        Lấy thông tin chi tiết của sinh viên đã được validated
        
        Args:
            student_id: ID sinh viên
            
        Returns:
            Dictionary với thông tin sinh viên hoặc None
        """
        # Tìm track có confirmed_student_id matching
        for track_id, track_state in face_tracker._tracks.items():
            if track_state.confirmed_student_id == student_id:
                stats = track_state.get_recognition_stats(self.window_size)
                return {
                    "student_id": student_id,
                    "student_name": student_id,  # TODO: Lookup from database if needed
                    "confidence": stats["avg_confidence"],
                    "track_id": track_id,
                    "confirmed_at": track_state.confirmed_at.isoformat() if track_state.confirmed_at else None
                }
        
        # Fallback: return basic info if not found in tracks
        if student_id in self._confirmed_students:
            return {
                "student_id": student_id,
                "student_name": student_id,
                "confidence": 0.0,
                "track_id": None,
                "confirmed_at": self._confirmed_students[student_id].isoformat()
            }
        
        return None
    
    def cleanup_old_confirmations(self, current_time: datetime):
        """
        Dọn dẹp các confirmation cũ khỏi debounce set
        
        Args:
            current_time: Thời điểm hiện tại
        """
        students_to_remove = []
        
        for student_id, confirmed_at in self._confirmed_students.items():
            time_since = (current_time - confirmed_at).total_seconds()
            # Xóa nếu đã quá 2x debounce time
            if time_since > self.debounce_seconds * 2:
                students_to_remove.append(student_id)
        
        for student_id in students_to_remove:
            del self._confirmed_students[student_id]
    
    def reset_debounce(self, student_id: Optional[str] = None):
        """
        Reset debounce cho một hoặc tất cả sinh viên
        
        Args:
            student_id: ID sinh viên cần reset, None để reset tất cả
        """
        if student_id is None:
            self._confirmed_students.clear()
            self.logger.info("Reset all debounce entries")
        elif student_id in self._confirmed_students:
            del self._confirmed_students[student_id]
            self.logger.info("Reset debounce for student", student_id=student_id)
    
    def get_stats(self) -> Dict:
        """Lấy thống kê validator"""
        return {
            "confirmation_threshold": self.confirmation_threshold,
            "window_size": self.window_size,
            "min_avg_confidence": self.min_avg_confidence,
            "min_success_rate": self.min_success_rate,
            "debounce_seconds": self.debounce_seconds,
            "confirmed_students_count": len(self._confirmed_students)
        }


# Global validator instance
_recognition_validator: Optional[RecognitionValidator] = None


def get_recognition_validator() -> RecognitionValidator:
    """
    Lấy singleton instance của RecognitionValidator
    
    Returns:
        RecognitionValidator instance
    """
    global _recognition_validator
    
    if _recognition_validator is None:
        _recognition_validator = RecognitionValidator(
            confirmation_threshold=settings.RECOGNITION_CONFIRMATION_THRESHOLD,
            window_size=settings.RECOGNITION_WINDOW_SIZE,
            min_avg_confidence=settings.RECOGNITION_MIN_AVG_CONFIDENCE,
            min_success_rate=settings.RECOGNITION_MIN_SUCCESS_RATE,
            debounce_seconds=settings.RECOGNITION_DEBOUNCE_SECONDS
        )
    
    return _recognition_validator


def initialize_recognition_validator(**kwargs) -> RecognitionValidator:
    """
    Khởi tạo hoặc cập nhật recognition validator
    
    Args:
        **kwargs: Các tham số khởi tạo
        
    Returns:
        RecognitionValidator instance
    """
    global _recognition_validator
    _recognition_validator = RecognitionValidator(**kwargs)
    return _recognition_validator
