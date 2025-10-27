"""
Recognition Validator Service - Xác thực nhận diện để giảm thiểu sai sót
"""
from typing import Optional, Dict, Set
from datetime import datetime, timezone, timedelta

from app.services.tracker import FaceTracker, TrackState
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
    6. Auto-adjustment: Tự động điều chỉnh ngưỡng theo frame rate
    """
    
    def __init__(
        self,
        face_tracker: 'FaceTracker',  # Per-session tracker instance
        confirmation_threshold: int = 3,  # Số lần nhận diện tối thiểu
        window_size: int = 5,  # Số frame xem xét
        min_avg_confidence: float = 0.5,  # Confidence trung bình tối thiểu
        min_success_rate: float = 0.6,  # Tỷ lệ thành công tối thiểu
        debounce_seconds: int = 30,  # Thời gian debounce (giây)
        auto_adjust_to_fps: bool = True,  # Tự động điều chỉnh theo FPS
        target_fps: float = 5.0  # FPS mục tiêu cho điều chỉnh
    ):
        """
        Khởi tạo Recognition Validator
        
        Args:
            face_tracker: Instance của FaceTracker (per-session)
            confirmation_threshold: Số lần nhận diện đúng tối thiểu để xác nhận
            window_size: Số frame gần nhất để xem xét
            min_avg_confidence: Confidence score trung bình tối thiểu
            min_success_rate: Tỷ lệ nhận diện thành công tối thiểu
            debounce_seconds: Thời gian debounce để tránh gửi callback lặp lại
            auto_adjust_to_fps: Tự động điều chỉnh ngưỡng theo FPS thực tế
            target_fps: FPS mục tiêu (5 FPS = 1 giây cần 5 frames)
        """
        super().__init__()
        
        self.face_tracker = face_tracker  # Per-session tracker
        self.confirmation_threshold = confirmation_threshold
        self.window_size = window_size
        self.min_avg_confidence = min_avg_confidence
        self.min_success_rate = min_success_rate
        self.debounce_seconds = debounce_seconds
        self.auto_adjust_to_fps = auto_adjust_to_fps
        self.target_fps = target_fps
        
        # Lưu các student_code đã được confirmed (để debouncing)
        self._confirmed_students: Dict[str, datetime] = {}
        
        # Frame rate tracking để auto-adjust
        self._frame_timestamps: list = []
        self._current_fps: float = target_fps
        
        self.logger.info(
            "RecognitionValidator initialized",
            confirmation_threshold=confirmation_threshold,
            window_size=window_size,
            min_avg_confidence=min_avg_confidence,
            min_success_rate=min_success_rate,
            debounce_seconds=debounce_seconds,
            auto_adjust_to_fps=auto_adjust_to_fps,
            target_fps=target_fps
        )
    
    def reset(self):
        """
        Reset validator state - Dùng khi bắt đầu session mới
        
        Clears:
        - Confirmed students history
        - FPS tracking data
        """
        self._confirmed_students.clear()
        self._frame_timestamps.clear()
        self._current_fps = self.target_fps
        self.logger.info("RecognitionValidator reset for new session")
    
    async def add_recognition(
        self,
        track_id: int,
        student_code: Optional[str],  # Changed from student_id to student_code
        confidence: float,
        timestamp: datetime
    ):
        """
        Thêm một bản ghi nhận diện vào track
        
        Args:
            track_id: ID của track
            student_code: Mã sinh viên (student_code, None nếu không nhận diện được)
            confidence: Độ tin cậy của recognition
            timestamp: Thời điểm nhận diện
        """
        # Update FPS tracking
        if self.auto_adjust_to_fps:
            self._update_fps_tracking(timestamp)
        
        track_state = await self.face_tracker.get_track_info(track_id)
        
        if track_state is None:
            self.logger.warning(
                "Track not found when adding recognition",
                track_id=track_id
            )
            return
        
        # Thêm vào recognition history
        track_state.add_recognition(student_code, confidence, timestamp)
    
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
        track_state = await self.face_tracker.get_track_info(track_id)
        
        if track_state is None:
            return None
        
        # Lấy thống kê nhận diện
        stats = track_state.get_recognition_stats(self.window_size)
        
        dominant_student_code = stats["dominant_student_code"]  # Changed from dominant_student_id
        dominant_count = stats["dominant_count"]
        avg_confidence = stats["avg_confidence"]
        success_rate = stats["success_rate"]
        
        # Kiểm tra các điều kiện validation
        
        # 1. Phải có student_code dominant
        if dominant_student_code is None:
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
        if dominant_student_code in self._confirmed_students:
            last_confirmed = self._confirmed_students[dominant_student_code]
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
        track_state.confirmed_student_code = dominant_student_code  # Changed from confirmed_student_id
        track_state.confirmed_at = current_time
        
        # Thêm vào debounce set
        self._confirmed_students[dominant_student_code] = current_time
        
        self.logger.info(
            "Recognition validated successfully",
            track_id=track_id,
            student_code=dominant_student_code,  # Changed from student_id
            validation_score=validation_score,
            avg_confidence=avg_confidence,
            success_rate=success_rate,
            dominant_count=dominant_count
        )
        
        return {
            "student_code": dominant_student_code,  # Changed from student_id
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
        active_tracks_count = await self.face_tracker.get_active_tracks_count()
        
        self.logger.info(
            f"Checking validation for {active_tracks_count} active tracks",
            active_tracks=active_tracks_count
        )
        
        if active_tracks_count == 0:
            return validated_students
        
        # Duyệt qua tất cả tracks và validate
        # (Cần access internal tracks - có thể cần thêm API)
        for track_id, track_state in self.face_tracker._tracks.items():
            # Log track state
            stats = track_state.get_recognition_stats(self.window_size)
            self.logger.info(
                f"[Track {track_id}] Stats: history_size={len(track_state.recognition_history)}, "
                f"dominant_student={stats['dominant_student_code']}, "  # Changed from dominant_student_id
                f"dominant_count={stats['dominant_count']}, "
                f"avg_conf={stats['avg_confidence']:.3f}, "
                f"success_rate={stats['success_rate']:.3f}, "
                f"already_confirmed={track_state.confirmed_student_code}"  # Changed from confirmed_student_id
            )
            
            # Bỏ qua nếu đã confirmed
            if track_state.confirmed_student_code is not None:  # Changed from confirmed_student_id
                # Already validated - return it only if not yet added to _confirmed_students (first validation)
                # This ensures we return the student_code on the FIRST frame they were validated
                if track_state.confirmed_student_code not in self._confirmed_students:
                    self.logger.info(
                        f"Found newly confirmed student from track state: {track_state.confirmed_student_code}"
                    )
                    validated_students.add(track_state.confirmed_student_code)
                    # Add to _confirmed_students to prevent returning again
                    self._confirmed_students[track_state.confirmed_student_code] = current_time
                continue
            
            # Thử validate
            validation_result = await self.validate_recognition(track_id, current_time)
            
            if validation_result is not None:
                validated_students.add(validation_result["student_code"])  # Changed from student_id
        
        return validated_students
    
    async def get_validated_student_data(self, student_code: str) -> Optional[Dict]:
        """
        Lấy thông tin chi tiết của sinh viên đã được validated
        
        Args:
            student_code: Mã sinh viên
            
        Returns:
            Dictionary với thông tin sinh viên hoặc None
        """
        # Tìm track có confirmed_student_code matching
        for track_id, track_state in self.face_tracker._tracks.items():
            if track_state.confirmed_student_code == student_code:  # Changed from confirmed_student_id
                stats = track_state.get_recognition_stats(self.window_size)
                return {
                    "student_code": student_code,  # Changed from student_id
                    "student_name": student_code,  # TODO: Lookup from database if needed
                    "confidence": stats["avg_confidence"],
                    "track_id": track_id,
                    "confirmed_at": track_state.confirmed_at.isoformat() if track_state.confirmed_at else None
                }
        
        # Fallback: return basic info if not found in tracks
        if student_code in self._confirmed_students:
            return {
                "student_code": student_code,  # Changed from student_id
                "student_name": student_code,
                "confidence": 0.0,
                "track_id": None,
                "confirmed_at": self._confirmed_students[student_code].isoformat()
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
    
    def _update_fps_tracking(self, timestamp: datetime):
        """
        Update FPS tracking and adjust thresholds if needed
        
        Args:
            timestamp: Current frame timestamp
        """
        self._frame_timestamps.append(timestamp)
        
        # Keep only last 30 frames for FPS calculation
        if len(self._frame_timestamps) > 30:
            self._frame_timestamps = self._frame_timestamps[-30:]
        
        # Calculate FPS if we have enough samples
        if len(self._frame_timestamps) >= 10:
            time_span = (self._frame_timestamps[-1] - self._frame_timestamps[0]).total_seconds()
            if time_span > 0:
                self._current_fps = (len(self._frame_timestamps) - 1) / time_span
                
                # Adjust thresholds based on actual FPS vs target FPS
                fps_ratio = self._current_fps / self.target_fps
                
                # Adjust window_size and confirmation_threshold proportionally
                # e.g., if FPS is 10 (2x target), we need 2x more frames
                adjusted_window = int(self.window_size * fps_ratio)
                adjusted_confirmation = int(self.confirmation_threshold * fps_ratio)
                
                # Apply adjustments if they differ significantly
                if abs(adjusted_window - self.window_size) > 1:
                    old_window = self.window_size
                    self.window_size = max(3, min(adjusted_window, 20))  # Clamp between 3-20
                    self.logger.info(
                        f"Adjusted window_size based on FPS",
                        current_fps=round(self._current_fps, 2),
                        target_fps=self.target_fps,
                        old_window=old_window,
                        new_window=self.window_size
                    )
                
                if abs(adjusted_confirmation - self.confirmation_threshold) > 1:
                    old_confirmation = self.confirmation_threshold
                    self.confirmation_threshold = max(2, min(adjusted_confirmation, 10))  # Clamp between 2-10
                    self.logger.info(
                        f"Adjusted confirmation_threshold based on FPS",
                        current_fps=round(self._current_fps, 2),
                        target_fps=self.target_fps,
                        old_confirmation=old_confirmation,
                        new_confirmation=self.confirmation_threshold
                    )
    
    def reset_debounce(self, student_code: Optional[str] = None):
        """
        Reset debounce cho một hoặc tất cả sinh viên
        
        Args:
            student_code: Mã sinh viên cần reset, None để reset tất cả
        """
        if student_code is None:
            self._confirmed_students.clear()
            self.logger.info("Reset all debounce entries")
        elif student_code in self._confirmed_students:
            del self._confirmed_students[student_code]
            self.logger.info("Reset debounce for student", student_code=student_code)
    
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


# ============= Per-Session Factory =============

def create_recognition_validator(
    face_tracker: FaceTracker,
    confirmation_threshold: int = 3,
    window_size: int = 5,
    min_avg_confidence: float = 0.5,
    min_success_rate: float = 0.6,
    debounce_seconds: int = 30,
    auto_adjust_to_fps: bool = True,
    target_fps: float = 5.0
) -> RecognitionValidator:
    """
    Factory function to create a new RecognitionValidator instance for each session
    
    Args:
        face_tracker: FaceTracker instance for this session
        confirmation_threshold: Số lần nhận diện tối thiểu
        window_size: Số frame xem xét
        min_avg_confidence: Confidence trung bình tối thiểu
        min_success_rate: Tỷ lệ thành công tối thiểu
        debounce_seconds: Thời gian debounce (giây)
        auto_adjust_to_fps: Tự động điều chỉnh theo FPS
        target_fps: FPS mục tiêu
        
    Returns:
        New RecognitionValidator instance
    """
    return RecognitionValidator(
        face_tracker=face_tracker,
        confirmation_threshold=confirmation_threshold,
        window_size=window_size,
        min_avg_confidence=min_avg_confidence,
        min_success_rate=min_success_rate,
        debounce_seconds=debounce_seconds,
        auto_adjust_to_fps=auto_adjust_to_fps,
        target_fps=target_fps
    )
