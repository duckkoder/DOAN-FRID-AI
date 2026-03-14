"""
Memory Manager - Quản lý bộ nhớ GPU/CPU cho realtime face recognition
Tránh memory leak và crash khi xử lý nhiều frames liên tục
"""
import gc
import os
import threading
from typing import Optional
from contextlib import contextmanager

import torch
import numpy as np

from app.core.logging import LoggerMixin


class MemoryManager(LoggerMixin):
    """
    Quản lý bộ nhớ cho GPU và CPU trong môi trường realtime
    
    Features:
    - Periodic CUDA cache cleanup
    - Garbage collection management
    - Memory usage monitoring
    - Automatic cleanup when threshold exceeded
    """
    
    def __init__(
        self,
        gpu_memory_threshold: float = 0.85,  # Trigger cleanup khi GPU usage > 85%
        cleanup_interval_frames: int = 50,    # Cleanup sau mỗi N frames
        enable_aggressive_gc: bool = True     # Bật GC aggressive mode
    ):
        """
        Args:
            gpu_memory_threshold: Ngưỡng GPU memory để trigger cleanup (0.0-1.0)
            cleanup_interval_frames: Số frames giữa các lần cleanup định kỳ
            enable_aggressive_gc: Bật chế độ GC aggressive
        """
        super().__init__()
        
        self.gpu_memory_threshold = gpu_memory_threshold
        self.cleanup_interval_frames = cleanup_interval_frames
        self.enable_aggressive_gc = enable_aggressive_gc
        
        self._frame_counter = 0
        self._lock = threading.Lock()
        
        # Check CUDA availability
        self.cuda_available = torch.cuda.is_available()
        
        if self.cuda_available:
            self.logger.info(
                "MemoryManager initialized with CUDA support",
                device=torch.cuda.get_device_name(0),
                total_memory_gb=torch.cuda.get_device_properties(0).total_memory / (1024**3),
                threshold=gpu_memory_threshold,
                cleanup_interval=cleanup_interval_frames
            )
        else:
            self.logger.info(
                "MemoryManager initialized (CPU only)",
                enable_aggressive_gc=enable_aggressive_gc
            )
    
    def get_gpu_memory_usage(self) -> dict:
        """
        Lấy thông tin sử dụng GPU memory
        
        Returns:
            Dict với keys: allocated_mb, cached_mb, total_mb, usage_ratio
        """
        if not self.cuda_available:
            return {
                'allocated_mb': 0,
                'cached_mb': 0,
                'total_mb': 0,
                'usage_ratio': 0
            }
        
        allocated = torch.cuda.memory_allocated() / (1024**2)  # MB
        cached = torch.cuda.memory_reserved() / (1024**2)  # MB
        total = torch.cuda.get_device_properties(0).total_memory / (1024**2)  # MB
        
        return {
            'allocated_mb': allocated,
            'cached_mb': cached,
            'total_mb': total,
            'usage_ratio': cached / total if total > 0 else 0
        }
    
    def should_cleanup(self) -> bool:
        """
        Kiểm tra có nên cleanup không
        
        Returns:
            True nếu cần cleanup
        """
        if not self.cuda_available:
            return False
        
        memory_info = self.get_gpu_memory_usage()
        return memory_info['usage_ratio'] > self.gpu_memory_threshold
    
    def cleanup_cuda_cache(self, force: bool = False) -> dict:
        """
        Dọn dẹp CUDA cache
        
        Args:
            force: Force cleanup ngay cả khi chưa đến threshold
            
        Returns:
            Memory info trước và sau cleanup
        """
        if not self.cuda_available:
            return {'status': 'no_cuda'}
        
        before = self.get_gpu_memory_usage()
        
        # Synchronize CUDA operations
        torch.cuda.synchronize()
        
        # Clear cache
        torch.cuda.empty_cache()
        
        after = self.get_gpu_memory_usage()
        
        freed_mb = before['cached_mb'] - after['cached_mb']
        
        if freed_mb > 10:  # Only log if freed > 10MB
            self.logger.info(
                "CUDA cache cleaned",
                freed_mb=f"{freed_mb:.2f}",
                before_cached_mb=f"{before['cached_mb']:.2f}",
                after_cached_mb=f"{after['cached_mb']:.2f}"
            )
        
        return {
            'status': 'cleaned',
            'freed_mb': freed_mb,
            'before': before,
            'after': after
        }
    
    def cleanup_python_gc(self) -> int:
        """
        Chạy Python garbage collector
        
        Returns:
            Số objects được thu hồi
        """
        if self.enable_aggressive_gc:
            # Collect all generations
            collected = gc.collect(generation=2)
        else:
            # Chỉ collect generation 0 (fastest)
            collected = gc.collect(generation=0)
        
        return collected
    
    def periodic_cleanup(self) -> dict:
        """
        Cleanup định kỳ - gọi sau mỗi frame được xử lý
        
        Returns:
            Dict với thông tin cleanup (nếu có)
        """
        with self._lock:
            self._frame_counter += 1
            
            result = {'cleaned': False, 'frame_count': self._frame_counter}
            
            # Cleanup định kỳ
            if self._frame_counter % self.cleanup_interval_frames == 0:
                # Python GC
                gc_collected = self.cleanup_python_gc()
                result['gc_collected'] = gc_collected
                
                # CUDA cache nếu vượt threshold
                if self.should_cleanup():
                    cuda_result = self.cleanup_cuda_cache()
                    result['cuda_cleaned'] = True
                    result['cuda_freed_mb'] = cuda_result.get('freed_mb', 0)
                    result['cleaned'] = True
            
            # Emergency cleanup nếu memory quá cao
            elif self.should_cleanup():
                self.logger.warning("Emergency GPU memory cleanup triggered!")
                self.cleanup_cuda_cache(force=True)
                self.cleanup_python_gc()
                result['emergency_cleanup'] = True
                result['cleaned'] = True
            
            return result
    
    def force_cleanup(self) -> dict:
        """
        Force cleanup tất cả - dùng khi end session hoặc có lỗi
        
        Returns:
            Cleanup result
        """
        self.logger.info("Forcing full memory cleanup")
        
        result = {}
        
        # Python GC - all generations
        result['gc_collected'] = gc.collect(generation=2)
        
        # CUDA
        if self.cuda_available:
            result['cuda'] = self.cleanup_cuda_cache(force=True)
        
        return result
    
    def reset_frame_counter(self):
        """Reset frame counter - gọi khi bắt đầu session mới"""
        with self._lock:
            self._frame_counter = 0
    
    @contextmanager
    def inference_context(self):
        """
        Context manager cho model inference
        Tự động cleanup tensors sau khi inference xong
        
        Usage:
            with memory_manager.inference_context():
                # Model inference here
                result = model(input)
        """
        try:
            yield
        finally:
            # Cleanup intermediate tensors
            if self.cuda_available:
                torch.cuda.synchronize()


# ============================================================
# NUMPY ARRAY OPTIMIZATION UTILITIES
# ============================================================

def optimize_image_array(image: np.ndarray, max_size: int = 640) -> np.ndarray:
    """
    Tối ưu image array để giảm memory footprint
    
    Args:
        image: Input image (H, W, C)
        max_size: Kích thước tối đa cho chiều dài nhất
        
    Returns:
        Optimized image
    """
    h, w = image.shape[:2]
    
    # Resize nếu quá lớn
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        import cv2
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Ensure contiguous memory layout
    if not image.flags['C_CONTIGUOUS']:
        image = np.ascontiguousarray(image)
    
    return image


def compress_face_crop(face_crop: np.ndarray, quality: int = 85) -> bytes:
    """
    Compress face crop to JPEG bytes để giảm memory khi lưu trữ
    
    Args:
        face_crop: Face crop RGB
        quality: JPEG quality (0-100)
        
    Returns:
        JPEG bytes
    """
    import cv2
    
    # Convert RGB to BGR for OpenCV
    bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
    
    # Encode to JPEG
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode('.jpg', bgr, encode_params)
    
    return buffer.tobytes()


def decompress_face_crop(jpeg_bytes: bytes) -> np.ndarray:
    """
    Decompress JPEG bytes back to RGB numpy array
    
    Args:
        jpeg_bytes: JPEG bytes
        
    Returns:
        Face crop RGB numpy array
    """
    import cv2
    
    # Decode JPEG
    nparr = np.frombuffer(jpeg_bytes, np.uint8)
    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert BGR to RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    
    return rgb


# ============================================================
# GLOBAL INSTANCE
# ============================================================

_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Lấy global MemoryManager instance - sử dụng config settings"""
    global _memory_manager
    if _memory_manager is None:
        from app.core.config import settings
        _memory_manager = MemoryManager(
            gpu_memory_threshold=settings.MEMORY_GPU_THRESHOLD,
            cleanup_interval_frames=settings.MEMORY_CLEANUP_INTERVAL,
            enable_aggressive_gc=settings.MEMORY_AGGRESSIVE_GC
        )
    return _memory_manager


def initialize_memory_manager(
    gpu_memory_threshold: Optional[float] = None,
    cleanup_interval_frames: Optional[int] = None,
    enable_aggressive_gc: Optional[bool] = None
) -> MemoryManager:
    """Khởi tạo global MemoryManager với config settings"""
    global _memory_manager
    from app.core.config import settings
    
    _memory_manager = MemoryManager(
        gpu_memory_threshold=gpu_memory_threshold or settings.MEMORY_GPU_THRESHOLD,
        cleanup_interval_frames=cleanup_interval_frames or settings.MEMORY_CLEANUP_INTERVAL,
        enable_aggressive_gc=enable_aggressive_gc if enable_aggressive_gc is not None else settings.MEMORY_AGGRESSIVE_GC
    )
    return _memory_manager


def shutdown_memory_manager():
    """Shutdown và cleanup final"""
    global _memory_manager
    if _memory_manager is not None:
        _memory_manager.force_cleanup()
        _memory_manager = None
