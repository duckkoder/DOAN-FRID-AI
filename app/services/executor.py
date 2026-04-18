"""
Thread Pool Executor cho CPU-intensive tasks
Cung cấp non-blocking execution cho model inference

Kiến trúc 3-Pool tách biệt:
- face_pool : Real-time Face AI (Detection / Recognition / Anti-Spoofing)
- rag_pool  : Text Embedding / RAG pipeline (batch, không ưu tiên)
- io_pool   : I/O thuần túy (file, DB nhỏ, HTTP callback)
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, Any, Optional
import os

from app.core.logging import LoggerMixin


# ---------------------------------------------------------------------------
# ExecutorManager - Quản lý nhiều pool riêng biệt
# ---------------------------------------------------------------------------

class ExecutorManager(LoggerMixin):
    """
    Quản lý 3 ThreadPool riêng biệt để tránh xung đột tài nguyên:

    - face_pool  : Dành riêng cho Face Detection / Recognition / Anti-Spoofing
                   (real-time). Không bao giờ bị block bởi tác vụ khác.
    - rag_pool   : Dành riêng cho Text Embedding / RAG pipeline (nặng, batch).
                   Chạy delay cũng được, không ảnh hưởng điểm danh.
    - io_pool    : Dành cho I/O như đọc/ghi file, callback HTTP, query DB nhỏ.
    """

    def __init__(self) -> None:
        super().__init__()
        cpu_count = os.cpu_count() or 4

        # Real-time AI: 3 thread (GPU là bottleneck thực sự, không phải CPU thread)
        self._face_executor = ThreadPoolExecutor(
            max_workers=3,
            thread_name_prefix="face_worker"
        )

        # RAG / Text Embedding: 2 thread, tránh chiếm hết CPU khi embed tài liệu lớn
        self._rag_executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix="rag_worker"
        )

        # I/O: callbacks, file reads, v.v.
        self._io_executor = ThreadPoolExecutor(
            max_workers=max(2, cpu_count - 2),
            thread_name_prefix="io_worker"
        )

        # Cached max_workers for legacy compat
        self.max_workers = 3

        self._loop: Optional[asyncio.AbstractEventLoop] = None

        self.logger.info(
            "ExecutorManager initialized (3-pool)",
            face_workers=3,
            rag_workers=2,
            io_workers=max(2, cpu_count - 2),
        )

    async def initialize(self) -> None:
        """Lấy event loop để sử dụng sau này."""
        self._loop = asyncio.get_event_loop()

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None:
            self._loop = asyncio.get_event_loop()
        return self._loop

    # ── Face Pool ──────────────────────────────────────────────────────────

    async def execute_face(self, func: Callable, *args, **kwargs) -> Any:
        """Chạy tác vụ AI khuôn mặt (Detection / Recognition / Anti-Spoofing)."""
        bound = partial(func, *args, **kwargs)
        return await self._get_loop().run_in_executor(self._face_executor, bound)

    # ── RAG Pool ───────────────────────────────────────────────────────────

    async def execute_rag(self, func: Callable, *args, **kwargs) -> Any:
        """Chạy tác vụ Text Embedding / RAG (nặng, không ưu tiên thời gian thực)."""
        bound = partial(func, *args, **kwargs)
        return await self._get_loop().run_in_executor(self._rag_executor, bound)

    # ── I/O Pool ───────────────────────────────────────────────────────────

    async def execute_io(self, func: Callable, *args, **kwargs) -> Any:
        """Chạy tác vụ I/O (file, DB, HTTP callback)."""
        bound = partial(func, *args, **kwargs)
        return await self._get_loop().run_in_executor(self._io_executor, bound)

    # ── Backward-compat: execute() → face pool ─────────────────────────────

    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Backward-compatible alias cho execute_face().
        Tất cả code cũ gọi executor.execute() sẽ tự động dùng face_pool.
        """
        return await self.execute_face(func, *args, **kwargs)

    async def execute_many(self, func: Callable, items: list, *common_args) -> list:
        """Chạy function cho nhiều items song song trên face_pool."""
        tasks = [self.execute_face(func, item, *common_args) for item in items]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def shutdown(self) -> None:
        self._face_executor.shutdown(wait=True)
        self._rag_executor.shutdown(wait=True)
        self._io_executor.shutdown(wait=True)
        self.logger.info("ExecutorManager shutdown complete")


# ---------------------------------------------------------------------------
# Globals và factory functions
# ---------------------------------------------------------------------------

_executor_manager: Optional[ExecutorManager] = None


def get_executor_manager() -> ExecutorManager:
    """Lấy global ExecutorManager instance."""
    global _executor_manager
    if _executor_manager is None:
        _executor_manager = ExecutorManager()
    return _executor_manager


def initialize_executor_manager() -> ExecutorManager:
    """Khởi tạo (hoặc reset) global ExecutorManager."""
    global _executor_manager
    _executor_manager = ExecutorManager()
    return _executor_manager


def shutdown_executor_manager() -> None:
    """Shutdown global ExecutorManager và giải phóng thread pools."""
    global _executor_manager
    if _executor_manager is not None:
        _executor_manager.shutdown()
        _executor_manager = None


# ── Legacy API (backward compatibility) ─────────────────────────────────────

def get_model_executor() -> ExecutorManager:
    """
    Alias legacy — trả về ExecutorManager.
    .execute() trên đối tượng này sẽ chạy trên face_pool (giống cũ).
    """
    return get_executor_manager()


def initialize_model_executor(max_workers: Optional[int] = None) -> ExecutorManager:
    """Alias legacy cho initialize_executor_manager()."""
    return initialize_executor_manager()


def shutdown_model_executor() -> None:
    """Alias legacy cho shutdown_executor_manager()."""
    shutdown_executor_manager()
