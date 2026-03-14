"""
Thread Pool Executor cho CPU-intensive tasks
Cung cấp non-blocking execution cho model inference
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, Any, Optional
import os

from app.core.logging import LoggerMixin


class ModelExecutor(LoggerMixin):
    """
    Executor để chạy model inference trong thread pool
    Tránh block event loop chính của FastAPI
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        """
        Args:
            max_workers: Số thread tối đa. Mặc định = số CPU cores
        """
        super().__init__()
        
        # Mặc định: số CPU cores, hoặc 4 nếu không xác định được
        self.max_workers = max_workers or min(32, (os.cpu_count() or 4) + 4)
        
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="model_worker"
        )
        
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        self.logger.info(
            "ModelExecutor initialized",
            max_workers=self.max_workers
        )
    
    async def initialize(self):
        """Lấy event loop để sử dụng sau này"""
        self._loop = asyncio.get_event_loop()
    
    async def execute(self, func: Callable, *args, **kwargs) -> Any:
        """
        Chạy function đồng bộ trong thread pool một cách bất đồng bộ
        
        Args:
            func: Function đồng bộ cần chạy
            *args, **kwargs: Arguments cho function
            
        Returns:
            Kết quả của function
        """
        if self._loop is None:
            self._loop = asyncio.get_event_loop()
        
        # Sử dụng partial để bind arguments
        bound_func = partial(func, *args, **kwargs)
        
        # Chạy trong thread pool
        return await self._loop.run_in_executor(self._executor, bound_func)
    
    async def execute_many(self, func: Callable, items: list, *common_args) -> list:
        """
        Chạy function cho nhiều items song song
        
        Args:
            func: Function đồng bộ cần chạy
            items: List các item để xử lý
            *common_args: Các arguments chung cho tất cả items
            
        Returns:
            List kết quả theo thứ tự items
        """
        tasks = [
            self.execute(func, item, *common_args) 
            for item in items
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def shutdown(self):
        """Shutdown executor khi ứng dụng dừng"""
        self._executor.shutdown(wait=True)
        self.logger.info("ModelExecutor shutdown complete")


# Global executor instance
_model_executor: Optional[ModelExecutor] = None


def get_model_executor() -> ModelExecutor:
    """Lấy global ModelExecutor instance"""
    global _model_executor
    if _model_executor is None:
        _model_executor = ModelExecutor()
    return _model_executor


def initialize_model_executor(max_workers: Optional[int] = None) -> ModelExecutor:
    """Khởi tạo global ModelExecutor"""
    global _model_executor
    _model_executor = ModelExecutor(max_workers=max_workers)
    return _model_executor


def shutdown_model_executor():
    """Shutdown global executor"""
    global _model_executor
    if _model_executor is not None:
        _model_executor.shutdown()
        _model_executor = None
