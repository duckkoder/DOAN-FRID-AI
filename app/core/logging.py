"""
Logging configuration với structlog
"""
import logging
import sys
from typing import Optional, Dict, Any
import structlog
from structlog.stdlib import LoggerFactory

from app.core.config import settings


def configure_logging():
    """Cấu hình logging cho ứng dụng"""
    
    # Cấu hình structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Cấu hình standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.LOG_LEVEL.upper())
    )


def get_logger(name: str, **context) -> structlog.stdlib.BoundLogger:
    """
    Tạo logger với context
    
    Args:
        name: Tên module/logger
        **context: Context bổ sung (session_id, request_id, etc.)
    
    Returns:
        Configured logger instance
    """
    logger = structlog.get_logger(name)
    if context:
        logger = logger.bind(**context)
    return logger


class LoggerMixin:
    """Mixin class để thêm logging vào các service classes"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger(self.__class__.__name__)
    
    def get_contextual_logger(self, **context) -> structlog.stdlib.BoundLogger:
        """Tạo logger với context bổ sung"""
        return self.logger.bind(**context)
