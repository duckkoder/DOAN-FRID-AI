"""
Main FastAPI application
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import configure_logging
from app.api.v1.router import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    configure_logging()
    
    from app.core.logging import get_logger
    from pathlib import Path
    logger = get_logger(__name__)
    
    try:
        from app.services.executor import initialize_model_executor, shutdown_model_executor
        from app.services.face_detection_service import initialize_face_detection_service
        from app.services.face_recognition_service import initialize_face_recognition_service
        from app.services.anti_spoofing_service import initialize_anti_spoofing_service
        from app.services.embedding_manager import embedding_manager
        from app.services.face_engine import initialize_face_engine
        from app.core.memory_manager import initialize_memory_manager, shutdown_memory_manager
        
        # ✅ Initialize memory manager FIRST
        memory_manager = initialize_memory_manager()
        logger.info(
            "MemoryManager initialized",
            gpu_threshold=settings.MEMORY_GPU_THRESHOLD,
            cleanup_interval=settings.MEMORY_CLEANUP_INTERVAL,
            max_faces_per_frame=settings.MEMORY_MAX_FACES_PER_FRAME
        )
        
        # ✅ Initialize model executor for async batch processing
        executor = initialize_model_executor()
        await executor.initialize()
        logger.info(f"ModelExecutor initialized with {executor.max_workers} workers")
        
        # Initialize detector
        detector_service = None
        if settings.DETECTOR_CHECKPOINT:
            try:
                detector_service = initialize_face_detection_service()
                logger.info(f"Face detection service initialized with model: {settings.DETECTOR_CHECKPOINT}")
            except Exception as e:
                logger.warning(f"Failed to initialize face detection service: {e}")
        
        # Initialize recognizer
        recognizer_service = None
        if settings.RECOGNIZER_CHECKPOINT:
            try:
                recognizer_service = initialize_face_recognition_service()
                logger.info(f"Face recognition service initialized with model: {settings.RECOGNIZER_CHECKPOINT}")
                
                # Load embeddings if embedding_dir exists
                if settings.EMBEDDING_DIR:
                    embedding_path = Path(settings.EMBEDDING_DIR)
                    if embedding_path.exists():
                        try:
                            recognizer_service.load_embedding_directory(embedding_path)
                            stats = recognizer_service.get_database_stats()
                            logger.info(
                                "Embeddings loaded on startup",
                                num_people=stats["num_people"],
                                total_vectors=stats["total_vectors"]
                            )
                        except Exception as e:
                            logger.warning(f"Failed to load embeddings: {e}")
                
            except Exception as e:
                logger.warning(f"Failed to initialize face recognition service: {e}")
        
        # ✅ Initialize anti-spoofing service (PATTERN GIỐNG detector/recognizer)
        anti_spoofing_service = None
        if settings.ANTISPOOFING_CHECKPOINT:  # ✅ Check config
            try:
                anti_spoofing_service = initialize_anti_spoofing_service(
                    checkpoint_path=settings.ANTISPOOFING_CHECKPOINT,
                    device=settings.ANTISPOOFING_DEVICE,
                    threshold=settings.ANTISPOOFING_THRESHOLD
                )
                logger.info(f"Anti-spoofing service initialized with model: {settings.ANTISPOOFING_CHECKPOINT}")
            except Exception as e:
                logger.warning(f"Failed to initialize anti-spoofing service: {e}")
                logger.warning("Anti-spoofing will be disabled")
        
        # Initialize face engine
        if detector_service or recognizer_service:
            recognition_validator = None  # Validator is now per-session
            
            initialize_face_engine(
                detector_service=detector_service,
                recognizer_service=recognizer_service,
                embedding_manager=embedding_manager,
                recognition_validator=recognition_validator,
                anti_spoofing_service=anti_spoofing_service  # ✅ Pass service
            )
            logger.info("Face engine initialized (validator will be per-session)")
        else:
            logger.warning("No face services initialized - running in API-only mode")
        
    except Exception as e:
        logger.error(f"Startup initialization error: {e}")
        logger.warning("Services not initialized - API will run in limited mode")
    
    yield
    
    # Shutdown
    from app.core.logging import get_logger
    logger = get_logger(__name__)
    
    # ✅ Shutdown memory manager - cleanup GPU memory
    try:
        from app.core.memory_manager import shutdown_memory_manager
        shutdown_memory_manager()
        logger.info("MemoryManager shutdown complete - GPU memory released")
    except Exception as e:
        logger.warning(f"Error during memory manager shutdown: {e}")
    
    try:
        from app.services.executor import shutdown_model_executor
        shutdown_model_executor()
        logger.info("ModelExecutor shutdown complete")
    except Exception as e:
        logger.warning(f"Error during executor shutdown: {e}")


# Tạo FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI Service cho face recognition và attendance tracking",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Cấu hình origins cụ thể trong production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API v1 router
app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": f"Welcome to {settings.APP_NAME}",
        "version": settings.APP_VERSION,
        "docs": "/docs"
    }

