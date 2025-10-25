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
    
    # Import logger trước để dùng trong exception handler
    from app.core.logging import get_logger
    from pathlib import Path
    logger = get_logger(__name__)
    
    # Khởi tạo face detection và recognition services (nếu có config)
    try:
        from app.services.face_detection_service import initialize_face_detection_service
        from app.services.face_recognition_service import initialize_face_recognition_service
        from app.services.embedding_manager import embedding_manager
        from app.services.face_engine import initialize_face_engine
        
        # Initialize detector
        detector_service = None
        if settings.DETECTOR_CHECKPOINT:
            try:
                detector_service = initialize_face_detection_service()
                logger.info("Face detection service initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize face detection service: {e}")
        
        # Initialize recognizer
        recognizer_service = None
        if settings.RECOGNIZER_CHECKPOINT:
            try:
                recognizer_service = initialize_face_recognition_service()
                logger.info("Face recognition service initialized")
                
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
        
        # Initialize face engine
        if detector_service or recognizer_service:
            # Initialize recognition validator
            from app.services.recognition_validator import RecognitionValidator
            validator = RecognitionValidator()
            
            initialize_face_engine(
                detector_service=detector_service,
                recognizer_service=recognizer_service,
                embedding_manager=embedding_manager,
                recognition_validator=validator
            )
            logger.info("Face engine initialized with validator")
        else:
            logger.warning("No face services initialized - running in API-only mode")
        
    except Exception as e:
        logger.error(f"Startup initialization error: {e}")
        logger.warning("Services not initialized - API will run in limited mode")
    
    yield
    
    # Shutdown
    pass


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

