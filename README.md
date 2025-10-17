# AI-Service

AI Service skeleton cho face recognition và attendance tracking. Đây là khung mã nguồn (skeleton) với các interface và stub implementations.

## Tính năng

- ✅ Quản lý sessions in-memory
- ✅ JWT authentication
- ✅ Face detection/recognition (stub)
- ✅ Face tracking (stub)
- ✅ Backend callback notifications (stub)
- ✅ S3/MinIO embeddings loading (stub)
- ✅ Structured logging
- ✅ OpenAPI documentation

## Cài đặt

1. Clone repository
2. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

3. Tạo file `.env` từ `.env.example`:
```bash
cp .env.example .env
```

4. Chỉnh sửa cấu hình trong `.env` nếu cần

## Chạy ứng dụng

### Development
```bash
# Sử dụng script run.py (khuyến nghị)
python run.py

# Hoặc chạy trực tiếp với uvicorn
uvicorn app.main:app --reload
```

### Production
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Documentation

Sau khi chạy ứng dụng, truy cập:
- OpenAPI docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/healthz
```

### Tạo Session
```bash
curl -X POST http://localhost:8000/api_ai/v1/sessions \
  -H 'Content-Type: application/json' \
  -d '{
    "class_id": "SE101",
    "backend_callback_url": "http://backend-internal/api/v1/attendance/update",
    "s3": {
      "bucket": "embeddings",
      "key": "classes/SE101.npz"
    }
  }'
```

### Lấy thông tin Session
```bash
curl http://localhost:8000/api_ai/v1/sessions/{SESSION_ID}
```

### Xóa Session
```bash
curl -X DELETE http://localhost:8000/api_ai/v1/sessions/{SESSION_ID}
```

### Xử lý Frame (cần JWT)
```bash
# Tạo test JWT token
python generate_token.py

# Sử dụng token để gọi API
curl -X POST http://localhost:8000/api_ai/v1/sessions/{SESSION_ID}/frames \
  -H 'Authorization: Bearer {JWT_TOKEN}' \
  -H 'Content-Type: application/json' \
  -d '{
    "timestamp": "2025-01-01T08:00:00Z",
    "client_seq": 1
  }'
```

## Cấu trúc thư mục

```
app/
├── api/                    # API endpoints
│   └── v1/                # API version 1
│       ├── router.py      # Router tập trung
│       └── endpoints/     # Endpoint modules
│           ├── health.py  # Health check
│           ├── sessions.py # Session management
│           └── frames.py  # Frame processing
├── core/                  # Core utilities
│   ├── config.py          # Configuration
│   ├── security.py        # JWT authentication
│   └── logging.py         # Logging setup
├── models/                # Pydantic models
│   └── schemas.py         # Request/response schemas
├── services/              # Business logic
│   ├── session_manager.py # Session management
│   ├── face_engine.py     # Face detection/recognition (stub)
│   ├── tracker.py         # Face tracking (stub)
│   ├── notifier.py        # Backend notifications (stub)
│   └── storage.py         # S3/MinIO client (stub)
└── main.py               # FastAPI application
```

## TODO - Cần implement thật

### Face Engine (`app/services/face_engine.py`)
- [ ] Face detection với YOLO/RetinaFace
- [ ] Face recognition với ArcFace/FaceNet
- [ ] Face embedding extraction

### Tracker (`app/services/tracker.py`)
- [ ] Multi-object tracking với DeepSORT/ByteTrack
- [ ] Kalman filter cho prediction
- [ ] IoU matching algorithm

### Storage (`app/services/storage.py`)
- [ ] S3/MinIO client với boto3/aioboto3
- [ ] Embeddings loading/saving
- [ ] Error handling và retry logic

### Notifier (`app/services/notifier.py`)
- [ ] HTTP client với retry và circuit breaker
- [ ] Background task processing
- [ ] Dead letter queue cho failed callbacks

## Docker

### Build image
```bash
docker build -f docker/Dockerfile -t ai-service .
```

### Run với docker-compose
```bash
docker-compose -f docker/docker-compose.yml up
```

## Development

### Linting
```bash
# TODO: Thêm pre-commit hooks
```

### Testing
```bash
# TODO: Thêm pytest tests
```

## License

MIT License
