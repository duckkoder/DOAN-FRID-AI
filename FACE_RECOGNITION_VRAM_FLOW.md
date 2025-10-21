# Face Recognition Flow Implementation - pgvector VRAM Loading

## Architecture Overview

Luồng xử lý face recognition với embeddings loaded vào VRAM:

```
┌─────────────────────────────────────────────────────────────────────┐
│ BƯỚC 1: Start Session (Query 1 lần DUY NHẤT)                        │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Backend (FastAPI)                                                   │
│ POST /api/v1/sessions                                               │
│ {                                                                   │
│   "class_id": "CS101",                                              │
│   "student_codes": ["102220312", "102220347", ...],  // 100 codes  │
│   "backend_callback_url": "http://backend/callback"                │
│ }                                                                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ AI-Service: SessionManager.create_session()                        │
│                                                                     │
│ 1. Create session_id = uuid4()                                     │
│ 2. Call _load_embeddings_from_database(student_codes)              │
│                                                                     │
│    ┌──────────────────────────────────────────────────────────┐   │
│    │ DatabaseService.get_embeddings_by_student_codes()        │   │
│    │                                                          │   │
│    │ Query pgvector:                                          │   │
│    │ SELECT student_code, student_id, embedding               │   │
│    │ FROM face_embeddings                                     │   │
│    │ WHERE student_code = ANY([100 codes])                    │   │
│    │   AND status = 'approved'                                │   │
│    │ ORDER BY student_code, id                                │   │
│    │                                                          │   │
│    │ Returns: 500 embeddings (100 students × 5 avg)          │   │
│    └──────────────────────────────────────────────────────────┘   │
│                                                                     │
│ 3. Call _load_embeddings_to_vram(embeddings_data)                  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ BƯỚC 2: Load vào VRAM (GPU Memory)                                 │
│                                                                     │
│ SessionManager._load_embeddings_to_vram():                          │
│                                                                     │
│ 1. Extract embeddings: List[List[float]] → np.ndarray (500, 512)   │
│ 2. Convert to torch.Tensor                                         │
│ 3. Move to GPU: tensor.cuda()                                      │
│ 4. Save to SessionData:                                             │
│    - gallery_embeddings: Tensor(500, 512) on GPU                   │
│    - gallery_labels: List[str] (500 student_codes)                 │
│    - gallery_student_ids: List[int] (500 student_ids)              │
│                                                                     │
│ Memory on GPU: ~1 MB (500 × 512 × 4 bytes)                         │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ SessionData stored in memory:                                      │
│ {                                                                   │
│   "session_id": "uuid",                                             │
│   "gallery_embeddings": Tensor(500, 512) on cuda:0,                │
│   "gallery_labels": ["102220312", "102220312", ...],               │
│   "gallery_student_ids": [101, 101, 102, ...],                     │
│   "embedding_count": 500,                                           │
│   "embeddings_loaded": True                                        │
│ }                                                                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ BƯỚC 3: Process Frame (Không tốn chi phí mạng)                     │
│                                                                     │
│ POST /api/v1/sessions/{session_id}/frames                          │
│ {                                                                   │
│   "frame_base64": "...",                                            │
│   "timestamp": "2025-10-21T..."                                    │
│ }                                                                   │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ FaceRecognitionService.process_frame()                              │
│                                                                     │
│ 1. Get session_data from SessionManager                             │
│ 2. Detect faces in frame                                            │
│ 3. Extract embeddings for each face: (N_faces, 512)                │
│ 4. Compare with gallery_embeddings in VRAM:                         │
│                                                                     │
│    ┌──────────────────────────────────────────────────────────┐   │
│    │ face_recognizer.identify()                               │   │
│    │                                                          │   │
│    │ query_emb: Tensor(1, 512) on GPU                         │   │
│    │ gallery_emb: Tensor(500, 512) on GPU                     │   │
│    │                                                          │   │
│    │ distances = cosine_distance(query, gallery)             │   │
│    │ # All computation on GPU, no database query!            │   │
│    │                                                          │   │
│    │ best_idx = argmin(distances)                             │   │
│    │ student_code = gallery_labels[best_idx]                  │   │
│    └──────────────────────────────────────────────────────────┘   │
│                                                                     │
│ 5. Return recognized students                                       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│ No database query during frame processing!                         │
│ All comparisons happen in VRAM (GPU memory)                        │
└─────────────────────────────────────────────────────────────────────┘
```

## Implementation Details

### 1. Database Service (Query pgvector)

**File**: `AI-service/app/services/database_service.py`

```python
class DatabaseService:
    def get_embeddings_by_student_codes(
        self,
        student_codes: List[str],
        status: str = "approved"
    ) -> List[Dict[str, Any]]:
        """
        Query embeddings từ pgvector.
        Đây là query DUY NHẤT khi start session.
        
        Returns:
            [
                {
                    'embedding_id': 1,
                    'student_id': 101,
                    'student_code': '102220312',
                    'embedding': [0.1, 0.2, ..., 0.5]  # 512 floats
                },
                ...
            ]
        """
```

**SQL Query**:
```sql
SELECT 
    id as embedding_id,
    student_id,
    student_code,
    embedding
FROM face_embeddings
WHERE student_code = ANY(%s)  -- PostgreSQL array
  AND status = %s
ORDER BY student_code, id
```

### 2. Session Manager (Load to VRAM)

**File**: `AI-service/app/services/session_manager.py`

```python
@dataclass
class SessionData:
    """Session data with embeddings in VRAM"""
    session_id: str
    class_id: str
    
    # Embeddings trong VRAM (GPU memory)
    gallery_embeddings: torch.Tensor  # Shape: (500, 512) on cuda:0
    gallery_labels: List[str]  # ['102220312', '102220312', ...]
    gallery_student_ids: List[int]  # [101, 101, 102, ...]
    embedding_count: int  # 500
    
    student_codes: List[str]  # Original 100 codes
    embeddings_loaded: bool = True


class SessionManager:
    async def create_session(self, request: SessionCreateRequest):
        """
        1. Query embeddings từ database (1 lần)
        2. Load vào VRAM
        3. Store trong _sessions dict
        """
        # Query database
        embeddings_data = await self._load_embeddings_from_database(
            request.student_codes
        )
        
        # Load to VRAM
        await self._load_embeddings_to_vram(session_data, embeddings_data)
    
    async def _load_embeddings_to_vram(
        self,
        session_data: SessionData,
        embeddings_data: List[Dict]
    ):
        """
        Convert embeddings to torch tensor and move to GPU.
        
        Steps:
        1. Extract embeddings: List[List[float]] (500, 512)
        2. Stack to numpy: np.array (500, 512)
        3. Convert to torch: torch.from_numpy()
        4. Move to GPU: tensor.cuda()
        5. Save to session_data
        """
        embeddings_list = [emb['embedding'] for emb in embeddings_data]
        labels_list = [emb['student_code'] for emb in embeddings_data]
        
        # Stack and convert
        embeddings_array = np.stack(embeddings_list)  # (500, 512)
        embeddings_tensor = torch.from_numpy(embeddings_array).float()
        
        # Move to GPU
        if torch.cuda.is_available():
            embeddings_tensor = embeddings_tensor.cuda()
        
        # Save to session
        session_data.gallery_embeddings = embeddings_tensor
        session_data.gallery_labels = labels_list
        session_data.embedding_count = len(embeddings_data)
```

### 3. Face Recognition (Compare in VRAM)

**File**: `AI-service/services/face_recognition_service.py`

```python
class FaceRecognitionService:
    async def process_frame(self, session_id: str, frame: np.ndarray):
        """
        Process frame WITHOUT database query.
        All comparisons in VRAM.
        """
        # Get session data (embeddings already in VRAM)
        session_data = await session_manager.get_session_data(session_id)
        
        if not session_data or not session_data.embeddings_loaded:
            raise ValueError("Session not ready")
        
        # Detect faces
        faces = self.face_detector.detect(frame)
        
        # Extract embeddings for detected faces
        face_embeddings = []
        for face in faces:
            emb = self.face_recognizer.extract_embedding(face.crop)
            face_embeddings.append(emb)
        
        # Compare with gallery (ALL IN VRAM, NO DATABASE!)
        recognized_students = []
        for face_emb in face_embeddings:
            student_code = self._identify_face(
                face_emb,
                session_data.gallery_embeddings,  # On GPU
                session_data.gallery_labels
            )
            recognized_students.append(student_code)
        
        return recognized_students
    
    def _identify_face(
        self,
        query_emb: torch.Tensor,  # (512,) on GPU
        gallery_embs: torch.Tensor,  # (500, 512) on GPU
        gallery_labels: List[str]
    ) -> Optional[str]:
        """
        Identify face by comparing with gallery.
        All computation on GPU!
        """
        # Cosine similarity on GPU
        distances = torch.cdist(
            query_emb.unsqueeze(0),  # (1, 512)
            gallery_embs,  # (500, 512)
            p=2
        )  # Result: (1, 500) on GPU
        
        # Find best match
        best_idx = torch.argmin(distances).item()
        best_distance = distances[0, best_idx].item()
        
        if best_distance < self.threshold:
            return gallery_labels[best_idx]
        
        return None
```

## API Endpoints

### 1. Create Session (Backend → AI-Service)

**Request**:
```bash
POST http://ai-service:8000/api/v1/sessions
Content-Type: application/json

{
  "class_id": "CS101",
  "student_codes": [
    "102220312",
    "102220347",
    "102220398",
    ...  // 100 student codes
  ],
  "backend_callback_url": "http://backend:8080/api/v1/attendance/callback",
  "max_duration_minutes": 60
}
```

**Response**:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "class_id": "CS101",
  "status": "active",
  "created_at": "2025-10-21T10:00:00Z",
  "backend_callback_url": "http://backend:8080/api/v1/attendance/callback",
  "s3_config": {"bucket": "", "key": ""},
  "embeddings_loaded": true,
  "total_frames_processed": 0
}
```

### 2. Process Frame (Frontend → AI-Service)

**Request**:
```bash
POST http://ai-service:8000/api/v1/sessions/{session_id}/frames
Content-Type: application/json

{
  "frame_base64": "iVBORw0KGgo...",
  "timestamp": "2025-10-21T10:00:01Z",
  "client_seq": 1
}
```

**Response**:
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "timestamp": "2025-10-21T10:00:01Z",
  "client_seq": 1,
  "processed_at": "2025-10-21T10:00:01.123Z",
  "detections": [
    {
      "bbox": [100, 100, 200, 200],
      "confidence": 0.95,
      "track_id": 1,
      "student_id": "102220312",
      "recognition_confidence": 0.85
    }
  ],
  "recognized_student_ids": ["102220312"],
  "total_faces": 1,
  "callback_sent": true
}
```

## Performance Characteristics

### Memory Usage

**PostgreSQL (Backend)**:
- 100 students × 5 embeddings × 512 floats × 4 bytes = **~1 MB**
- Index size: **~2 MB** (IVFFlat)
- Total: **~3 MB** per class

**AI-Service VRAM (GPU)**:
- Gallery embeddings: 500 × 512 × 4 bytes = **~1 MB**
- Session metadata: **~10 KB**
- Total per session: **~1 MB**

With 10 concurrent sessions: **~10 MB VRAM**

### Network Traffic

**Session Start (once)**:
- Request: ~5 KB (100 student codes)
- Database query: 1 query returning ~1 MB
- Response: ~1 KB

**Frame Processing (30 FPS)**:
- Request: ~500 KB (1920x1080 JPEG base64)
- Response: ~5 KB (detections)
- **No database query!**

### Latency

**Session Start**:
- Database query: **50-100ms**
- Load to VRAM: **10-20ms**
- Total: **~100ms** (one-time cost)

**Frame Processing**:
- Face detection: **20-50ms**
- Embedding extraction: **10-20ms** per face
- Comparison (in VRAM): **<1ms** for 500 embeddings
- Total: **30-70ms** per frame

**No network latency** for comparison (all in VRAM)!

## Configuration

### .env for AI-Service

```env
# PostgreSQL pgvector
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=attendance_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_password

# GPU settings
RECOGNIZER_DEVICE=cuda
DETECTOR_DEVICE=cuda

# Recognition threshold
RECOGNIZER_THRESHOLD=1.5
```

### .env for Backend

```env
# PostgreSQL
DATABASE_URL=postgresql://postgres:password@localhost:5432/attendance_db

# AI-Service
AI_SERVICE_URL=http://localhost:8000
```

## Testing

### 1. Test Database Connection

```python
from app.services.database_service import get_database_service

db_service = get_database_service()

# Test connection
if db_service.test_connection():
    print("✅ Database connected")

# Get stats
stats = db_service.get_embedding_stats()
print(f"Total students: {stats['total_students']}")
print(f"Total embeddings: {stats['total_embeddings']}")
```

### 2. Test Session Creation

```bash
curl -X POST http://localhost:8000/api/v1/sessions \
  -H "Content-Type: application/json" \
  -d '{
    "class_id": "CS101",
    "student_codes": ["102220312", "102220347"],
    "backend_callback_url": "http://backend/callback"
  }'
```

### 3. Check VRAM Usage

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
```

## Troubleshooting

### 1. Database Connection Error

```
Error: could not connect to server
```

**Solution**:
- Check PostgreSQL is running
- Verify connection string in .env
- Test with: `psql -h localhost -U postgres -d attendance_db`

### 2. pgvector Extension Not Found

```
Error: type "vector" does not exist
```

**Solution**:
```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

### 3. CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solution**:
- Reduce concurrent sessions
- Clear cache: `torch.cuda.empty_cache()`
- Check memory: `nvidia-smi`

### 4. No Embeddings Loaded

```
Warning: No embeddings to load to VRAM
```

**Solution**:
- Check student_codes exist in database
- Verify embeddings have status='approved'
- Run: `SELECT * FROM face_embeddings WHERE student_code IN (...)`

## Next Steps

1. ✅ Setup pgvector in PostgreSQL
2. ✅ Create FaceEmbedding model in Backend
3. ✅ Create DatabaseService in AI-Service
4. ✅ Update SessionManager to load embeddings to VRAM
5. 🔄 Update FaceRecognitionService to use gallery embeddings
6. 🔄 Test end-to-end flow
7. 🔄 Optimize batch processing
8. 🔄 Add monitoring and logging

## References

- pgvector documentation: https://github.com/pgvector/pgvector
- PyTorch CUDA: https://pytorch.org/docs/stable/cuda.html
- psycopg2: https://www.psycopg.org/docs/
