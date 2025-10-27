# AI-Service Implementation Summary

## ✅ Đã Hoàn Thành

### 1. Configuration (`app/core/config.py`)
- ✅ Added Backend integration settings:
  ```python
  BACKEND_JWT_SECRET: str  # Must match Backend SECRET_KEY
  BACKEND_CALLBACK_SECRET: str  # Must match Backend AI_SERVICE_SECRET
  ```

### 2. JWT Authentication (`app/core/jwt_utils.py`)
- ✅ **verify_websocket_token()**: Verify JWT token từ Backend
  - Decode với `BACKEND_JWT_SECRET`
  - Verify token type = "websocket"
  - Check expiry
  - Extract user_id, session_id, role

- ✅ **verify_user_permission()**: RBAC check
  - Verify session_id match
  - Check user trong allowed_users list

### 3. Session Management (`app/services/session_manager.py`)
- ✅ Updated `SessionData` dataclass:
  ```python
  backend_session_id: int  # Backend session ID
  allowed_users: List[str]  # RBAC user list
  ```

- ✅ Updated `create_session()`:
  - Receive backend_session_id, allowed_users từ request
  - Store trong session data

### 4. Schemas (`app/models/schemas.py`)
- ✅ Updated `SessionCreateRequest`:
  ```python
  backend_session_id: int
  ws_token: str
  allowed_users: List[str]
  ```

### 5. API Endpoints

#### A. POST `/api/v1/sessions` (`app/api/v1/endpoints/sessions.py`)
- ✅ Already implemented
- ✅ Creates session and loads embeddings vào VRAM
- ✅ Returns session_id, status, embeddings_loaded

#### B. WebSocket `/api/v1/sessions/{session_id}/stream` (`app/api/v1/endpoints/frames.py`)
- ✅ **NEW ENDPOINT**: WebSocket cho frame streaming

**Features:**
1. JWT authentication với Backend's secret
2. RBAC permission check
3. Rate limiting (30 FPS)
4. Frame size validation (max 2MB)
5. Real-time detection + recognition + tracking
6. Multi-frame validation
7. Callback to Backend với HMAC signature
8. Real-time updates to Client

**Message Types:**
- `connection_established`: Khi connect thành công
- `frame_processed`: Mỗi frame xử lý xong
- `student_validated`: Khi sinh viên pass validation
- `session_status`: Statistics định kỳ (every 30 frames)
- `error`: Khi có lỗi

### 6. Callback Sender (`app/services/notifier.py`)
- ✅ Updated `send_attendance_update()`:
  - Generate HMAC-SHA256 signature
  - Payload: JSON với separators=(',', ':') (no spaces)
  - Send header `X-AI-Signature`

**HMAC Signature:**
```python
signature = hmac.new(
    BACKEND_CALLBACK_SECRET.encode(),
    json.dumps(payload, separators=(',', ':')).encode(),
    hashlib.sha256
).hexdigest()
```

### 7. Environment Configuration
- ✅ Updated `.env.example` với Backend integration variables

## 🎯 Flow Chi Tiết

### 1. Backend Creates Session
```python
# Backend → AI-Service
POST /api/v1/sessions
{
  "backend_session_id": 50,
  "class_id": "1",
  "student_codes": ["102220347", "102220348", ...],
  "backend_callback_url": "http://backend:8001/api/v1/attendance/webhook/ai-recognition",
  "ws_token": "eyJhbGc...",  # Sample token
  "allowed_users": ["123"]  # teacher user_id
}

# AI-Service Response
{
  "session_id": "uuid-abc-123",
  "class_id": "1",
  "status": "active",
  "created_at": "2025-10-24T10:00:00Z",
  "backend_callback_url": "...",
  "embeddings_loaded": true,
  "total_frames_processed": 0
}
```

### 2. Client Connects WebSocket
```javascript
const ws = new WebSocket(
  `ws://ai-service:8096/api/v1/sessions/uuid-abc-123/stream?token=${jwt_token}`
);

ws.onopen = () => {
  console.log("Connected");
  startSendingFrames();
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  
  switch(data.type) {
    case "frame_processed":
      // Update UI với detections
      break;
    case "student_validated":
      // Show validated student
      break;
  }
};
```

### 3. AI-Service Processes Frames
```python
# WebSocket receives binary frame
frame_data = await websocket.receive_bytes()

# 1. Detect faces
detections = await face_engine.detect_faces(frame_data)

# 2. Recognize với VRAM embeddings
detections = await face_engine.recognize_faces(
    detections,
    frame_data,
    gallery_embeddings=session.gallery_embeddings,  # On GPU
    gallery_labels=session.gallery_labels
)

# 3. Track faces
detections = await face_tracker.update(detections)

# 4. Update recognition history
await face_engine.update_recognition_history(detections, current_time)

# 5. Get validated students (multi-frame)
validated_students = await face_engine.get_validated_students(current_time)

# 6. Send to client
await websocket.send_json({
    "type": "frame_processed",
    "detections": [...],
    "total_faces": len(detections)
})

# 7. Send callback to Backend (if new validated student)
if new_validated_student:
    await send_callback_to_backend(student_data)
```

### 4. AI-Service → Backend Callback
```python
# Generate HMAC signature
payload = {
    "session_id": "uuid-abc-123",
    "class_id": "1",
    "recognized_students": ["102220347"],
    "timestamp": "2025-10-24T10:00:00Z",
    "total_faces_detected": 2
}

payload_str = json.dumps(payload, separators=(',', ':'))
signature = hmac.new(
    BACKEND_CALLBACK_SECRET.encode(),
    payload_str.encode(),
    hashlib.sha256
).hexdigest()

# Send to Backend
response = await http_client.post(
    backend_callback_url,
    json=payload,
    headers={"X-AI-Signature": signature}
)
```

## 🔐 Security

### 1. WebSocket Authentication
- **JWT Token** từ Backend
- **Token Type**: "websocket"
- **Payload**: user_id, session_id, role, exp
- **Verification**: Decode với `BACKEND_JWT_SECRET`
- **RBAC**: Check user trong `allowed_users`

### 2. Callback Security
- **HMAC-SHA256** signature
- **Secret**: `BACKEND_CALLBACK_SECRET` (shared với Backend)
- **Payload**: JSON với separators=(',', ':')
- **Header**: `X-AI-Signature`

### 3. Rate Limiting
- **Max FPS**: 30 frames/second
- **Max Frame Size**: 2MB
- **Validation**: Check trước khi process

### 4. Session Security
- **Session expiry**: Configurable max_duration_minutes
- **Status check**: Only process if status="active"
- **User permission**: Verify trước khi accept WebSocket

## 📊 WebSocket Messages

### 1. Client → Server
```
Binary frames (JPEG/PNG bytes)
```

### 2. Server → Client

**connection_established**
```json
{
  "type": "connection_established",
  "session_id": "uuid",
  "message": "Connected to AI-Service"
}
```

**frame_processed**
```json
{
  "type": "frame_processed",
  "detections": [
    {
      "bbox": [100, 150, 250, 350],
      "track_id": 1,
      "student_id": "102220347",
      "student_name": "Nguyen Van A",
      "confidence": 0.85,
      "is_validated": false
    }
  ],
  "total_faces": 1,
  "timestamp": "2025-10-24T10:00:01.234Z"
}
```

**student_validated**
```json
{
  "type": "student_validated",
  "student": {
    "student_code": "102220347",
    "student_name": "Nguyen Van A",
    "track_id": 1,
    "avg_confidence": 0.85,
    "frame_count": 10,
    "recognition_count": 8,
    "validation_passed_at": "2025-10-24T10:00:03Z"
  }
}
```

**session_status** (every 30 frames)
```json
{
  "type": "session_status",
  "status": "active",
  "stats": {
    "total_frames_processed": 150,
    "total_faces_detected": 180,
    "validated_students": 25
  }
}
```

**error**
```json
{
  "type": "error",
  "message": "Frame too large (max 2MB)"
}
```

## 🧪 Testing

### Test JWT Token Generation (Backend)
```python
# Backend generates token
from app.core.security import create_websocket_token

token = create_websocket_token(
    user_id=123,
    session_id=50,
    role="teacher"
)
print(token)
```

### Test JWT Verification (AI-Service)
```python
# AI-Service verifies token
from app.core.jwt_utils import verify_websocket_token

payload = verify_websocket_token(token)
print(payload)
# Output: {'user_id': 123, 'session_id': 50, 'role': 'teacher', 'type': 'websocket', 'exp': ...}
```

### Test HMAC Signature
```python
# AI-Service generates signature
import hmac, hashlib, json

payload = {
    "session_id": "uuid",
    "recognized_students": ["102220347"]
}

signature = hmac.new(
    "shared-secret".encode(),
    json.dumps(payload, separators=(',', ':')).encode(),
    hashlib.sha256
).hexdigest()

print(signature)
```

### Test WebSocket Connection
```javascript
// Client test
const ws = new WebSocket(
  "ws://localhost:8096/api/v1/sessions/uuid/stream?token=YOUR_JWT_TOKEN"
);

ws.onopen = () => {
  console.log("Connected!");
  
  // Send test frame
  fetch("test-image.jpg")
    .then(res => res.blob())
    .then(blob => ws.send(blob));
};

ws.onmessage = (event) => {
  console.log("Received:", JSON.parse(event.data));
};
```

## 📁 Files Created/Modified

### Created:
- `app/core/jwt_utils.py`
- `AI_SERVICE_IMPLEMENTATION.md` (this file)

### Modified:
- `app/core/config.py`
- `app/models/schemas.py`
- `app/services/session_manager.py`
- `app/services/notifier.py`
- `app/api/v1/endpoints/frames.py`
- `.env.example`

## 🚀 Deployment

### Environment Variables Required
```bash
# AI-Service .env
BACKEND_JWT_SECRET=your-secret-key-here-change-in-production
BACKEND_CALLBACK_SECRET=shared-secret-key-for-hmac-verification

# Must match Backend:
# - BACKEND_JWT_SECRET = Backend SECRET_KEY
# - BACKEND_CALLBACK_SECRET = Backend AI_SERVICE_SECRET
```

### Start AI-Service
```bash
python run.py
# or
uvicorn app.main:app --host 0.0.0.0 --port 8096
```

### Health Check
```bash
curl http://localhost:8096/api/v1/health
```

## 🔍 Common Issues

### 1. WebSocket Token Invalid
```
Error: Invalid token: Signature verification failed
```
**Solution**: Verify `BACKEND_JWT_SECRET` matches Backend's `SECRET_KEY`

### 2. Webhook Signature Mismatch
```
Error: Invalid signature
```
**Solution**: 
- Check `BACKEND_CALLBACK_SECRET` matches Backend's `AI_SERVICE_SECRET`
- Ensure payload JSON has no spaces: `separators=(',', ':')`

### 3. Session Not Found
```
Error: Session not found
```
**Solution**: Ensure Backend created session successfully trước khi connect WebSocket

### 4. Permission Denied
```
Error: Permission denied
```
**Solution**: Check user_id trong JWT token match với `allowed_users` trong session

## ✅ Integration Test Checklist

- [ ] POST /sessions creates session successfully
- [ ] Embeddings loaded vào VRAM
- [ ] WebSocket accepts valid JWT token
- [ ] WebSocket rejects invalid token
- [ ] WebSocket rejects expired token
- [ ] RBAC check works correctly
- [ ] Frame processing works end-to-end
- [ ] Multi-frame validation works
- [ ] Callback sent với correct HMAC signature
- [ ] Backend receives và verifies callback
- [ ] Real-time messages sent to client
- [ ] Rate limiting works (30 FPS)
- [ ] Frame size validation works (max 2MB)
- [ ] Session status updates correctly

## 🎉 Ready for Integration Testing!

AI-Service implementation đã hoàn thành! Bây giờ có thể:
1. ✅ Start Backend
2. ✅ Start AI-Service  
3. ✅ Test end-to-end flow
4. ✅ Implement Frontend WebSocket client
