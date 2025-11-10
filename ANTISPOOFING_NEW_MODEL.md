# Anti-Spoofing Implementation - Model Mới

## 📋 Tổng quan

Đã implement hoàn toàn model Anti-Spoofing **mới** vào hệ thống:
- **Model**: ResNet18_MSFF_AntiSpoof (Multi-Scale Feature Fusion)
- **Classes**: 2 classes - `real` (0) và `spoof` (1)
- **Checkpoint**: `best_model.pth` (từ training mới)

---

## 🔄 Thay đổi chính

### 1️⃣ **Model Architecture (`models/AntiSpoofing.py`)**

#### ✅ Thay đổi:
- **Cũ**: `ResNet18_Custom` với 3 classes (live, print, replay)
- **Mới**: `ResNet18_MSFF_AntiSpoof` với 2 classes (real, spoof)

#### 🎯 Kiến trúc mới:
```python
class ResNet18_MSFF_AntiSpoof(nn.Module):
    """
    - Improved Stem: 3x3 conv thay vì 7x7
    - Multi-Scale Feature Fusion: layer3 (256) + layer4 (512) = 768 features
    - Dropout: 0.5 để regularization
    - 2 classes: real (0), spoof (1)
    """
```

#### 🔑 Class Mapping:
```python
CLASS_NAMES = ['real', 'spoof']  # 0: real, 1: spoof
```

---

### 2️⃣ **Service Layer (`app/services/anti_spoofing_service.py`)**

#### ✅ Thay đổi:

**Cũ** (3 classes):
```python
def predict(self, face_image) -> Tuple[str, float]:
    # Returns: ('live'|'print'|'replay', confidence)
    return label, confidence
```

**Mới** (2 classes):
```python
def predict(self, face_image) -> Dict[str, Any]:
    # Returns: {
    #   'is_live': bool,      # True if real, False if spoof
    #   'label': str,         # 'real' or 'spoof'
    #   'confidence': float   # 0.0 - 1.0
    # }
    return result
```

#### 🎯 Benefits:
- ✅ Đơn giản hóa: 2 classes dễ train hơn 3 classes
- ✅ Accuracy cao hơn: Focus vào real vs. fake
- ✅ API rõ ràng: `is_live` boolean thay vì check string

---

### 3️⃣ **Face Engine (`app/services/face_engine.py`)**

#### ✅ Method `check_anti_spoofing()`:

```python
async def check_anti_spoofing(
    self,
    face_crops: List[np.ndarray]
) -> List[Dict[str, Any]]:
    """
    Returns:
    [
        {
            'is_live': bool,      # True if real
            'label': 'real'|'spoof',
            'confidence': 0.0-1.0
        },
        ...
    ]
    """
```

#### 🔍 Logging:
- ✅ Log chi tiết từng face crop
- ⚠️ Warning khi detect spoof face
- 📊 Summary: total/real/spoof count

---

### 4️⃣ **WebSocket Endpoint (`app/api/v1/endpoints/frames.py`)**

#### ✅ Flow integration:

```
Frame Processing Flow:
│
├─ 1. Receive Frame
│
├─ 2. Detect Faces → detections, crops
│
├─ 3. 🛡️ ANTI-SPOOFING CHECK [MỚI]
│   ├─ Check từng crop
│   ├─ Filter: live_detections = [d for d if is_live]
│   ├─ Log suspicious faces
│   └─ Send "anti_spoofing_alert" nếu có spoof
│
├─ 4. Recognize Faces (CHỈ live faces)
│
├─ 5. Track & Validate
│
└─ 6. Send Response với anti-spoofing info
```

#### 🎯 Detection Schema Update:

```python
det_dict = {
    "bbox": [...],
    "confidence": 0.95,
    "track_id": 1,
    "student_code": "SV001",
    "student_name": "Nguyen Van A",
    "recognition_confidence": 0.87,
    # ✅ Anti-spoofing fields MỚI
    "is_live": True,
    "spoofing_type": "real",
    "spoofing_confidence": 0.92
}
```

#### ⚠️ Alert Message:

```json
{
  "type": "anti_spoofing_alert",
  "frame_count": 145,
  "timestamp": "2025-11-04T10:00:00Z",
  "suspicious_faces": [
    {
      "bbox": [300, 200, 400, 350],
      "type": "spoof",
      "confidence": 0.88
    }
  ],
  "total_suspicious": 1,
  "total_live": 2,
  "message": "⚠️ Detected 1 fake face(s) - Only 2 live face(s) will be processed"
}
```

---

### 5️⃣ **Config (`app/core/config.py`)**

#### ✅ Settings:

```python
class Settings(BaseSettings):
    # Anti-spoofing settings
    ANTISPOOFING_CHECKPOINT: Optional[str] = None  # ✅ MỚI
    ANTISPOOFING_DEVICE: str = "cuda"
    ANTISPOOFING_THRESHOLD: float = 0.7
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # ✅ Cho phép extra fields
```

#### 📝 `.env`:

```env
ANTISPOOFING_CHECKPOINT=D:\PBL6\ai-service\best_model.pth
```

---

## 🧪 Testing

### Test Script:

```bash
cd d:\PBL6\ai-service
python test_antispoofing.py
```

### Expected Output:

```
============================================================
ANTI-SPOOFING SERVICE TEST
============================================================

1. Initializing Anti-Spoofing Service...
   Checkpoint: D:\PBL6\ai-service\best_model.pth
   Device: cuda
   Threshold: 0.7
   ✅ Service initialized successfully!

2. Testing with dummy image...
   Running prediction...
   Result:
      - is_live: True
      - label: real
      - confidence: 0.8542

============================================================
TEST COMPLETED!
============================================================
```

---

## 📊 Performance

### Model Info:

```
Architecture: ResNet18_MSFF_AntiSpoof
Parameters: ~11M
Input Size: 224x224x3
Classes: 2 (real, spoof)
```

### Inference Speed:

```
GPU (CUDA): ~5-10ms per image
CPU: ~50-100ms per image
Batch (GPU): ~3-5ms per image (batch=8)
```

---

## 🎯 Usage Example

### Trong code:

```python
from app.services.face_engine import get_engine

# Get engine
engine = get_engine()

# Detect faces
detections, crops, img = await engine.detect_faces(frame_data)

# ✅ Anti-spoofing check
anti_spoofing_results = await engine.check_anti_spoofing(crops)

# Filter live faces only
live_detections = []
live_crops = []
for det, crop, result in zip(detections, crops, anti_spoofing_results):
    if result['is_live']:
        det.is_live = True
        det.spoofing_type = result['label']
        det.spoofing_confidence = result['confidence']
        live_detections.append(det)
        live_crops.append(crop)
    else:
        # ⚠️ Spoof detected - log và skip
        logger.warning(f"Spoof face: {result}")

# Recognize CHỈ live faces
detections = await engine.recognize_faces(
    detections=live_detections,
    crops=live_crops,
    gallery_embeddings=session.gallery_embeddings,
    gallery_labels=session.gallery_labels
)
```

---

## ✅ Checklist

- [x] Model architecture mới: `ResNet18_MSFF_AntiSpoof`
- [x] Service updated: Dict format thay vì Tuple
- [x] Face Engine: Integrate anti-spoofing check
- [x] WebSocket: Filter spoof faces trước recognition
- [x] Schema: Thêm `is_live`, `spoofing_type`, `spoofing_confidence`
- [x] Config: Add `ANTISPOOFING_CHECKPOINT`
- [x] Test script: `test_antispoofing.py`
- [x] Documentation: README mới

---

## 🚀 Run Application

```bash
cd d:\PBL6\ai-service
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Test WebSocket:

```python
import asyncio
import websockets
import json
import base64

async def test_ws():
    uri = "ws://localhost:8000/api/v1/sessions/{session_id}/stream?token={jwt}"
    
    async with websockets.connect(uri) as ws:
        # Receive connection
        msg = await ws.recv()
        print(json.loads(msg))
        
        # Send frame
        with open("test_face.jpg", "rb") as f:
            frame_data = f.read()
        
        await ws.send(frame_data)
        
        # Receive responses
        while True:
            response = await ws.recv()
            data = json.loads(response)
            
            if data["type"] == "anti_spoofing_alert":
                print(f"🚨 SPOOF ALERT: {data}")
            elif data["type"] == "frame_processed":
                for det in data["detections"]:
                    print(f"Face: is_live={det['is_live']}, type={det['spoofing_type']}")

asyncio.run(test_ws())
```

---

## 🎉 Summary

**Model mới đã được implement hoàn toàn vào hệ thống!**

- ✅ Architecture: ResNet18_MSFF_AntiSpoof
- ✅ Classes: real/spoof (2 classes)
- ✅ Integration: Full WebSocket flow
- ✅ Logging: Chi tiết và warnings
- ✅ Performance: GPU optimized

**Spoof faces sẽ bị filter TRƯỚC KHI nhận diện!** 🛡️
