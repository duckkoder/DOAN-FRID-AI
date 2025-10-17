"""
YOLOv3 Face Detector - Integrated Implementation
=================================================
Kết hợp model architecture và detection logic trong một file duy nhất.
Bỏ phần CLI/Interactive để tập trung cho API service.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile

# Cho phép load ảnh bị cắt
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==============================================================================
# CONSTANTS & HYPERPARAMETERS
# ==============================================================================

IMAGE_SIZE = 416
NUM_CLASSES = 1  # Chỉ có class "face"
CLASS_LABELS = ["face"]
DROPOUT_PROB = 0.3

# Grid sizes cho 3 scales
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]  # [13, 26, 52]

# Anchor boxes
ANCHORS = [
    [(0.25, 0.25), (0.35, 0.35), (0.45, 0.45)],  # Scale 1: Large faces
    [(0.10, 0.10), (0.15, 0.15), (0.20, 0.20)],  # Scale 2: Medium faces
    [(0.03, 0.03), (0.05, 0.05), (0.08, 0.08)],  # Scale 3: Small faces
]

# Default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================================
# MODEL ARCHITECTURE
# ==============================================================================

class CNNBlock(nn.Module):
    """Khối CNN cơ bản: Conv2d → BatchNorm2d → LeakyReLU"""
    def __init__(self, in_channels, out_channels, use_batch_norm=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not use_batch_norm, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1)
        self.use_batch_norm = use_batch_norm

    def forward(self, x):
        x = self.conv(x)
        if self.use_batch_norm:
            x = self.bn(x)
            return self.activation(x)
        else:
            return x


class ResidualBlock(nn.Module):
    """Residual Block với skip connections"""
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        res_layers = []
        for _ in range(num_repeats):
            res_layers += [
                nn.Sequential(
                    nn.Conv2d(channels, channels // 2, kernel_size=1),
                    nn.BatchNorm2d(channels // 2),
                    nn.LeakyReLU(0.1),
                    nn.Conv2d(channels // 2, channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.LeakyReLU(0.1)
                )
            ]
        self.layers = nn.ModuleList(res_layers)
        self.use_residual = use_residual

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)
        return x


class ScalePrediction(nn.Module):
    """Module dự đoán cho mỗi scale"""
    def __init__(self, in_channels, num_classes, dropout_prob=0.0):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(2 * in_channels),
            nn.LeakyReLU(0.1),
        )
        self.dropout = nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity()
        self.pred = nn.Conv2d(2 * in_channels, (num_classes + 5) * 3, kernel_size=1)
        self.num_classes = num_classes

    def forward(self, x):
        x = self.projection(x)
        x = self.dropout(x)
        output = self.pred(x)
        output = output.view(x.size(0), 3, self.num_classes + 5, x.size(2), x.size(3))
        output = output.permute(0, 1, 3, 4, 2)
        return output


class YOLOv3(nn.Module):
    """Mô hình YOLOv3 cho face detection"""
    def __init__(self, in_channels=3, num_classes=1, dropout_prob=DROPOUT_PROB):
        super().__init__()
        self.num_classes = num_classes
        
        # Darknet-53 backbone
        self.layers = nn.ModuleList([
            CNNBlock(in_channels, 32, kernel_size=3, stride=1, padding=1),
            CNNBlock(32, 64, kernel_size=3, stride=2, padding=1),
            ResidualBlock(64, num_repeats=1),
            CNNBlock(64, 128, kernel_size=3, stride=2, padding=1),
            ResidualBlock(128, num_repeats=2),
            CNNBlock(128, 256, kernel_size=3, stride=2, padding=1),
            ResidualBlock(256, num_repeats=8),
            CNNBlock(256, 512, kernel_size=3, stride=2, padding=1),
            ResidualBlock(512, num_repeats=8),
            CNNBlock(512, 1024, kernel_size=3, stride=2, padding=1),
            ResidualBlock(1024, num_repeats=4),
            CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0),
            CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1),
            CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0),
            CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1),
            CNNBlock(1024, 512, kernel_size=1, stride=1, padding=0),
        ])
        
        # Scale predictions
        self.scale1_pred = nn.Sequential(
            CNNBlock(512, 1024, kernel_size=3, stride=1, padding=1),
            ScalePrediction(1024, num_classes, dropout_prob=dropout_prob),
        )
        self.scale1_upsample = CNNBlock(512, 256, kernel_size=1, stride=1, padding=0)
        self.upsample = nn.Upsample(scale_factor=2)
        
        self.scale2_layers = nn.Sequential(
            CNNBlock(768, 256, kernel_size=1, stride=1, padding=0),
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1),
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0),
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1),
            CNNBlock(512, 256, kernel_size=1, stride=1, padding=0),
        )
        self.scale2_pred = nn.Sequential(
            CNNBlock(256, 512, kernel_size=3, stride=1, padding=1),
            ScalePrediction(512, num_classes, dropout_prob=dropout_prob),
        )
        self.scale2_upsample = CNNBlock(256, 128, kernel_size=1, stride=1, padding=0)
        
        self.scale3_layers = nn.Sequential(
            CNNBlock(384, 128, kernel_size=1, stride=1, padding=0),
            CNNBlock(128, 256, kernel_size=3, stride=1, padding=1),
            CNNBlock(256, 128, kernel_size=1, stride=1, padding=0),
            CNNBlock(128, 256, kernel_size=3, stride=1, padding=1),
            CNNBlock(256, 128, kernel_size=1, stride=1, padding=0),
        )
        self.scale3_pred = nn.Sequential(
            CNNBlock(128, 256, kernel_size=3, stride=1, padding=1),
            ScalePrediction(256, num_classes, dropout_prob=dropout_prob),
        )

    def forward(self, x):
        outputs = []
        route_connections = {}
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            outputs.append(x)
            if i == 6:  # 52x52x256
                route_connections['52x52'] = x
            elif i == 8:  # 26x26x512
                route_connections['26x26'] = x
        
        # Scale 1 (13x13)
        scale1_output = self.scale1_pred(x)
        x = self.scale1_upsample(x)
        x = self.upsample(x)
        x = torch.cat([x, route_connections['26x26']], dim=1)
        
        # Scale 2 (26x26)
        x = self.scale2_layers(x)
        scale2_output = self.scale2_pred(x)
        x = self.scale2_upsample(x)
        x = self.upsample(x)
        x = torch.cat([x, route_connections['52x52']], dim=1)
        
        # Scale 3 (52x52)
        x = self.scale3_layers(x)
        scale3_output = self.scale3_pred(x)
        
        return scale1_output, scale2_output, scale3_output


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def iou_boxes(box1, box2, is_pred=True):
    """Tính IoU giữa 2 bounding boxes"""
    if is_pred:
        b1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
        b1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
        b1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
        b1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2
        
        b2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
        b2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
        b2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
        b2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2
        
        x1 = torch.max(b1_x1, b2_x1)
        y1 = torch.max(b1_y1, b2_y1)
        x2 = torch.min(b1_x2, b2_x2)
        y2 = torch.min(b1_y2, b2_y2)
        
        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        box1_area = abs((b1_x2 - b1_x1) * (b1_y2 - b1_y1))
        box2_area = abs((b2_x2 - b2_x1) * (b2_y2 - b2_y1))
        
        return intersection / (box1_area + box2_area - intersection + 1e-6)
    else:
        return iou_boxes(torch.tensor(box1), torch.tensor(box2), is_pred=True)


def nms(bboxes, iou_threshold, confidence_threshold):
    """Non-Maximum Suppression"""
    bboxes = [box for box in bboxes if box[1] > confidence_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []
    
    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box for box in bboxes
            if box[0] != chosen_box[0]
            or iou_boxes(torch.tensor(chosen_box[2:]), torch.tensor(box[2:])) < iou_threshold
        ]
        bboxes_after_nms.append(chosen_box)
    
    return bboxes_after_nms


def convert_cells_to_bboxes(predictions, anchors, s, is_predictions=True):
    """Chuyển đổi cell predictions sang bounding boxes"""
    batch_size = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    
    if is_predictions:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]
    
    cell_indices = (
        torch.arange(s)
        .repeat(predictions.shape[0], 3, s, 1)
        .unsqueeze(-1)
        .to(predictions.device)
    )
    
    x = 1 / s * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / s * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    width_height = 1 / s * box_predictions[..., 2:4]
    
    converted_bboxes = torch.cat(
        (best_class, scores, x, y, width_height), dim=-1
    ).reshape(batch_size, num_anchors * s * s, 6)
    
    return converted_bboxes.tolist()


# ==============================================================================
# PREPROCESSING & POSTPROCESSING
# ==============================================================================

def preprocess_image(image, target_size=416):
    """Tiền xử lý ảnh cho inference"""
    # Đọc ảnh
    if isinstance(image, (str, Path)):
        image = cv2.imread(str(image))
        if image is None:
            raise ValueError(f"Không thể đọc ảnh: {image}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    original_image = image.copy()
    original_h, original_w = image.shape[:2]
    
    if original_h == 0 or original_w == 0:
        raise ValueError("Ảnh không hợp lệ")
    
    # Tính toán letterbox
    scale = min(target_size / original_w, target_size / original_h)
    nw = max(1, int(scale * original_w))
    nh = max(1, int(scale * original_h))
    pad_x = max(0, (target_size - nw) // 2)
    pad_y = max(0, (target_size - nh) // 2)
    
    letterbox_info = {
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "input_size": target_size,
        "new_width": nw,
        "new_height": nh,
    }
    
    # Resize và padding
    image_resized = cv2.resize(image, (nw, nh))
    new_image = np.full((target_size, target_size, 3), 128, dtype=np.uint8)
    new_image[pad_y:pad_y+nh, pad_x:pad_x+nw] = image_resized
    
    # Normalize và convert to tensor
    processed_image = new_image.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(processed_image).permute(2, 0, 1).unsqueeze(0).to(device)
    
    return image_tensor, original_image, (original_h, original_w), letterbox_info


def postprocess_detections(predictions, original_size, letterbox_info, conf_threshold=0.5, nms_threshold=0.45):
    """Xử lý predictions thành bounding boxes cuối cùng"""
    original_h, original_w = original_size
    
    # Chuẩn bị anchors
    anchors = torch.tensor(ANCHORS) * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    anchors = anchors.to(device)
    
    # Convert predictions
    all_bboxes = []
    for i, prediction in enumerate(predictions):
        bboxes = convert_cells_to_bboxes(prediction, anchors[i], S[i], is_predictions=True)
        all_bboxes.extend(bboxes[0])
    
    # Áp dụng NMS
    final_bboxes = nms(all_bboxes, nms_threshold, conf_threshold)
    
    # Quy đổi về ảnh gốc
    input_size = float(letterbox_info.get("input_size", IMAGE_SIZE))
    scale = float(letterbox_info.get("scale", 1.0))
    pad_x = float(letterbox_info.get("pad_x", 0.0))
    pad_y = float(letterbox_info.get("pad_y", 0.0))
    
    adjusted_bboxes = []
    for box in final_bboxes:
        class_pred, confidence, x, y, width, height = box
        
        # Chuyển sang pixel trong letterbox
        box_x = x * input_size
        box_y = y * input_size
        box_w = width * input_size
        box_h = height * input_size
        
        # Loại bỏ padding và scale về ảnh gốc
        box_x = (box_x - pad_x) / scale
        box_y = (box_y - pad_y) / scale
        box_w = box_w / scale
        box_h = box_h / scale
        
        # Convert to corners và clamp
        x1 = max(0.0, min(original_w, box_x - box_w / 2))
        y1 = max(0.0, min(original_h, box_y - box_h / 2))
        x2 = max(0.0, min(original_w, box_x + box_w / 2))
        y2 = max(0.0, min(original_h, box_y + box_h / 2))
        
        box_w = max(1e-6, x2 - x1)
        box_h = max(1e-6, y2 - y1)
        box_x = x1 + box_w / 2
        box_y = y1 + box_h / 2
        
        # Normalize về [0,1]
        x_norm = box_x / original_w
        y_norm = box_y / original_h
        w_norm = box_w / original_w
        h_norm = box_h / original_h
        
        adjusted_bboxes.append([class_pred, confidence, x_norm, y_norm, w_norm, h_norm])
    
    return adjusted_bboxes


def load_model(checkpoint_path):
    """Load YOLOv3 model từ checkpoint"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ Không tìm thấy checkpoint: {checkpoint_path}")
    
    model = YOLOv3(num_classes=NUM_CLASSES, dropout_prob=DROPOUT_PROB).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    
    return model


# ==============================================================================
# DETECTION DATACLASS & WRAPPER
# ==============================================================================

@dataclass(frozen=True)
class Detection:
    bbox: Tuple[int, int, int, int]
    confidence: float
    label: str
    normalized_bbox: Tuple[float, float, float, float]
    
    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


class YOLOFaceDetector:
    """Wrapper class cho YOLOv3 face detection"""
    
    def __init__(
        self,
        checkpoint_path: Union[str, os.PathLike[str]],
        *,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        device: Optional[Union[str, torch.device]] = None,
        image_size: Optional[int] = None,
    ) -> None:
        # Xử lý device parameter
        global_device = globals()['device']
        if device is None:
            self.device = global_device
        else:
            self.device = torch.device(device) if isinstance(device, str) else device
            # Cập nhật global device
            globals()['device'] = self.device
        
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.image_size = image_size or IMAGE_SIZE
        self.model = load_model(str(checkpoint_path))
        self.model.to(self.device)
        self.model.eval()
        self._labels = CLASS_LABELS
    
    @property
    def labels(self) -> Sequence[str]:
        return self._labels
    
    def detect(
        self,
        image: Union[str, np.ndarray],
        *,
        return_crops: bool = False,
        pad: int = 0,
    ) -> Tuple[List[Detection], Optional[List[np.ndarray]], np.ndarray]:
        """Phát hiện khuôn mặt trong ảnh"""
        # Preprocess
        tensor, original_image, size, letterbox_info = preprocess_image(image, target_size=self.image_size)
        
        # Inference
        with torch.no_grad():
            predictions = self.model(tensor)
        
        # Postprocess
        raw_detections = postprocess_detections(
            predictions,
            size,
            letterbox_info,
            conf_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
        )
        
        # Convert to Detection objects
        detections = self._to_detections(raw_detections, original_image.shape[:2])
        
        # Crop faces if requested
        crops = None
        if return_crops and detections:
            crops = [self._crop_face(original_image, det, pad) for det in detections]
        
        return detections, crops, original_image
    
    def _to_detections(self, raw: List[Sequence[float]], shape: Tuple[int, int]) -> List[Detection]:
        """Convert raw detections to Detection objects"""
        h, w = shape
        results: List[Detection] = []
        
        for item in raw:
            if len(item) != 6:
                continue
            class_id, conf, cx_norm, cy_norm, w_norm, h_norm = item
            
            cx = cx_norm * w
            cy = cy_norm * h
            bw = w_norm * w
            bh = h_norm * h
            
            x1 = max(0, int(round(cx - bw / 2)))
            y1 = max(0, int(round(cy - bh / 2)))
            x2 = min(w, int(round(cx + bw / 2)))
            y2 = min(h, int(round(cy + bh / 2)))
            
            label_idx = int(class_id) if 0 <= int(class_id) < len(self._labels) else 0
            label = self._labels[label_idx]
            
            results.append(
                Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=float(conf),
                    label=label,
                    normalized_bbox=(float(cx_norm), float(cy_norm), float(w_norm), float(h_norm)),
                )
            )
        
        return results
    
    @staticmethod
    def _crop_face(image: np.ndarray, detection: Detection, pad: int) -> np.ndarray:
        """Crop khuôn mặt từ ảnh"""
        top = max(0, detection.bbox[1] - pad)
        left = max(0, detection.bbox[0] - pad)
        bottom = min(image.shape[0], detection.bbox[3] + pad)
        right = min(image.shape[1], detection.bbox[2] + pad)
        return image[top:bottom, left:right].copy()

