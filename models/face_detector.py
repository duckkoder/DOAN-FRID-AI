from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

try:
    import yolov3_inference_only_person as yolo
except ImportError as exc:  # pragma: no cover - configuration error helper
    raise ImportError("Cannot import yolov3_inference_only_person. Ensure the file is on PYTHONPATH.") from exc


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
    """Thin wrapper that exposes the YOLOv3 face detector as a reusable class."""

    def __init__(
        self,
        checkpoint_path: Union[str, os.PathLike[str]],
        *,
        conf_threshold: float = 0.5,
        nms_threshold: float = 0.45,
        device: Optional[Union[str, torch.device]] = None,
        image_size: Optional[int] = None,
    ) -> None:
        if device is None:
            self.device = yolo.device
        else:
            self.device = torch.device(device)
            yolo.device = self.device
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.image_size = image_size or getattr(yolo, "IMAGE_SIZE", 416)
        self.model = yolo.load_model(str(checkpoint_path))
        self.model.to(self.device)
        self.model.eval()
        self._labels: Sequence[str] = getattr(yolo, "CLASS_LABELS", ["face"])

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
        tensor, original_image, size, letterbox_info = yolo.preprocess_image(image, target_size=self.image_size)
        with torch.no_grad():
            predictions = self.model(tensor)
        raw_detections = yolo.postprocess_detections(
            predictions,
            size,
            letterbox_info,
            conf_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
        )
        detections = self._to_detections(raw_detections, original_image.shape[:2])
        crops = None
        if return_crops and detections:
            crops = [self._crop_face(original_image, det, pad) for det in detections]
        return detections, crops, original_image

    def _to_detections(self, raw: List[Sequence[float]], shape: Tuple[int, int]) -> List[Detection]:
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
        top = max(0, detection.bbox[1] - pad)
        left = max(0, detection.bbox[0] - pad)
        bottom = min(image.shape[0], detection.bbox[3] + pad)
        right = min(image.shape[1], detection.bbox[2] + pad)
        return image[top:bottom, left:right].copy()
