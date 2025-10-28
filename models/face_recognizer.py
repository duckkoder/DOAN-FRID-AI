from __future__ import annotations

import os
import pathlib
import pickle
import math
from contextlib import suppress
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch import Tensor
from torch.nn import (
    AdaptiveAvgPool2d,
    BatchNorm1d,
    BatchNorm2d,
    Conv2d,
    Dropout,
    Linear,
    MaxPool2d,
    Module,
    PReLU,
    ReLU,
    Sequential,
    Sigmoid,
)

# ---------------------------------------------------------------------------
# Backbone definition (ArcFace IR-SE 50)
# ---------------------------------------------------------------------------


def l2_norm(input_tensor: Tensor, axis: int = 1) -> Tensor:
    norm = torch.norm(input_tensor, 2, axis, keepdim=True).clamp(min=1e-12)
    return torch.div(input_tensor, norm)


class Flatten(Module):
    def forward(self, x: Tensor) -> Tensor:  # noqa: D401 - simple reshape
        return x.view(x.size(0), -1)


class SEModule(Module):
    def __init__(self, channels: int, reduction: int) -> None:
        super().__init__()
        self.avg_pool = AdaptiveAvgPool2d(1)
        self.fc1 = Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return identity * x


class BottleneckIR(Module):
    def __init__(self, in_channels: int, depth: int, stride: int) -> None:
        super().__init__()
        if in_channels == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channels, depth, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(depth),
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channels),
            Conv2d(in_channels, depth, kernel_size=3, stride=1, padding=1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, kernel_size=3, stride=stride, padding=1, bias=False),
            BatchNorm2d(depth),
        )

    def forward(self, x: Tensor) -> Tensor:
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


class BottleneckIRSE(Module):
    def __init__(self, in_channels: int, depth: int, stride: int) -> None:
        super().__init__()
        if in_channels == depth:
            self.shortcut_layer = MaxPool2d(1, stride)
        else:
            self.shortcut_layer = Sequential(
                Conv2d(in_channels, depth, kernel_size=1, stride=stride, bias=False),
                BatchNorm2d(depth),
            )
        self.res_layer = Sequential(
            BatchNorm2d(in_channels),
            Conv2d(in_channels, depth, kernel_size=3, stride=1, padding=1, bias=False),
            PReLU(depth),
            Conv2d(depth, depth, kernel_size=3, stride=stride, padding=1, bias=False),
            BatchNorm2d(depth),
            SEModule(depth, 16),
        )

    def forward(self, x: Tensor) -> Tensor:
        shortcut = self.shortcut_layer(x)
        res = self.res_layer(x)
        return res + shortcut


@dataclass(frozen=True)
class Block:
    in_channels: int
    depth: int
    stride: int


def _get_blocks(num_layers: int) -> List[List[Block]]:
    if num_layers == 50:
        return [
            [Block(64, 64, 2)] + [Block(64, 64, 1) for _ in range(2)],
            [Block(64, 128, 2)] + [Block(128, 128, 1) for _ in range(3)],
            [Block(128, 256, 2)] + [Block(256, 256, 1) for _ in range(13)],
            [Block(256, 512, 2)] + [Block(512, 512, 1) for _ in range(2)],
        ]
    if num_layers == 100:
        return [
            [Block(64, 64, 2)] + [Block(64, 64, 1) for _ in range(2)],
            [Block(64, 128, 2)] + [Block(128, 128, 1) for _ in range(12)],
            [Block(128, 256, 2)] + [Block(256, 256, 1) for _ in range(29)],
            [Block(256, 512, 2)] + [Block(512, 512, 1) for _ in range(2)],
        ]
    if num_layers == 152:
        return [
            [Block(64, 64, 2)] + [Block(64, 64, 1) for _ in range(2)],
            [Block(64, 128, 2)] + [Block(128, 128, 1) for _ in range(7)],
            [Block(128, 256, 2)] + [Block(256, 256, 1) for _ in range(35)],
            [Block(256, 512, 2)] + [Block(512, 512, 1) for _ in range(2)],
        ]
    raise ValueError("Unsupported number of layers for ArcFace backbone")


class Backbone(Module):
    def __init__(self, num_layers: int = 50, drop_ratio: float = 0.6, mode: str = "ir_se") -> None:
        super().__init__()
        if num_layers not in (50, 100, 152):
            raise ValueError("num_layers should be 50, 100 or 152")
        if mode not in ("ir", "ir_se"):
            raise ValueError("mode should be 'ir' or 'ir_se'")
        blocks = _get_blocks(num_layers)
        unit = BottleneckIR if mode == "ir" else BottleneckIRSE
        self.input_layer = Sequential(
            Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            BatchNorm2d(64),
            PReLU(64),
        )
        modules: List[Module] = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit(bottleneck.in_channels, bottleneck.depth, bottleneck.stride))
        self.body = Sequential(*modules)
        self.output_layer = Sequential(
            BatchNorm2d(512),
            Dropout(drop_ratio),
            Flatten(),
            Linear(512 * 7 * 7, 512),
            BatchNorm1d(512),
        )

    def forward(self, x: Tensor) -> Tensor:  # noqa: D401 - canonical forward
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)


# ---------------------------------------------------------------------------
# Face recognizer wrapper
# ---------------------------------------------------------------------------


class FaceRecognizer:
    """Load ArcFace backbone checkpoints and expose verification helpers."""

    def __init__(
        self,
        checkpoint_path: Union[str, os.PathLike[str]],
        *,
        device: Optional[Union[str, torch.device]] = None,
        threshold: float = 1.5,
        transform: Optional[T.Compose] = None,
        knn_k: int = 5,
        enable_dynamic_threshold: bool = True,
        per_identity_quantile: float = 0.95,
        per_identity_margin: float = 0.05,
        identity_threshold_min_scale: float = 0.5,
        identity_threshold_max_scale: float = 1.5,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.checkpoint_path = pathlib.Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        self.threshold = threshold
        self.model = self._load_model()
        self.model.to(self.device)
        self.model.eval()
        self.transform = transform or T.Compose(
            [
                T.Resize((112, 112)),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self._database: Dict[str, Tensor] = {}
        self.knn_k = max(1, int(knn_k))
        self.enable_dynamic_threshold = bool(enable_dynamic_threshold)
        self.per_identity_quantile = float(per_identity_quantile)
        self.per_identity_margin = float(per_identity_margin)
        self.identity_threshold_min_scale = float(identity_threshold_min_scale)
        self.identity_threshold_max_scale = float(identity_threshold_max_scale)
        self._centroids: Dict[str, Tensor] = {}
        self._identity_thresholds: Dict[str, float] = {}
        self._gallery_embeddings: Optional[Tensor] = None
        self._gallery_labels: List[str] = []
        self._stats_stale = True

    # ------------------------------------------------------------------
    # Model loading utilities
    # ------------------------------------------------------------------

    def _load_model(self) -> Module:
        # errors = []
        # # Try TorchScript first
        # try:
        #     model = torch.jit.load(str(self.checkpoint_path), map_location=self.device)
        #     model.eval()
        #     return model
        # except Exception as exc:  # noqa: PERF203 - collecting error info
        #     errors.append(('torchscript', exc))
        # # Try cross-platform pickle load (handles pathlib paths, etc.)
        # try:
        #     checkpoint = self._cross_platform_load(self.checkpoint_path)
        #     return self._build_model_from_checkpoint(checkpoint)
        # except Exception as exc:
        #     errors.append(('pickle', exc))
        # # Fall back to plain torch.load (safe default: weights_only=True in recent PyTorch)
        # try:
        #     checkpoint = torch.load(str(self.checkpoint_path), map_location=self.device)
        #     return self._build_model_from_checkpoint(checkpoint)
        # except Exception as exc:
        #     errors.append(('torch.load', exc))
        # # Final fallback: allowlist numpy scalar and opt-in to full pickle (weights_only=False)
        # try:
        #     checkpoint = self._torch_load_with_allowlist()
        #     return self._build_model_from_checkpoint(checkpoint)
        # except Exception as exc:
        #     errors.append(('torch.load(weights_only=False)', exc))
        
        try:
            checkpoint = torch.load(
                str(self.checkpoint_path), 
                map_location=self.device,
                weights_only=False
            )
            return self._build_model_from_checkpoint(checkpoint)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")

        details = '; '.join(f"{stage}: {type(err).__name__}({err})" for stage, err in errors)
        raise RuntimeError(f"Failed to load ArcFace checkpoint: {self.checkpoint_path}. Tried {details}")

    def _build_model_from_checkpoint(self, checkpoint: object) -> Module:
        if isinstance(checkpoint, Module):
            checkpoint = checkpoint.to(self.device)
            checkpoint.eval()
            return checkpoint
        if not isinstance(checkpoint, dict):
            raise TypeError("Unsupported checkpoint type; expected state dict or dict container")
        state_dict = self._extract_state_dict(checkpoint)
        model = Backbone(num_layers=50, drop_ratio=0.6, mode="ir_se")
        model.load_state_dict(state_dict)
        return model

    @staticmethod
    def _extract_state_dict(container: Dict[str, object]) -> Dict[str, Tensor]:
        possible_keys = (
            "model_state_dict",
            "state_dict",
            "backbone",
            "model",
        )
        state_dict: Optional[Dict[str, Tensor]] = None
        for key in possible_keys:
            candidate = container.get(key)
            if isinstance(candidate, dict):
                state_dict = candidate
                break
        if state_dict is None:
            if all(isinstance(v, Tensor) for v in container.values()):
                state_dict = container  # Already a state dict
            else:
                raise KeyError("Cannot find state dict in checkpoint container")
        # Handle DataParallel prefixes if present
        state_dict = {  # type: ignore[assignment]
            (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()
        }
        return state_dict

    @staticmethod
    def _cross_platform_load(path: pathlib.Path) -> Dict[str, Tensor]:
        class _Unpickler(pickle.Unpickler):
            def find_class(self, module: str, name: str):  # noqa: D401 - pickling helper
                if module == "pathlib" and name in {"PosixPath", "WindowsPath"}:
                    return pathlib.Path
                return super().find_class(module, name)

        with path.open("rb") as file:
            return _Unpickler(file).load()

    def _torch_load_with_allowlist(self) -> object:
        scalar = getattr(np.core.multiarray, 'scalar', None)
        allowed = [scalar] if scalar is not None else []
        serialization_module = getattr(torch, 'serialization', None)
        if serialization_module is not None and allowed:
            safe_globals_ctx = getattr(serialization_module, 'safe_globals', None)
            if safe_globals_ctx is not None:
                with safe_globals_ctx(allowed):
                    return torch.load(str(self.checkpoint_path), map_location=self.device, weights_only=False)
            add_safe_globals = getattr(serialization_module, 'add_safe_globals', None)
            if add_safe_globals is not None:
                with suppress(Exception):
                    add_safe_globals(allowed)
        return torch.load(str(self.checkpoint_path), map_location=self.device, weights_only=False)



    @staticmethod
    def _sanitize_identity(identity: str) -> str:
        cleaned = identity.strip()
        if not cleaned:
            raise ValueError("Identity name cannot be empty")
        invalid_chars = set('<>:"/\|?*')
        return ''.join('_' if ch in invalid_chars else ch for ch in cleaned)

    def sanitize_identity(self, identity: str) -> str:
        return self._sanitize_identity(identity)

    def save_embedding(
        self,
        root: Union[str, os.PathLike[str]],
        identity: str,
        embedding: Union[Tensor, np.ndarray],  # ✅ Support both Tensor and ndarray
    ) -> pathlib.Path:
        clean_identity = self._sanitize_identity(identity)
        root_path = pathlib.Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        person_dir = root_path / clean_identity
        person_dir.mkdir(parents=True, exist_ok=True)
        existing = [p for p in person_dir.glob('embedding_*.pt')]
        next_index = len(existing) + 1
        
        # ✅ Convert numpy to tensor if needed
        if isinstance(embedding, np.ndarray):
            embedding_cpu = torch.from_numpy(embedding).to(dtype=torch.float32)
        else:
            embedding_cpu = embedding.detach().cpu()
        
        if embedding_cpu.ndim > 1:
            embedding_cpu = embedding_cpu.view(-1)
        file_path = person_dir / f'embedding_{next_index:04d}.pt'
        torch.save(embedding_cpu, file_path)
        self.append_embedding(clean_identity, embedding)
        return file_path

    def append_embedding(self, identity: str, embedding: Union[Tensor, np.ndarray]) -> None:
        # ✅ Convert numpy to tensor if needed
        if isinstance(embedding, np.ndarray):
            vector = torch.from_numpy(embedding).to(self.device, dtype=torch.float32)
        else:
            vector = embedding.detach().to(self.device)
        
        if vector.ndim == 1:
            vector = vector.unsqueeze(0)
        elif vector.ndim != 2:
            vector = vector.view(1, -1)
        normalized = l2_norm(vector, axis=1)
        if identity in self._database:
            self._database[identity] = torch.cat([self._database[identity], normalized], dim=0)
        else:
            self._database[identity] = normalized
        self._mark_database_dirty()

    def ingest_image_folder(
        self,
        source_root: Union[str, os.PathLike[str]],
        destination_root: Union[str, os.PathLike[str]],
        *,
        tta: bool = False,
        min_images: int = 1,
        allowed_exts: Optional[Sequence[str]] = None,
    ) -> Dict[str, int]:
        source_path = pathlib.Path(source_root)
        destination_path = pathlib.Path(destination_root)
        if not source_path.exists():
            raise FileNotFoundError(f"Source folder not found: {source_path}")
        destination_path.mkdir(parents=True, exist_ok=True)
        extensions = tuple(e.lower() for e in (allowed_exts or ('.jpg', '.jpeg', '.png', '.bmp', '.webp')))
        saved_per_identity: Dict[str, int] = {}
        for person_dir in sorted(p for p in source_path.iterdir() if p.is_dir()):
            images = [p for p in sorted(person_dir.iterdir()) if p.suffix.lower() in extensions]
            if len(images) < min_images:
                continue
            identity = self._sanitize_identity(person_dir.name)
            for image_path in images:
                try:
                    embedding = self.extract_features(image_path, tta=tta)  # Returns np.ndarray
                except Exception:
                    continue
                self.save_embedding(destination_path, identity, embedding)
                saved_per_identity[identity] = saved_per_identity.get(identity, 0) + 1
        if not saved_per_identity:
            raise ValueError("No embeddings were created from the provided folder")
        return saved_per_identity

    def load_embedding_directory(
        self,
        root: Union[str, os.PathLike[str]],
        *,
        min_embeddings: int = 1,
    ) -> Dict[str, Tensor]:
        root_path = pathlib.Path(root)
        root_path.mkdir(parents=True, exist_ok=True)
        database: Dict[str, Tensor] = {}
        for person_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
            embeddings: List[Tensor] = []
            for emb_file in sorted(person_dir.glob('*.pt')):
                try:
                    tensor = self._load_tensor_file(emb_file)
                except Exception:
                    continue
                if isinstance(tensor, dict):
                    for key in ('embedding', 'tensor', 'vector'):
                        if key in tensor:
                            tensor = tensor[key]
                            break
                if not isinstance(tensor, torch.Tensor):
                    continue
                tensor = tensor.to(self.device, dtype=torch.float32)
                if tensor.ndim == 1:
                    embeddings.append(tensor)
                elif tensor.ndim == 2:
                    for row in tensor:
                        embeddings.append(row)
                else:
                    embeddings.append(tensor.view(-1))
            if len(embeddings) >= min_embeddings:
                stacked = torch.stack(embeddings, dim=0)
                stacked = l2_norm(stacked, axis=1)
                database[person_dir.name] = stacked.to(self.device)
        self._database = database
        self._mark_database_dirty()
        return database

    def _load_tensor_file(self, path: pathlib.Path) -> Tensor:
        loaders = [
            lambda: torch.load(str(path), map_location=self.device, weights_only=False),
            lambda: torch.load(str(path), map_location=self.device),
        ]
        for loader in loaders:
            with suppress(Exception):
                tensor = loader()
                return tensor
        raise RuntimeError(f'Failed to load embedding tensor: {path}')

    def _mark_database_dirty(self) -> None:
        self._stats_stale = True
        self._gallery_embeddings = None
        self._gallery_labels = []

    def _ensure_statistics(self) -> None:
        if self._stats_stale:
            self._recompute_statistics()

    def _recompute_statistics(self) -> None:
        centroids: Dict[str, Tensor] = {}
        thresholds: Dict[str, float] = {}
        gallery_embeddings: List[Tensor] = []
        gallery_labels: List[str] = []
        base_threshold = float(self.threshold)
        for name, embeddings in self._database.items():
            vectors = embeddings
            if vectors.ndim == 1:
                vectors = vectors.unsqueeze(0)
            if vectors.device != self.device:
                vectors = vectors.to(self.device)
            centroid = l2_norm(vectors.mean(dim=0, keepdim=True), axis=1).squeeze(0)
            centroids[name] = centroid
            thresholds[name] = self._compute_identity_threshold(vectors, centroid, base_threshold)
            gallery_embeddings.append(vectors)
            gallery_labels.extend([name] * vectors.size(0))
        if gallery_embeddings:
            self._gallery_embeddings = torch.cat(gallery_embeddings, dim=0)
        else:
            self._gallery_embeddings = None
        self._gallery_labels = gallery_labels
        self._centroids = centroids
        self._identity_thresholds = thresholds
        self._stats_stale = False

    def _compute_identity_threshold(
        self,
        vectors: Tensor,
        centroid: Tensor,
        global_threshold: float,
    ) -> float:
        if vectors.ndim == 1:
            vectors = vectors.unsqueeze(0)
        centroid = centroid.to(vectors.device)
        distances = torch.sum((vectors - centroid.unsqueeze(0)) ** 2, dim=1)
        if distances.numel() == 0:
            return float(global_threshold)
        if distances.numel() == 1:
            base = float(distances.item()) + self.per_identity_margin
        else:
            base = self._percentile(distances, self.per_identity_quantile) + self.per_identity_margin
        mean_distance = float(distances.mean().item())
        base = max(base, mean_distance)
        lower = global_threshold * self.identity_threshold_min_scale
        upper = global_threshold * self.identity_threshold_max_scale
        if lower > upper:
            lower, upper = upper, lower
        threshold = max(lower, min(base, upper))
        if not math.isfinite(threshold):
            threshold = global_threshold
        return float(threshold)

    @staticmethod
    def _percentile(distances: Tensor, q: float) -> float:
        if distances.numel() == 0:
            return 0.0
        q = float(q)
        if q < 0.0:
            q = 0.0
        elif q > 1.0:
            q = 1.0
        sorted_vals, _ = torch.sort(distances.detach().float().cpu())
        index = int(round((sorted_vals.numel() - 1) * q))
        index = max(0, min(index, sorted_vals.numel() - 1))
        return float(sorted_vals[index].item())

    # ------------------------------------------------------------------
    # Feature extraction and helpers
    # ------------------------------------------------------------------

    def extract_features(
        self,
        image: Union[str, os.PathLike[str], Image.Image, np.ndarray],
        *,
        tta: bool = False,
        tta_mode: str = 'basic',
    ) -> np.ndarray:
        """
        Extract face embedding features with optional test-time augmentation.
        
        Args:
            image: Input image
            tta: Enable test-time augmentation
            tta_mode: 'basic' (flip only) or 'advanced' (multiple augmentations)
            
        Returns:
            Normalized embedding as numpy array (512,)
        """
        tensor = self._prepare_tensor(image)
        feats = self._forward_tensor(tensor)
        
        if tta:
            if tta_mode == 'basic':
                # Basic: horizontal flip only
                flipped = torch.flip(tensor, dims=[3])
                feats = (feats + self._forward_tensor(flipped)) / 2.0
                
            elif tta_mode == 'advanced':
                # Advanced: multiple augmentations
                augmented_feats = [feats]
                
                # 1. Horizontal flip
                flipped = torch.flip(tensor, dims=[3])
                augmented_feats.append(self._forward_tensor(flipped))
                
                # 2. Brightness adjustments
                bright = torch.clamp(tensor * 1.1, 0, 1)
                augmented_feats.append(self._forward_tensor(bright))
                
                dark = torch.clamp(tensor * 0.9, 0, 1)
                augmented_feats.append(self._forward_tensor(dark))
                
                # Average all augmentations
                feats = torch.stack(augmented_feats).mean(dim=0)
        
        # ✅ FIX: Convert to CPU then numpy
        normalized = l2_norm(feats).detach().cpu().numpy()
        return normalized.squeeze(0) if normalized.ndim > 1 else normalized

    def _prepare_tensor(self, image: Union[str, os.PathLike[str], Image.Image, np.ndarray]) -> Tensor:
        pil_image = self._to_pil(image)
        tensor = self.transform(pil_image).unsqueeze(0)
        return tensor.to(self.device, non_blocking=True)

    @staticmethod
    def _to_pil(image: Union[str, os.PathLike[str], Image.Image, np.ndarray]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, (str, os.PathLike)):
            with Image.open(image) as img:
                return img.convert("RGB")
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = np.stack([image] * 3, axis=-1)
            if image.shape[2] == 4:
                image = image[..., :3]
            return Image.fromarray(image.astype(np.uint8)).convert("RGB")
        raise TypeError("Unsupported image type for feature extraction")

    def _forward_tensor(self, tensor: Tensor) -> Tensor:
        with torch.no_grad():
            output = self.model(tensor)
        if isinstance(output, (list, tuple)):
            output = output[0]
        return output

    # ------------------------------------------------------------------
    # Verification utilities
    # ------------------------------------------------------------------

    def compare(
        self,
        image_a: Union[str, os.PathLike[str], Image.Image, np.ndarray],
        image_b: Union[str, os.PathLike[str], Image.Image, np.ndarray],
        *,
        threshold: Optional[float] = None,
        tta: bool = False,
    ) -> Dict[str, Union[bool, float]]:
        threshold = threshold if threshold is not None else self.threshold
        emb_a = self.extract_features(image_a, tta=tta)  # Returns np.ndarray
        emb_b = self.extract_features(image_b, tta=tta)  # Returns np.ndarray
        distance = self._distance(emb_a, emb_b)
        return {
            "distance": distance,
            "threshold": float(threshold),
            "is_same": distance < threshold,
            "confidence": float(1.0 / (1.0 + distance)),
        }

    def build_database(
        self,
        root: Union[str, os.PathLike[str]],
        *,
        tta: bool = False,
        min_images: int = 1,
    ) -> Dict[str, Tensor]:
        root_path = pathlib.Path(root)
        if not root_path.exists():
            raise FileNotFoundError(f"Database folder not found: {root_path}")
        database: Dict[str, Tensor] = {}
        for person_dir in sorted(p for p in root_path.iterdir() if p.is_dir()):
            embeddings: List[Tensor] = []
            for image_path in sorted(person_dir.glob("*")):
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    continue
                try:
                    embedding = self.extract_features(image_path, tta=tta)  # Returns np.ndarray
                    # Convert numpy to tensor for stacking
                    embeddings.append(torch.from_numpy(embedding))
                except Exception:
                    continue
            if len(embeddings) >= min_images:
                stacked = torch.stack(embeddings, dim=0)
                stacked = l2_norm(stacked, axis=1)
                database[person_dir.name] = stacked.to(self.device)
        if not database:
            raise ValueError("Database folder does not contain any valid face images")
        self._database = database
        self._mark_database_dirty()
        return self._database

    def identify(
        self,
        image: Union[str, os.PathLike[str], Image.Image, np.ndarray],
        *,
        threshold: Optional[float] = None,
        tta: bool = False,
        gallery_embeddings: Optional[Tensor] = None,
        gallery_labels: Optional[List[str]] = None,
    ) -> Dict[str, Union[str, float, Sequence[Dict[str, float]]]]:
        """
        Identify a face from an image.
        
        Args:
            image: Input image (path, PIL Image, or numpy array)
            threshold: Recognition threshold (override default)
            tta: Enable test-time augmentation
            gallery_embeddings: External gallery embeddings tensor (N, 512) on GPU [OPTIONAL]
            gallery_labels: External gallery labels list (N,) [OPTIONAL]
            
        Returns:
            Recognition result dictionary
            
        Note:
            If gallery_embeddings and gallery_labels are provided, they will be used
            instead of the internal database. This is useful for session-based recognition
            where embeddings are loaded from database and kept in VRAM.
        """
        # Use external gallery if provided, otherwise use internal database
        if gallery_embeddings is not None and gallery_labels is not None:
            if gallery_embeddings.shape[0] != len(gallery_labels):
                raise ValueError(f"Mismatch: {gallery_embeddings.shape[0]} embeddings vs {len(gallery_labels)} labels")
            gallery = gallery_embeddings
            labels = gallery_labels
            # For external gallery, use global threshold only (no dynamic thresholds)
            use_dynamic = False
        else:
            if not self._database:
                raise RuntimeError("Face database is empty. Call build_database() first.")
            self._ensure_statistics()
            if not self._gallery_labels or self._gallery_embeddings is None:
                raise RuntimeError("Face database is empty. Call build_database() first.")
            gallery = self._gallery_embeddings
            labels = self._gallery_labels
            use_dynamic = self.enable_dynamic_threshold
            
        global_threshold = threshold if threshold is not None else self.threshold
        global_threshold = float(global_threshold)
        query_np = self.extract_features(image, tta=tta)  # Returns np.ndarray (512,)
        
        # ✅ Convert numpy to tensor and move to same device as gallery
        query = torch.from_numpy(query_np).to(gallery.device, dtype=torch.float32)
        
        diff = gallery - query.unsqueeze(0)
        distances_all = torch.sum(diff * diff, dim=1)
        k = min(self.knn_k, distances_all.shape[0])
        if k == 0:
            raise RuntimeError("Face database is empty. Call build_database() first.")
        topk_distances, topk_indices = torch.topk(distances_all, k=k, largest=False)
        vote_scores: Dict[str, float] = {}
        nearest_per_identity: Dict[str, float] = {}
        neighbors: List[Dict[str, float]] = []
        topk_distance_list = topk_distances.detach().cpu().tolist()
        topk_index_list = topk_indices.detach().cpu().tolist()
        for dist, idx in zip(topk_distance_list, topk_index_list):
            name = labels[idx]  # Use external or internal labels
            weight = 1.0 / (dist + 1e-6)
            vote_scores[name] = vote_scores.get(name, 0.0) + weight
            if name not in nearest_per_identity or dist < nearest_per_identity[name]:
                nearest_per_identity[name] = dist
            neighbors.append({"person": name, "distance": float(dist)})
        if vote_scores:
            best_name = max(
                vote_scores.items(),
                key=lambda item: (item[1], -nearest_per_identity[item[0]]),
            )[0]
            best_distance = float(nearest_per_identity[best_name])
        else:
            min_index = int(torch.argmin(distances_all).item())
            best_name = labels[min_index]  # Use external or internal labels
            best_distance = float(distances_all[min_index].item())
            neighbors.insert(0, {"person": best_name, "distance": best_distance})
            vote_scores[best_name] = 1.0
            nearest_per_identity[best_name] = best_distance
        centroid_distance = None
        # Only use centroid if internal database is used
        if use_dynamic and best_name in self._centroids:
            centroid = self._centroids[best_name]
            centroid_distance = float(torch.sum((centroid - query) ** 2).item())
        identity_threshold = self._identity_thresholds.get(best_name, float(global_threshold)) if use_dynamic else float(global_threshold)
        if threshold is not None:
            identity_threshold = min(identity_threshold, float(global_threshold))
        used_threshold = identity_threshold if use_dynamic else float(global_threshold)
        distance_for_conf = centroid_distance if centroid_distance is not None else best_distance
        vote_total = sum(vote_scores.values())
        vote_ratio = float(vote_scores.get(best_name, 0.0) / vote_total) if vote_total else 0.0
        confidence = 1.0 / (1.0 + distance_for_conf)
        confidence *= 0.5 + 0.5 * vote_ratio
        confidence = float(max(0.0, min(1.0, confidence)))
        recognized = best_name if distance_for_conf < used_threshold else "Unknown"
        candidates: List[Dict[str, float]] = []
        for name, score in vote_scores.items():
            min_dist = float(nearest_per_identity.get(name, float("inf")))
            vote_share = float(score / vote_total) if vote_total else 0.0
            candidate_conf = float(1.0 / (1.0 + min_dist)) if math.isfinite(min_dist) else 0.0
            candidates.append(
                {
                    "person": name,
                    "distance": min_dist,
                    "confidence": candidate_conf,
                    "vote_score": float(score),
                    "vote_share": vote_share,
                }
            )
        candidates.sort(key=lambda item: (item["distance"], -item["vote_score"]))
        neighbors.sort(key=lambda item: item["distance"])
        response = {
            "person": recognized,
            "best_candidate": best_name,
            "distance": float(best_distance),
            "centroid_distance": float(centroid_distance) if centroid_distance is not None else None,
            "confidence": confidence,
            "threshold": float(used_threshold),
            "identity_threshold": float(identity_threshold),
            "global_threshold": float(global_threshold),
            "vote_ratio": vote_ratio,
            "candidates": candidates[:5],
            "knn_neighbors": neighbors[:k],
        }
        return response

    @staticmethod
    def _distance(a: Union[Tensor, np.ndarray], b: Union[Tensor, np.ndarray]) -> float:
        """Calculate L2 distance between two embeddings"""
        # Convert numpy arrays to tensors if needed
        if isinstance(a, np.ndarray):
            a = torch.from_numpy(a)
        if isinstance(b, np.ndarray):
            b = torch.from_numpy(b)
        
        diff = a - b
        return float(torch.sum(diff * diff, dim=-1).mean().item())

    def assess_face_quality(self, image: Union[np.ndarray, Image.Image]) -> Dict[str, float]:
        """
        Assess face crop quality for filtering low-quality samples.
        
        Returns:
            Dict with 'blur', 'brightness', 'contrast', 'quality_score' (0-1)
        """
        import cv2
        
        # Convert to numpy array if PIL Image
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Convert RGB to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 1. Blur detection using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(100.0, laplacian_var / 10.0)  # Normalize to 0-100
        
        # 2. Brightness (mean intensity)
        brightness = float(gray.mean())
        
        # 3. Contrast (standard deviation)
        contrast = float(gray.std())
        
        # 4. Combined quality score (0-1)
        quality = 0.0
        
        # Blur component (40%): penalize if blur_score < 50
        quality += 0.4 * (1.0 if blur_score > 50 else blur_score / 50.0)
        
        # Brightness component (30%): optimal range 80-180
        brightness_score = 1.0 - min(abs(brightness - 130) / 130.0, 1.0)
        quality += 0.3 * brightness_score
        
        # Contrast component (30%): penalize if contrast < 30
        contrast_score = 1.0 if contrast > 30 else contrast / 30.0
        quality += 0.3 * contrast_score
        
        return {
            'blur': float(blur_score),
            'brightness': float(brightness),
            'contrast': float(contrast),
            'quality_score': float(max(0.0, min(1.0, quality))),
        }

    def calibrate_confidence(
        self,
        distance: float,
        threshold: float,
        quality_score: float = 1.0,
        stability_score: float = 1.0,
    ) -> float:
        """
        Calibrate confidence score based on multiple factors.
        
        Args:
            distance: Embedding distance to match
            threshold: Recognition threshold
            quality_score: Face quality score (0-1)
            stability_score: Temporal stability score (0-1)
            
        Returns:
            Calibrated confidence (0-1)
        """
        # Base confidence from distance
        base_conf = 1.0 / (1.0 + distance)
        
        # Margin factor: boost if well below threshold
        margin = threshold - distance
        if margin > 0:
            margin_factor = 1.0 + 0.3 * (margin / threshold)
        else:
            margin_factor = 0.7
        
        # Combine all factors
        calibrated = base_conf * margin_factor * quality_score * stability_score
        
        return float(max(0.0, min(1.0, calibrated)))

