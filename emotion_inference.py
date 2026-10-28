"""Inference-only DAN emotion classification for the validated selected face.

The caller supplies an untouched BGR frame and an AttentionFrame. Holistic
landmarks are normalized to that frame's selected-person crop, whose rectangle
is (left, top, width, height). No dataset, training or model download is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, isfinite
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from networks.dan import DAN


@dataclass(frozen=True)
class EmotionResult:
    """0=neutral, 1=positive, 2=missing; face_bbox uses original-frame xyxy."""

    label: int
    probability: float | None = None
    face_bbox: tuple[int, int, int, int] | None = None
    detail: str = ""


class DanEmotionPredictor:
    def __init__(self, weights: str | Path, device: str = "auto", *,
                 min_face_pixels: int = 24, min_face_points: int = 100,
                 face_padding: float = 0.1):
        weights = Path(weights).expanduser().resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"DAN weights not found: {weights}")
        if device not in ("auto", "cpu", "cuda"):
            raise ValueError("device must be auto, cpu or cuda")
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if not isinstance(min_face_pixels, int) or min_face_pixels < 1:
            raise ValueError("min_face_pixels must be a positive integer")
        if not isinstance(min_face_points, int) or min_face_points < 3:
            raise ValueError("min_face_points must be an integer of at least three")
        if not isfinite(face_padding) or not 0 <= face_padding <= 0.5:
            raise ValueError("face_padding must lie between zero and 0.5")
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available()
                                   else "cpu" if device == "auto" else device)
        self.min_face_pixels = min_face_pixels
        self.min_face_points = min_face_points
        self.face_padding = float(face_padding)
        # Match the original experiment's float32 evaluation settings.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = DAN(num_class=2, num_head=4, pretrained=False)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(device=self.device, dtype=torch.float32).eval()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def _face_box(self, frame: np.ndarray, attention_frame: Any):
        tracking = attention_frame.tracking
        if (tracking.state != "locked" or tracking.target_id is None
                or tracking.pose is None):
            return None, "target_not_trustworthy"
        target = next((item for item in tracking.tracks
                       if item.track_id == tracking.target_id), None)
        if target is None or not target.observed:
            return None, "target_not_observed"
        result, rectangle = attention_frame.holistic, attention_frame.crop_rect
        face = getattr(result, "face_landmarks", None)
        if face is None or rectangle is None:
            return None, "no_selected_face"
        try:
            bounds = np.asarray(rectangle, dtype=float)
            target_box = np.asarray(target.bbox, dtype=float)
            if (bounds.shape != (4,) or target_box.shape != (4,)
                    or not np.isfinite(np.r_[bounds, target_box]).all()
                    or not np.equal(bounds, np.floor(bounds)).all()):
                return None, "invalid_face_crop"
            left, top, width, height = bounds.astype(int)
            frame_height, frame_width = frame.shape[:2]
            if (left < 0 or top < 0 or width < 1 or height < 1
                    or left + width > frame_width or top + height > frame_height
                    or target_box[2] <= target_box[0] or target_box[3] <= target_box[1]):
                return None, "invalid_face_crop"
            points = np.asarray([(point.x, point.y) for point in face.landmark], dtype=float)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < self.min_face_points:
                return None, "insufficient_face_landmarks"
            valid = np.isfinite(points).all(axis=1) & (points >= 0).all(axis=1) & (points <= 1).all(axis=1)
            # Reject a malformed mesh instead of allowing a few stray points to
            # turn a whole-person rectangle into a purported face crop.
            if valid.sum() < self.min_face_points or valid.mean() < 0.9:
                return None, "invalid_face_landmarks"
            pixels = points[valid] * [width, height] + [left, top]
            low, high = pixels.min(axis=0), pixels.max(axis=0)
            size = high - low
            if np.any(size < self.min_face_pixels):
                return None, "face_too_small"
            center = (low + high) / 2
            if not (target_box[0] <= center[0] <= target_box[2]
                    and target_box[1] <= center[1] <= target_box[3]):
                return None, "face_outside_selected_target"
            # The face must agree with Holistic's own visible nose when one is
            # available. The analyzer already verifies its shoulders against
            # the independently validated selected-person Tasks pose.
            pose = getattr(result, "pose_landmarks", None)
            if pose is not None and len(pose.landmark):
                nose = pose.landmark[0]
                if getattr(nose, "visibility", 0.0) >= 0.7:
                    nose_pixel = np.array([nose.x * width + left, nose.y * height + top])
                    if (not np.isfinite(nose_pixel).all()
                            or np.any(nose_pixel < low - size * 0.2)
                            or np.any(nose_pixel > high + size * 0.2)):
                        return None, "face_pose_mismatch"
            side = float(size.max()) * (1 + 2 * self.face_padding)
            x1 = max(left, floor(center[0] - side / 2))
            y1 = max(top, floor(center[1] - side / 2))
            x2 = min(left + width, ceil(center[0] + side / 2))
            y2 = min(top + height, ceil(center[1] + side / 2))
            if min(x2 - x1, y2 - y1) < self.min_face_pixels:
                return None, "face_too_small"
            return (x1, y1, x2, y2), "selected_face"
        except (AttributeError, IndexError, TypeError, ValueError, OverflowError):
            return None, "invalid_face_landmarks"

    def process(self, frame: np.ndarray, attention_frame: Any) -> EmotionResult:
        if (not isinstance(frame, np.ndarray) or frame.ndim != 3
                or frame.shape[2] != 3 or frame.dtype != np.uint8
                or not frame.shape[0] or not frame.shape[1]):
            raise ValueError("frame must be a nonempty uint8 BGR image")
        box, detail = self._face_box(frame, attention_frame)
        if box is None:
            return EmotionResult(2, detail=detail)
        x1, y1, x2, y2 = box
        rgb = np.ascontiguousarray(frame[y1:y2, x1:x2, ::-1])
        image = self.transform(Image.fromarray(rgb)).unsqueeze(0).to(
            device=self.device, dtype=torch.float32)
        with torch.inference_mode():
            logits, _, _ = self.model(image)
            if logits.shape != (1, 2) or not torch.isfinite(logits).all():
                raise RuntimeError("DAN returned invalid binary logits")
            probability = float(torch.softmax(logits.float(), dim=1)[0, 1].item())
        return EmotionResult(int(probability > 0.5), probability, box,
                             "positive" if probability > 0.5 else "neutral")
