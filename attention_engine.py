"""Target-only hand/toy analysis on the shoulder-selected OC-SORT person.

Person detection, shoulder selection and identity recovery remain in the copied
``medispipe_final`` tracking pipeline. Holistic receives only the selected
person's crop, and its shoulders must agree with the validated Tasks pose.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite
from typing import Any

import cv2
import mediapipe as mp
import numpy as np

from target_selection import crop_person


def calculate_iou(box_a, box_b) -> float:
    """Return intersection over union for original-image xyxy rectangles."""
    a, b = np.asarray(box_a, dtype=float), np.asarray(box_b, dtype=float)
    if a.shape != (4,) or b.shape != (4,) or not np.isfinite(np.r_[a, b]).all():
        return 0.0
    intersection = np.maximum(0.0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2]))
    intersection_area = float(np.prod(intersection))
    area_a = float(np.prod(np.maximum(0.0, a[2:] - a[:2])))
    area_b = float(np.prod(np.maximum(0.0, b[2:] - b[:2])))
    union = area_a + area_b - intersection_area
    return intersection_area / union if union > 0 else 0.0


def get_hand_bbox(landmarks, img_w: int, img_h: int, padding: float = 0.05) -> list[int]:
    """Convert crop-normalized hand landmarks to a clipped crop xyxy box."""
    if img_w < 1 or img_h < 1 or not isfinite(padding) or padding < 0:
        raise ValueError("image dimensions must be positive and padding nonnegative")
    points = np.array([(item.x * img_w, item.y * img_h) for item in landmarks.landmark])
    if points.size == 0 or not np.isfinite(points).all():
        raise ValueError("hand landmarks must contain finite coordinates")
    low, high = points.min(axis=0), points.max(axis=0)
    margin = (high - low) * padding
    low = np.clip(low - margin, [0, 0], [img_w - 1, img_h - 1])
    high = np.clip(high + margin, [0, 0], [img_w - 1, img_h - 1])
    return [int(low[0]), int(low[1]), int(high[0]), int(high[1])]


@dataclass
class AttentionFrame:
    tracking: Any
    toy_boxes: list[list[int]]
    hand_boxes: list[list[int]]
    holistic: Any | None
    crop_rect: tuple[int, int, int, int] | None
    playing: bool | None
    detail: str
    target_bbox: np.ndarray | None

    @property
    def status(self) -> str:
        if self.playing is None:
            return "unknown"
        return "playing" if self.playing else "none"


class AttentionAnalyzer:
    """Own target-specific Holistic state and clear it whenever identity changes.

    ``toy_model`` and ``tracking_pipeline`` are injected so execution and identity
    behavior can be tested without loading neural networks. The analyzer owns
    the supplied tracking pipeline and closes it when its context exits.
    """

    def __init__(self, toy_model, tracking_pipeline, fps: float, *,
                 iou_threshold: float = 0.0, history_seconds: float = 0.5,
                 holistic_factory=None):
        if not isfinite(fps) or fps <= 0:
            raise ValueError("fps must be finite and positive")
        if not isfinite(iou_threshold) or not 0 <= iou_threshold <= 1:
            raise ValueError("iou_threshold must lie between zero and one")
        if not isfinite(history_seconds) or history_seconds <= 0:
            raise ValueError("history_seconds must be finite and positive")
        self.toy_model = toy_model
        self.tracking_pipeline = tracking_pipeline
        self.iou_threshold = float(iou_threshold)
        self.history_ms = history_seconds * 1000
        self._history = deque(maxlen=max(1, int(fps * history_seconds)))
        self._holistic_factory = holistic_factory or mp.solutions.holistic.Holistic
        self._holistic = None
        self._target_key = None
        self._frame_index = 0
        self._last_timestamp = None
        self._closed = False

    def _discard_hands(self):
        self._history.clear()
        self._target_key = None
        instance, self._holistic = self._holistic, None
        if instance is not None:
            instance.close()

    def reset_selection(self):
        """Restart shoulder selection, retaining the pipeline's OC-SORT clock."""
        self._discard_hands()
        self.tracking_pipeline.reset_selection()

    def select_target(self, track_id: int):
        selected = self.tracking_pipeline.select_target(track_id)
        self._discard_hands()
        return selected

    def _toy_boxes(self, frame) -> list[list[int]]:
        height, width = frame.shape[:2]
        boxes = []
        # Model input is the untouched frame, before any display annotations.
        for result in self.toy_model(frame, verbose=False):
            if result.boxes is None:
                continue
            for box in result.boxes:
                xyxy = box.xyxy[0]
                if hasattr(xyxy, "cpu"):
                    xyxy = xyxy.cpu().numpy()
                values = np.asarray(xyxy, dtype=float).reshape(-1)
                if values.size != 4 or not np.isfinite(values).all():
                    continue
                values = np.clip(values, [0, 0, 0, 0], [width - 1, height - 1, width - 1, height - 1])
                bounds = values.astype(int).tolist()
                if bounds[2] > bounds[0] and bounds[3] > bounds[1]:
                    boxes.append(bounds)
        return boxes

    def _shoulders_match(self, result, rectangle, reference_pose) -> bool:
        pose = result.pose_landmarks
        if pose is None or len(pose.landmark) < 13:
            return False
        reference = np.asarray(reference_pose.points, dtype=float)
        if reference.ndim != 2 or reference.shape[0] < 13 or reference.shape[1] != 2:
            return False
        expected = reference[[11, 12]]
        left, top, width, height = rectangle
        shoulders = [pose.landmark[index] for index in (11, 12)]
        confidence = getattr(self.tracking_pipeline.config, "landmark_confidence", 0.7)
        visibility = np.array([item.visibility for item in shoulders], dtype=float)
        actual = np.array([(left + item.x * width, top + item.y * height) for item in shoulders])
        if (not np.isfinite(np.r_[actual.ravel(), expected.ravel(), visibility]).all()
                or np.any(visibility < confidence)):
            return False
        # Legacy Holistic does not supply Tasks-style presence confidence. Its
        # visibility and agreement with the independently validated Tasks pose
        # are checked explicitly; missing presence is never fabricated.
        shoulder_width = float(np.linalg.norm(expected[0] - expected[1]))
        if shoulder_width <= 0:
            return False
        tolerance = max(20.0, 0.5 * shoulder_width)
        return bool(np.all(np.linalg.norm(actual - expected, axis=1) <= tolerance))

    def process(self, frame, timestamp_ms: int, frame_index: int | None = None) -> AttentionFrame:
        if self._closed:
            raise RuntimeError("AttentionAnalyzer is closed")
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be a BGR image")
        if not isfinite(timestamp_ms) or (self._last_timestamp is not None and timestamp_ms <= self._last_timestamp):
            raise ValueError("frame timestamps must be finite and strictly increasing")
        self._last_timestamp = timestamp_ms
        tracking = self.tracking_pipeline.process(
            frame, timestamp_ms, self._frame_index if frame_index is None else frame_index)
        self._frame_index += 1
        toys = self._toy_boxes(frame)
        target = next((track for track in tracking.tracks if track.track_id == tracking.target_id), None)
        target_bbox = np.asarray(target.bbox, dtype=float).copy() if target is not None else None

        def unknown(detail):
            self._discard_hands()
            return AttentionFrame(tracking, toys, [], None, None, None, detail, target_bbox)

        if (tracking.state != "locked" or tracking.pose is None or tracking.target_id is None
                or target is None or not target.observed):
            return unknown(tracking.detail)
        key = (tracking.target_id, tracking.selection_generation)
        if key != self._target_key:
            self._discard_hands()
            self._target_key = key
        padding = getattr(self.tracking_pipeline.config, "crop_padding", 0.08)
        cropped = crop_person(frame, target.bbox, padding)
        if cropped is None:
            return unknown("invalid_target_crop")
        crop, rectangle = cropped
        try:
            if self._holistic is None:
                self._holistic = self._holistic_factory(
                    static_image_mode=False, model_complexity=1, smooth_landmarks=True,
                    refine_face_landmarks=True, min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            result = self._holistic.process(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            if not self._shoulders_match(result, rectangle, tracking.pose):
                return unknown("holistic_target_shoulders_mismatch")
            left, top, width, height = rectangle
            hands = []

            def invalid_hands():
                # Hand geometry cannot invalidate an independently verified
                # face from the same selected-person result.
                self._history.clear()
                return AttentionFrame(tracking, toys, [], result, rectangle,
                                      None, "invalid_hand_landmarks", target_bbox)

            for hand in (result.left_hand_landmarks, result.right_hand_landmarks):
                if hand is None:
                    continue
                try:
                    box = get_hand_bbox(hand, width, height, padding=0.05)
                except (TypeError, ValueError, OverflowError):
                    return invalid_hands()
                if box[2] <= box[0] or box[3] <= box[1]:
                    return invalid_hands()
                hands.append([box[0] + left, box[1] + top, box[2] + left, box[3] + top])
        except (RuntimeError, ValueError) as error:
            return unknown(f"holistic_inference_error: {error}")

        while self._history and timestamp_ms - self._history[0][0] >= self.history_ms:
            self._history.popleft()
        self._history.append((timestamp_ms, toys))
        playing = (any(calculate_iou(hand, toy) > self.iou_threshold
                       for hand in hands for _, history_toys in self._history for toy in history_toys)
                   if hands else None)
        detail = "hand_toy_overlap" if playing else ("no_hand_toy_overlap" if hands else "no_hands_detected")
        return AttentionFrame(tracking, toys, hands, result, rectangle, playing, detail, target_bbox)

    def close(self):
        if not self._closed:
            self._closed = True
            try:
                self._discard_hands()
            finally:
                self.tracking_pipeline.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
