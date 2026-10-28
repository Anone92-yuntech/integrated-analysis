"""Video-wide seat reference from ten seconds of stable observed boxes.

Calibration uses real detector matches without a pose-quality gate. A frozen
reference survives target reselection. All deadlines use source-video time.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np
from target_selection import MappedPose, TrackObservation


@dataclass(frozen=True)
class SeatConfig:
    baseline_seconds: float = 10.0
    min_samples: int = 10
    max_displacement_ratio: float = 0.25
    manual_window_seconds: float = 10.0
    # Legacy compatibility; departure and return are immediate IoU rules.
    departure_ratio: float = 1.0
    hold_seconds: float = 1.0
    return_seconds: float = 0.7
    max_gap_seconds: float = 0.25
    confidence: float = 0.7
    iou_threshold: float = 0.5

    def __post_init__(self) -> None:
        for name in ("baseline_seconds", "max_displacement_ratio",
                     "manual_window_seconds", "departure_ratio", "hold_seconds",
                     "return_seconds", "max_gap_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if (not isinstance(self.min_samples, int) or isinstance(self.min_samples, bool)
                or self.min_samples < 1):
            raise ValueError("min_samples must be a positive integer")
        if not isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must lie between zero and one")
        if not isfinite(self.iou_threshold) or not 0 < self.iou_threshold <= 1:
            raise ValueError("iou_threshold must be finite and in (0,1]")


@dataclass(frozen=True)
class SeatResult:
    state: str
    detail: str
    displacement_ratio: float | None = None
    held_seconds: float = 0.0
    seat_roi: tuple[float, float, float, float] | None = None
    baseline_anchor: tuple[float, float] | None = None
    iou: float | None = None
    calibration_seconds: float = 0.0
    calibration_samples: int = 0
    manual_remaining_seconds: float = 0.0
    reference_source: str | None = None


class SeatDetector:
    """Freeze mean bbox corners, then compare each observed box by IoU.

    Stable centers stay within max_displacement_ratio times the initial box
    width of the first center, preventing slow drift through small frame steps.
    The boundary is inclusive. An optional constructor ROI is used directly.
    Later manual boxes require the interval [cancel, cancel + 10 seconds).
    Return after departure additionally needs trusted head/shoulder points.
    """

    def __init__(self, config: SeatConfig | None = None, roi: Any = None) -> None:
        self.config = config or SeatConfig()
        self._last_timestamp_ms: int | None = None
        self._control_timestamp_ms: int | None = None
        self._identity_key: Any = None
        self._identity_initialized = False
        self._manual_started_ms: int | None = None
        self._manual_deadline_ms: float | None = None
        self._seat_roi: np.ndarray | None = None
        self._baseline_anchor: np.ndarray | None = None
        self._reference_source: str | None = None
        self._departure_started_ms: int | None = None
        self._away = False
        self._clear_calibration()
        if roi is not None:
            box = self._manual_box(roi)
            if box is None:
                raise ValueError("roi must be a finite positive original-pixel rectangle")
            self._freeze(box, "initial_manual")

    @staticmethod
    def _timestamp(value: Any) -> int:
        if (not isinstance(value, (int, np.integer))
                or isinstance(value, (bool, np.bool_)) or value < 0):
            raise ValueError("timestamp_ms must be a nonnegative integer")
        return int(value)

    def _control_time(self, timestamp_ms: int) -> int:
        timestamp_ms = self._timestamp(timestamp_ms)
        for previous in (self._last_timestamp_ms, self._control_timestamp_ms):
            if previous is not None and timestamp_ms < previous:
                raise ValueError("seat controls cannot precede the latest source time")
        return timestamp_ms

    def reset(self, *, preserve_reference: bool = True) -> None:
        """Reset subject history while preserving the video's fixed seat.

        A new video normally uses a new detector. An explicit full reset also
        clears cancellation deadlines and permits the source clock to restart.
        """
        self._clear_calibration()
        self._identity_key = None
        self._identity_initialized = False
        self._away = False
        self._departure_started_ms = None
        if not preserve_reference:
            self._seat_roi = None
            self._baseline_anchor = None
            self._reference_source = None
            self._manual_started_ms = None
            self._manual_deadline_ms = None
            self._last_timestamp_ms = None
            self._control_timestamp_ms = None

    def cancel(self, timestamp_ms: int) -> None:
        """Clear the seat and open a source-time manual replacement window."""
        timestamp_ms = self._control_time(timestamp_ms)
        self._control_timestamp_ms = timestamp_ms
        self._seat_roi = None
        self._baseline_anchor = None
        self._reference_source = None
        self._away = False
        self._departure_started_ms = None
        self._clear_calibration()
        self._manual_started_ms = timestamp_ms
        self._manual_deadline_ms = timestamp_ms + self.config.manual_window_seconds * 1000

    @property
    def baseline_ready(self) -> bool:
        return self._seat_roi is not None

    @property
    def seat_roi(self) -> tuple[float, float, float, float] | None:
        return None if self._seat_roi is None else tuple(map(float, self._seat_roi))

    @property
    def baseline_scale(self) -> None:
        """Legacy accessor: the IoU rule has no shoulder scale."""
        return None

    @property
    def manual_deadline_ms(self) -> float | None:
        return self._manual_deadline_ms

    def manual_available(self, timestamp_ms: int) -> bool:
        timestamp_ms = self._timestamp(timestamp_ms)
        return (not self.baseline_ready and self._manual_started_ms is not None
                and self._manual_started_ms <= timestamp_ms < self._manual_deadline_ms)

    def manual_remaining_seconds(self, timestamp_ms: int) -> float:
        timestamp_ms = self._timestamp(timestamp_ms)
        if not self.manual_available(timestamp_ms):
            return 0.0
        return max(0.0, (self._manual_deadline_ms - timestamp_ms) / 1000)

    def set_manual_roi(self, roi: Any, timestamp_ms: int,
                       frame_shape: Any = None) -> bool:
        """Immediately freeze a valid box within the cancellation window.

        Invalid, cancelled, or out-of-frame selections leave state unchanged.
        """
        timestamp_ms = self._control_time(timestamp_ms)
        if not self.manual_available(timestamp_ms):
            return False
        box = self._manual_box(roi, frame_shape)
        if box is None:
            return False
        self._control_timestamp_ms = timestamp_ms
        self._freeze(box, "manual")
        self._away = False
        self._departure_started_ms = None
        self._manual_started_ms = None
        self._manual_deadline_ms = None
        return True

    @staticmethod
    def _box(value: Any) -> np.ndarray | None:
        try:
            box = np.asarray(value, dtype=float).reshape(-1)
        except (ValueError, TypeError):
            return None
        if (box.size != 4 or not np.isfinite(box).all()
                or np.any(box[2:] <= box[:2])):
            return None
        return box.copy()

    @staticmethod
    def _frame_box(frame_shape: Any) -> np.ndarray | None:
        try:
            height, width = float(frame_shape[0]), float(frame_shape[1])
        except (TypeError, IndexError, ValueError):
            return None
        if not isfinite(height) or not isfinite(width) or height <= 0 or width <= 0:
            return None
        return np.array([0.0, 0.0, width, height])

    @classmethod
    def _manual_box(cls, value: Any, frame_shape: Any = None) -> np.ndarray | None:
        box = cls._box(value)
        if box is None or np.any(box[:2] < 0):
            return None
        if frame_shape is not None:
            frame_box = cls._frame_box(frame_shape)
            if frame_box is None or np.any(box[2:] > frame_box[2:]):
                return None
        return box

    @staticmethod
    def _inside(point: np.ndarray, box: np.ndarray, tolerance: float = 0.0) -> bool:
        margin = (box[2:] - box[:2]) * tolerance
        return bool(np.all(point >= box[:2] - margin)
                    and np.all(point <= box[2:] + margin))

    @staticmethod
    def _iou(first: np.ndarray, second: np.ndarray) -> float | None:
        overlap = np.maximum(0.0, np.minimum(first[2:], second[2:])
                             - np.maximum(first[:2], second[:2]))
        intersection = float(np.prod(overlap))
        union = float(np.prod(first[2:] - first[:2])
                      + np.prod(second[2:] - second[:2]) - intersection)
        if not isfinite(union) or union <= 0 or not isfinite(intersection):
            return None
        value = intersection / union
        return value if isfinite(value) else None

    def _point(self, pose: MappedPose, index: int) -> np.ndarray | None:
        try:
            point = np.asarray(pose.points[index], dtype=float)
            visibility = float(pose.visibility[index])
            presence = float(pose.presence[index])
        except (AttributeError, IndexError, TypeError, ValueError):
            return None
        if (point.shape != (2,) or not np.isfinite(point).all()
                or not isfinite(visibility) or not isfinite(presence)
                or min(visibility, presence) < self.config.confidence):
            return None
        return point

    def _head_shoulders_valid(self, pose: MappedPose | None, box: np.ndarray,
                              frame_box: np.ndarray | None) -> bool:
        """Pose quality is needed only for a return after observed departure."""
        if pose is None:
            return False
        head = self._point(pose, 0)
        if head is None:
            face = [self._point(pose, i) for i in range(1, 11)]
            face = [point for point in face if point is not None]
            if len(face) < 3:
                return False
            head = np.median(face, axis=0)
        shoulders = [self._point(pose, i) for i in (11, 12)]
        if any(point is None for point in shoulders):
            return False
        width = float(np.linalg.norm(shoulders[0] - shoulders[1]))
        if width < 4 or width > 1.15 * (box[2] - box[0]):
            return False
        signature = [head, *shoulders]
        return (all(self._inside(point, box, 0.06) for point in signature)
                and (frame_box is None or all(self._inside(point, frame_box) for point in signature)))

    def _clear_calibration(self) -> None:
        self._baseline_started_ms: int | None = None
        self._initial_center: np.ndarray | None = None
        self._initial_width: float | None = None
        self._box_samples: list[np.ndarray] = []
        self._displacement_ratio: float | None = None

    def _freeze(self, box: np.ndarray, source: str) -> None:
        self._seat_roi = box.copy()
        self._baseline_anchor = (box[:2] + box[2:]) / 2
        self._reference_source = source
        self._clear_calibration()

    def _result(self, state: str, detail: str, timestamp_ms: int,
                iou: float | None = None) -> SeatResult:
        anchor = None if self._baseline_anchor is None else tuple(map(float, self._baseline_anchor))
        held = (0.0 if self._departure_started_ms is None
                else max(0.0, (timestamp_ms - self._departure_started_ms) / 1000))
        elapsed = (0.0 if self._baseline_started_ms is None
                   else max(0.0, (timestamp_ms - self._baseline_started_ms) / 1000))
        return SeatResult(state, detail, displacement_ratio=self._displacement_ratio,
                          held_seconds=held, seat_roi=self.seat_roi,
                          baseline_anchor=anchor, iou=iou,
                          calibration_seconds=elapsed,
                          calibration_samples=len(self._box_samples),
                          manual_remaining_seconds=self.manual_remaining_seconds(timestamp_ms),
                          reference_source=self._reference_source)

    def _unknown(self, detail: str, timestamp_ms: int) -> SeatResult:
        self._departure_started_ms = None
        if not self.baseline_ready:
            self._clear_calibration()
        return self._result("unknown", detail, timestamp_ms)

    def update(self, pose: MappedPose | None, track: TrackObservation | None,
               timestamp_ms: int, baseline_scale: float | None = None,
               frame_shape: Any = None, identity_key: Any = None) -> SeatResult:
        # baseline_scale remains accepted for callers of the old shoulder rule.
        timestamp_ms = self._timestamp(timestamp_ms)
        if self._last_timestamp_ms is not None and timestamp_ms <= self._last_timestamp_ms:
            raise ValueError("seat timestamps must increase strictly")
        if self._control_timestamp_ms is not None and timestamp_ms < self._control_timestamp_ms:
            raise ValueError("seat update cannot precede the latest control time")
        gap = None if self._last_timestamp_ms is None else timestamp_ms - self._last_timestamp_ms
        self._last_timestamp_ms = timestamp_ms
        key = (identity_key if identity_key is not None
               else track.track_id if track is not None else self._identity_key)
        if self._identity_initialized and key != self._identity_key:
            self._clear_calibration()
            self._away = False
            self._departure_started_ms = None
        self._identity_key, self._identity_initialized = key, True
        if gap is not None and gap > self.config.max_gap_seconds * 1000:
            self._departure_started_ms = None
            if not self.baseline_ready:
                self._clear_calibration()
        if track is None or not track.observed:
            return self._unknown("target_not_observed", timestamp_ms)
        box = self._box(track.bbox)
        if box is None:
            return self._unknown("invalid_person_box", timestamp_ms)
        frame_box = None if frame_shape is None else self._frame_box(frame_shape)
        if frame_shape is not None:
            if frame_box is None:
                return self._unknown("invalid_frame_shape", timestamp_ms)
            if np.any(np.minimum(box[2:], frame_box[2:]) <= np.maximum(box[:2], frame_box[:2])):
                return self._unknown("person_box_outside_frame", timestamp_ms)
        center = (box[:2] + box[2:]) / 2
        if not self.baseline_ready:
            restarted = False
            if self._initial_center is not None:
                ratio = float(np.linalg.norm(center - self._initial_center) / self._initial_width)
                if ratio > self.config.max_displacement_ratio:
                    self._clear_calibration()
                    restarted = True
            if self._baseline_started_ms is None:
                self._baseline_started_ms = timestamp_ms
                self._initial_center = center.copy()
                self._initial_width = float(box[2] - box[0])
            self._displacement_ratio = float(np.linalg.norm(center - self._initial_center)
                                             / self._initial_width)
            self._box_samples.append(box.copy())
            elapsed_ms = timestamp_ms - self._baseline_started_ms
            stable = (elapsed_ms >= self.config.baseline_seconds * 1000
                      and len(self._box_samples) >= self.config.min_samples)
            if not stable or self.manual_available(timestamp_ms):
                detail = ("calibration_restarted_displacement" if restarted else
                          "awaiting_manual_deadline" if stable else
                          "collecting_stable_observed_boxes")
                return self._result("calibrating", detail, timestamp_ms)
            self._freeze(np.mean(np.asarray(self._box_samples), axis=0), "automatic")
            self._manual_started_ms = None
            self._manual_deadline_ms = None
        iou = self._iou(box, self._seat_roi)
        if iou is None:
            return self._unknown("invalid_box_iou", timestamp_ms)
        if iou < self.config.iou_threshold:
            if self._departure_started_ms is None:
                self._departure_started_ms = timestamp_ms
            self._away = True
            return self._result("away", "box_iou_below_threshold", timestamp_ms, iou)
        if self._away:
            if not self._inside(center, self._seat_roi):
                return self._result("away", "awaiting_box_center_return", timestamp_ms, iou)
            if not self._head_shoulders_valid(pose, box, frame_box):
                return self._result("away", "awaiting_return_pose", timestamp_ms, iou)
            self._away = False
        self._departure_started_ms = None
        return self._result("seated", "observed_box_inside_seat", timestamp_ms, iou)
