"""Identity-isolated summaries in fixed, source-time five-second windows.

Zero means the requested threshold was not reached.  It is not evidence that
an action was absent: consult the per-module valid/unknown counts.  Sample
intervals describe actual observations, never an assumed five-second duration.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
import math
from typing import Any


WINDOW_SECONDS = 5
BODY_FIELDS = ("head_shake", "both_wave", "table_rest", "seat_leave")
PRIMARY_FIELDS = ("toy_interaction", "emotion", *BODY_FIELDS)
WINDOW_SPEC = {
    "window_seconds": 5,
    "window_alignment": "source time; half-open [0,5), [5,10), ...",
    "group_by": ["target_id", "selection_generation"],
    "toy_trigger_frames_per_source_second": 3,
    "toy_interactive_seconds_per_window": 3,
    "body_active_frames_per_window": 5,
    "emotion_labels": {"0": "neutral", "1": "positive", "2": "missing"},
    "emotion_valid_ratio_required": "> 0.5 of this identity group's observed frames",
    "emotion_positive_frames_required": 5,
    "frame_counting": "one fresh observation per unique source frame; never repeat cached inference",
    "body_zero": "threshold not reached; inspect valid and unknown frame counts",
    "unknown_body_states": ["unknown", "calibrating", "candidate"],
    "observed_interval": "minimum and maximum actual sample timestamps; see observation_segments_json",
    "partial": "clip boundary, source gap, or identity group covering only part of a window",
    "missing_windows": "never synthesized",
}
SPECIFICATION = WINDOW_SPEC

FIELDNAMES = (
    "window_start_seconds", "window_end_seconds", "target_id", "selection_generation",
    *PRIMARY_FIELDS,
    "observed_start_seconds", "observed_end_seconds", "observed_span_seconds",
    "first_source_frame_index", "last_source_frame_index", "frame_count",
    "Partial_Window", "selection_partial", "source_gap", "unselected",
    "observation_segments_json",
    "toy_trigger_frames", "toy_valid_frames", "toy_unknown_frames",
    "toy_interactive_seconds", "toy_second_counts_json",
    "emotion_neutral_frames", "emotion_positive_frames", "emotion_valid_frames",
    "emotion_missing_frames", "emotion_valid_ratio",
    *(name for behavior in BODY_FIELDS for name in
      (f"{behavior}_active_frames", f"{behavior}_valid_frames", f"{behavior}_unknown_frames")),
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _body_value(value: Any) -> bool | None:
    if value is True or value == "active":
        return True
    if value is False or value == "normal":
        return False
    if value is None or value in ("unknown", "calibrating", "candidate"):
        return None
    raise ValueError(f"unsupported behavior status: {value!r}")


@dataclass
class _Group:
    target_id: int | None
    generation: int
    first_ms: float
    last_ms: float
    first_index: int | None
    last_index: int | None
    frames: int = 0
    toy_true: int = 0
    toy_valid: int = 0
    neutral: int = 0
    positive: int = 0
    body_active: dict = field(default_factory=lambda: dict.fromkeys(BODY_FIELDS, 0))
    body_valid: dict = field(default_factory=lambda: dict.fromkeys(BODY_FIELDS, 0))
    seconds: dict = field(default_factory=dict)
    segments: list = field(default_factory=list)

    def add(self, timestamp_ms, frame_index, playing, emotion, statuses, new_segment):
        self.frames += 1
        self.last_ms, self.last_index = timestamp_ms, frame_index
        if new_segment or not self.segments:
            self.segments.append({
                "start_seconds": timestamp_ms / 1000,
                "end_seconds": timestamp_ms / 1000,
                "first_source_frame_index": frame_index,
                "last_source_frame_index": frame_index,
                "frame_count": 0,
            })
        segment = self.segments[-1]
        segment["end_seconds"] = timestamp_ms / 1000
        segment["last_source_frame_index"] = frame_index
        segment["frame_count"] += 1
        second = math.floor(timestamp_ms / 1000)
        counts = self.seconds.setdefault(second, {"trigger_frames": 0, "valid_frames": 0,
                                                  "unknown_frames": 0})
        if playing is None:
            counts["unknown_frames"] += 1
        else:
            self.toy_valid += 1
            self.toy_true += int(playing)
            counts["valid_frames"] += 1
            counts["trigger_frames"] += int(playing)
        self.neutral += int(emotion == 0)
        self.positive += int(emotion == 1)
        for name, status in statuses.items():
            if status is not None:
                self.body_valid[name] += 1
                self.body_active[name] += int(status)

    def row(self, bucket, total_frames, source_partial, source_gap):
        valid = self.neutral + self.positive
        # Integer arithmetic keeps the exact-half boundary unambiguous.
        emotion = (int(self.positive >= 5) if valid * 2 > self.frames else 2)
        second_counts = {
            str(second): {**counts, "interactive": int(counts["trigger_frames"] >= 3)}
            for second, counts in sorted(self.seconds.items())
        }
        interactive_seconds = sum(item["interactive"] for item in second_counts.values())
        selection_partial = self.frames != total_frames or self.target_id is None
        row = {
            "window_start_seconds": bucket * WINDOW_SECONDS,
            "window_end_seconds": (bucket + 1) * WINDOW_SECONDS,
            "target_id": self.target_id,
            "selection_generation": self.generation,
            "toy_interaction": int(interactive_seconds >= 3),
            "emotion": emotion,
            "observed_start_seconds": self.first_ms / 1000,
            "observed_end_seconds": self.last_ms / 1000,
            "observed_span_seconds": (self.last_ms - self.first_ms) / 1000,
            "first_source_frame_index": self.first_index,
            "last_source_frame_index": self.last_index,
            "frame_count": self.frames,
            "Partial_Window": bool(source_partial or selection_partial),
            "selection_partial": selection_partial,
            "source_gap": source_gap,
            "unselected": self.target_id is None,
            "observation_segments_json": _json(self.segments),
            "toy_trigger_frames": self.toy_true,
            "toy_valid_frames": self.toy_valid,
            "toy_unknown_frames": self.frames - self.toy_valid,
            "toy_interactive_seconds": interactive_seconds,
            "toy_second_counts_json": _json(second_counts),
            "emotion_neutral_frames": self.neutral,
            "emotion_positive_frames": self.positive,
            "emotion_valid_frames": valid,
            "emotion_missing_frames": self.frames - valid,
            "emotion_valid_ratio": valid / self.frames,
        }
        for name in BODY_FIELDS:
            row[name] = int(self.body_active[name] >= 5)
            row[f"{name}_active_frames"] = self.body_active[name]
            row[f"{name}_valid_frames"] = self.body_valid[name]
            row[f"{name}_unknown_frames"] = self.frames - self.body_valid[name]
        return row


class WindowLog:
    """Write CSV to a caller-owned text stream; the caller supplies UTF-8 BOM.

    ``add`` takes fresh frame results and returns rows completed by this call.
    ``finish`` flushes the final bucket once and never closes the stream.
    ``end_seconds`` is the exclusive source endpoint, usually the final frame's
    source time plus its duration.  With no endpoint the last sample time is
    used conservatively, so the final bucket is marked partial.

    A source gap larger than ``max_gap_seconds`` marks both affected buckets
    partial and starts a new observation segment. Missing buckets are omitted.
    Results for a repeated identity within the same bucket share a row, while
    its disjoint actual observations remain explicit in the segment JSON.
    """

    def __init__(self, stream, *, max_gap_seconds=1.0):
        if not math.isfinite(max_gap_seconds) or max_gap_seconds <= 0:
            raise ValueError("max_gap_seconds must be finite and positive")
        self.stream = stream
        self.max_gap_ms = float(max_gap_seconds) * 1000
        self.writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        self.rows: list[dict] = []
        self._bucket = None
        self._groups = {}
        self._bucket_frames = 0
        self._source_partial = False
        self._source_gap = False
        self._last_ms = None
        self._last_index = None
        self._last_key = None
        self._finished = False
        self.writer.writeheader()
        self.stream.flush()

    def _emit(self, *, end_seconds=None):
        if self._bucket is None:
            return []
        partial = self._source_partial or self._source_gap
        if end_seconds is not None and end_seconds < (self._bucket + 1) * WINDOW_SECONDS:
            partial = True
        emitted = [group.row(self._bucket, self._bucket_frames, partial, self._source_gap)
                   for group in self._groups.values()]
        for row in emitted:
            self.writer.writerow(row)
        self.stream.flush()
        self.rows.extend(emitted)
        return emitted

    def add(self, timestamp_ms, target_id, selection_generation, playing, emotion, statuses,
            *, source_frame_index=None):
        if self._finished:
            raise RuntimeError("cannot add after finish()")
        if (isinstance(timestamp_ms, bool) or not isinstance(timestamp_ms, (int, float))
                or not math.isfinite(timestamp_ms) or timestamp_ms < 0
                or (self._last_ms is not None and timestamp_ms <= self._last_ms)):
            raise ValueError("source timestamps must be finite, nonnegative and strictly increasing")
        if (source_frame_index is not None and
                (isinstance(source_frame_index, bool) or not isinstance(source_frame_index, int)
                 or source_frame_index < 0
                 or (self._last_index is not None and source_frame_index <= self._last_index))):
            raise ValueError("source frame indices must be nonnegative and strictly increasing")
        if target_id is not None and (isinstance(target_id, bool) or not isinstance(target_id, int)):
            raise ValueError("target_id must be an integer or None")
        if (isinstance(selection_generation, bool) or not isinstance(selection_generation, int)
                or selection_generation < 0):
            raise ValueError("selection_generation must be a nonnegative integer")
        if playing is not True and playing is not False and playing is not None:
            raise ValueError("playing must be bool or None; multiple toy detections must be reduced first")
        if isinstance(emotion, bool) or not isinstance(emotion, int) or emotion not in (0, 1, 2):
            raise ValueError("emotion must be 0, 1 or 2")
        states = {name: _body_value((statuses or {}).get(name)) for name in BODY_FIELDS}
        if target_id is None:
            playing, emotion = None, 2
            states = dict.fromkeys(BODY_FIELDS)
        timestamp_ms = float(timestamp_ms)
        bucket = math.floor(timestamp_ms / (WINDOW_SECONDS * 1000))
        key = target_id, selection_generation
        gap = self._last_ms is not None and timestamp_ms - self._last_ms > self.max_gap_ms
        emitted = []
        bucket_changed = bucket != self._bucket
        if bucket_changed:
            if self._bucket is not None:
                self._source_gap |= gap
                emitted = self._emit()
            self._bucket = bucket
            self._groups = {}
            self._bucket_frames = 0
            self._source_gap = gap
            self._source_partial = self._last_ms is None and timestamp_ms > bucket * WINDOW_SECONDS * 1000
        else:
            self._source_gap |= gap
        if key not in self._groups:
            self._groups[key] = _Group(target_id, selection_generation, timestamp_ms,
                                       timestamp_ms, source_frame_index, source_frame_index)
        self._groups[key].add(timestamp_ms, source_frame_index, playing, emotion, states,
                              bucket_changed or gap or key != self._last_key)
        self._bucket_frames += 1
        self._last_ms, self._last_key = timestamp_ms, key
        if source_frame_index is not None:
            self._last_index = source_frame_index
        return emitted

    def finish(self, end_seconds=None):
        if self._finished:
            return []
        if end_seconds is not None and (
                not isinstance(end_seconds, (int, float)) or isinstance(end_seconds, bool)
                or not math.isfinite(end_seconds) or end_seconds < 0
                or (self._last_ms is not None and end_seconds * 1000 < self._last_ms)):
            raise ValueError("end_seconds must be finite and no earlier than the last observation")
        endpoint = self._last_ms / 1000 if end_seconds is None and self._last_ms is not None else end_seconds
        emitted = self._emit(end_seconds=endpoint)
        self._finished = True
        return emitted


FiveSecondWindowLog = WindowLog
