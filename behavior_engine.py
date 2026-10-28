"""Combine independent observed behaviors without interpreting emotions."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from motion_detection import MotionConfig, MotionDetector, MotionResult
from seat_detection import SeatConfig, SeatDetector, SeatResult


@dataclass
class BehaviorFrame:
    statuses: dict[str, str]
    details: dict[str, str]
    metrics: dict
    motion: MotionResult
    seat: SeatResult


class BehaviorEngine:
    def __init__(self, motion_config=None, seat_config=None, seat_roi=None):
        self.motion = MotionDetector(motion_config or MotionConfig())
        self.seat_config = seat_config or SeatConfig()
        self.seat_roi = seat_roi
        self.seat = SeatDetector(self.seat_config, roi=seat_roi)
        self._selection_key = None
        self._pose_recoveries = None

    def reset(self):
        self.motion.reset()
        self.seat.reset(preserve_reference=True)
        self.seat_roi = self.seat.seat_roi
        self._selection_key = None
        self._pose_recoveries = None

    def clear_seat(self, timestamp_ms):
        self.seat.cancel(timestamp_ms)
        self.seat_roi = None

    def set_seat_roi(self, roi, timestamp_ms, frame_shape=None):
        accepted = self.seat.set_manual_roi(roi, timestamp_ms, frame_shape)
        if accepted:
            self.seat_roi = self.seat.seat_roi
        return accepted

    def update(self, analysis, frame_shape):
        key = (analysis.selection_generation, analysis.target_id)
        if key != self._selection_key:
            self.motion.reset()
            # The full key below resets only unfinished seat calibration and
            # the identity-specific away latch, preserving the video's seat.
            self._selection_key = key
        elif self._pose_recoveries != analysis.pose_recovery_count:
            # A new VIDEO instance may shift estimates; don't interpret its
            # initial correction as a head turn or fast wrist displacement.
            self.motion.reset()
        self._pose_recoveries = analysis.pose_recovery_count
        posture = analysis.posture
        scale = posture.baseline_shoulder_width if posture is not None else None
        if scale is None and analysis.target_id is not None:
            scale = analysis.statistics.get(analysis.target_id, {}).get('median_width')
        track = next((t for t in analysis.tracks
                      if t.track_id == analysis.target_id and t.observed), None)
        pose = analysis.pose if track is not None and analysis.state != 'uncertain' else None
        motion = self.motion.update(pose, analysis.timestamp_ms, scale=scale,
                                    bbox=track.bbox if track is not None else None)
        # Box-based departure remains independent of MediaPipe validity or a
        # pose-anchor rejection. A rejected pose still cannot confirm return.
        seat = self.seat.update(pose, track, analysis.timestamp_ms,
                                frame_shape=frame_shape, identity_key=key)
        self.seat_roi = seat.seat_roi
        statuses = {'head_shake': motion.head_state,
                    'left_wave': motion.left_state, 'right_wave': motion.right_state}
        if motion.left_state == motion.right_state == 'active':
            both = 'active'
        elif 'unknown' in (motion.left_state, motion.right_state):
            both = 'unknown'
        elif 'calibrating' in (motion.left_state, motion.right_state):
            both = 'calibrating'
        else:
            both = 'normal'
        statuses['both_wave'] = both
        statuses['table_rest'] = ({'resting': 'active', 'normal': 'normal',
                                   'candidate': 'candidate', 'calibrating': 'calibrating'}
                                  .get(posture.state, 'unknown') if posture and pose is not None else 'unknown')
        statuses['seat_leave'] = {'away': 'active', 'seated': 'normal',
                                  'candidate': 'candidate', 'calibrating': 'calibrating'}.get(seat.state, 'unknown')
        details = {'head_shake': motion.head_detail, 'left_wave': motion.left_detail,
                   'right_wave': motion.right_detail, 'table_rest': posture.detail if posture else 'no_target',
                   'seat_leave': seat.detail}
        metrics = dict(motion.metrics)
        metrics.update({'head_turn_count': motion.turn_count,
                        'seat_iou_threshold': self.seat_config.iou_threshold,
                        'posture': asdict(posture) if posture else None,
                        'seat': asdict(seat), 'tracking_state': analysis.state,
                        'tracking_detail': analysis.detail,
                        'auto_reset_count': analysis.auto_reset_count,
                        'auto_reset_reason': analysis.auto_reset_reason,
                        'pose_failure_seconds': analysis.pose_failure_seconds,
                        'pose_recovery_count': analysis.pose_recovery_count})
        return BehaviorFrame(statuses, details, metrics, motion, seat)
