"""Seat lifecycle contracts using source timestamps and synthetic geometry.

No detector, video, MediaPipe model, or neural network is loaded by these tests.
"""
from types import SimpleNamespace
import unittest

import numpy as np

from behavior_engine import BehaviorEngine
from seat_detection import SeatConfig, SeatDetector
from target_selection import MappedPose, TrackObservation


def track(box=(100, 100, 200, 300), *, track_id=1, observed=True):
    return TrackObservation(track_id, np.asarray(box, dtype=float), observed, 100, 0, 0)


def shifted(dx=0, dy=0, **kwargs):
    return track(np.array([100, 100, 200, 300]) + [dx, dy, dx, dy], **kwargs)


def pose():
    points = np.full((33, 2), [150.0, 160.0])
    points[11], points[12] = [125, 200], [175, 200]
    return MappedPose(points, np.ones(33), np.ones(33))


def analysis(ms, *, target_id=1, generation=1, dx=0, observed=True, has_pose=False):
    return SimpleNamespace(
        selection_generation=generation, target_id=target_id,
        pose_recovery_count=0, posture=None,
        statistics={} if target_id is None else {target_id: {'median_width': 50}},
        tracks=[] if target_id is None else [shifted(dx, track_id=target_id, observed=observed)],
        pose=pose() if has_pose else None, state='locked' if observed else 'lost',
        timestamp_ms=ms, detail='fixture', auto_reset_count=0,
        auto_reset_reason='', pose_failure_seconds=0.0)


class IntegratedSeatTests(unittest.TestCase):
    def config(self, **changes):
        values = dict(baseline_seconds=.2, min_samples=3)
        values.update(changes)
        return SeatConfig(**values)

    def calibrate(self, detector=None, start=0, **kwargs):
        detector = detector or SeatDetector(self.config())
        result = None
        for ms in (start, start + 100, start + 200):
            result = detector.update(None, track(), ms, **kwargs)
        self.assertTrue(detector.baseline_ready)
        self.assertEqual(result.state, 'seated')
        return detector, result

    def test_default_requires_ten_source_seconds_without_pose(self):
        detector = SeatDetector()
        self.assertEqual(detector.config.baseline_seconds, 10)
        self.assertEqual(detector.config.max_displacement_ratio, .25)
        for ms in range(0, 10000, 100):
            result = detector.update(None, track(), ms)
            self.assertEqual(result.state, 'calibrating')
            self.assertFalse(detector.baseline_ready)
        result = detector.update(None, track(), 10000)
        self.assertEqual(result.state, 'seated')
        self.assertEqual(result.seat_roi, (100, 100, 200, 300))
        self.assertEqual(result.reference_source, 'automatic')

    def test_box_corners_use_arithmetic_mean_not_median(self):
        detector = SeatDetector(self.config())
        boxes = ([100, 100, 200, 300], [100, 100, 200, 300], [120, 100, 220, 300])
        for ms, box in zip((0, 100, 200), boxes):
            result = detector.update(None, track(box), ms)
        np.testing.assert_allclose(result.seat_roi, np.mean(boxes, axis=0))
        self.assertNotEqual(result.seat_roi, (100, 100, 200, 300))

    def test_gradual_drift_compares_fixed_initial_center(self):
        detector = SeatDetector(self.config(baseline_seconds=1))
        for ms, dx in ((0, 0), (100, 10), (200, 20), (300, 30)):
            result = detector.update(None, shifted(dx), ms)
        self.assertEqual(result.detail, 'calibration_restarted_displacement')
        self.assertEqual(result.calibration_seconds, 0)
        self.assertEqual(result.calibration_samples, 1)
        for ms in range(400, 1300, 100):
            result = detector.update(None, shifted(30), ms)
            self.assertFalse(detector.baseline_ready)
        result = detector.update(None, shifted(30), 1300)
        self.assertEqual(result.seat_roi, (130, 100, 230, 300))

    def test_displacement_uses_euclidean_distance_and_initial_width(self):
        for box in ([120, 120, 220, 320], [-320, 100, 680, 300]):
            with self.subTest(box=box):
                detector = SeatDetector(self.config())
                detector.update(None, track(), 0)
                result = detector.update(None, track(box), 100)
                self.assertEqual(result.detail, 'calibration_restarted_displacement')
                self.assertEqual(result.calibration_samples, 1)

    def test_displacement_exactly_at_limit_remains_in_collection(self):
        detector = SeatDetector(self.config())
        detector.update(None, track(), 0)
        result = detector.update(None, shifted(25), 100)
        self.assertEqual(result.calibration_samples, 2)
        self.assertEqual(result.displacement_ratio, .25)
        result = detector.update(None, shifted(25), 200)
        self.assertTrue(detector.baseline_ready)

    def test_loss_and_prediction_cannot_fill_calibration_window(self):
        for missing in (None, shifted(observed=False)):
            with self.subTest(missing=missing):
                detector = SeatDetector(self.config())
                detector.update(None, track(), 0)
                detector.update(None, track(), 100)
                result = detector.update(None, missing, 200)
                self.assertEqual(result.state, 'unknown')
                self.assertEqual(result.calibration_samples, 0)
                for ms in (300, 400):
                    result = detector.update(None, track(), ms)
                    self.assertFalse(detector.baseline_ready)
                self.assertEqual(detector.update(None, track(), 500).state, 'seated')

    def test_source_gap_restarts_unfinished_calibration(self):
        detector = SeatDetector(self.config())
        detector.update(None, track(), 0)
        detector.update(None, track(), 100)
        result = detector.update(None, track(), 1000)
        self.assertEqual(result.calibration_samples, 1)
        self.assertEqual(result.calibration_seconds, 0)
        detector.update(None, track(), 1100)
        self.assertEqual(detector.update(None, track(), 1200).state, 'seated')

    def test_same_numeric_id_new_generation_never_reuses_samples(self):
        detector = SeatDetector(self.config())
        detector.update(None, track(), 0, identity_key=(1, 1))
        detector.update(None, track(), 100, identity_key=(1, 1))
        result = detector.update(None, shifted(20), 200, identity_key=(2, 1))
        self.assertEqual(result.calibration_samples, 1)
        detector.update(None, shifted(20), 300, identity_key=(2, 1))
        result = detector.update(None, shifted(20), 400, identity_key=(2, 1))
        self.assertEqual(result.seat_roi, (120, 100, 220, 300))

    def test_minimum_samples_remains_required(self):
        detector = SeatDetector(self.config(min_samples=4))
        for ms in (0, 100, 200):
            self.assertEqual(detector.update(None, track(), ms).state, 'calibrating')
        self.assertEqual(detector.update(None, track(), 300).state, 'seated')

    def test_reference_survives_identity_change_loss_gap_and_r_reset(self):
        detector, before = self.calibrate(identity_key=(1, 1))
        detector.update(None, None, 300, identity_key=(2, None))
        result = detector.update(None, shifted(300, track_id=8), 5000, identity_key=(2, 8))
        self.assertEqual(result.state, 'away')
        self.assertEqual(result.seat_roi, before.seat_roi)
        detector.reset()
        result = detector.update(None, track(track_id=9), 5100, identity_key=(3, 9))
        self.assertEqual(result.seat_roi, before.seat_roi)
        self.assertEqual(result.state, 'seated')

    def test_identity_change_resets_old_subject_away_latch(self):
        detector, _ = self.calibrate(identity_key=(1, 1))
        detector.update(None, shifted(300), 300, identity_key=(1, 1))
        result = detector.update(None, track(track_id=2), 400, identity_key=(1, 2))
        self.assertEqual(result.state, 'seated')

    def test_cancel_removes_seat_and_opens_exact_ten_second_window(self):
        detector, _ = self.calibrate()
        detector.cancel(1000)
        self.assertFalse(detector.baseline_ready)
        self.assertEqual(detector.manual_remaining_seconds(1000), 10)
        self.assertFalse(detector.manual_available(999))
        self.assertTrue(detector.manual_available(10999))
        self.assertFalse(detector.manual_available(11000))
        self.assertEqual(detector.manual_remaining_seconds(11000), 0)

    def test_manual_roi_is_immediate_and_needs_no_pose_or_target(self):
        detector = SeatDetector()
        self.assertFalse(detector.set_manual_roi((10, 20, 100, 200), 0))
        detector.cancel(1000)
        self.assertTrue(detector.set_manual_roi((10, 20, 100, 200), 1000, (600, 1000)))
        self.assertTrue(detector.baseline_ready)
        self.assertEqual(detector.seat_roi, (10, 20, 100, 200))
        self.assertFalse(detector.manual_available(1000))
        result = detector.update(None, None, 1000)
        self.assertEqual(result.state, 'unknown')
        self.assertEqual(result.reference_source, 'manual')

    def test_invalid_manual_selection_does_not_consume_window(self):
        detector = SeatDetector()
        detector.cancel(0)
        for box, shape in ((None, (600, 1000)), ((10, 10, 10, 20), (600, 1000)),
                           ((-1, 0, 20, 20), None), ((0, 0, np.inf, 20), None),
                           ((0, 0, 1001, 20), (600, 1000)),
                           ((0, 0, 20, 20), (0, 1000))):
            self.assertFalse(detector.set_manual_roi(box, 0, shape))
            self.assertFalse(detector.baseline_ready)
            self.assertTrue(detector.manual_available(0))
        self.assertTrue(detector.set_manual_roi((0, 0, 1000, 600), 0, (600, 1000, 3)))

    def test_manual_selection_at_deadline_is_rejected(self):
        detector = SeatDetector()
        detector.cancel(0)
        self.assertFalse(detector.set_manual_roi((0, 0, 50, 50), 10000))
        self.assertFalse(detector.baseline_ready)

    def test_cancellation_auto_mean_cannot_finalize_before_manual_deadline(self):
        detector, _ = self.calibrate(SeatDetector(self.config(manual_window_seconds=1)))
        detector.cancel(300)
        for ms in range(400, 1300, 100):
            result = detector.update(None, shifted(20), ms)
            self.assertEqual(result.state, 'calibrating')
            self.assertFalse(detector.baseline_ready)
        result = detector.update(None, shifted(20), 1300)
        self.assertEqual(result.state, 'seated')
        self.assertEqual(result.seat_roi, (120, 100, 220, 300))

    def test_unstable_replacement_waits_beyond_manual_deadline(self):
        detector = SeatDetector(self.config(baseline_seconds=.4, manual_window_seconds=.5))
        detector.cancel(300)
        for ms, dx in ((400, 0), (500, 0), (600, 50), (700, 50), (800, 50), (900, 50)):
            result = detector.update(None, shifted(dx), ms)
            self.assertFalse(detector.baseline_ready)
        result = detector.update(None, shifted(50), 1000)
        self.assertEqual(result.seat_roi, (150, 100, 250, 300))

    def test_no_observations_at_deadline_never_fabricates_a_reference(self):
        detector = SeatDetector(self.config(manual_window_seconds=.5))
        detector.cancel(0)
        result = detector.update(None, None, 1000)
        self.assertEqual(result.state, 'unknown')
        self.assertFalse(detector.baseline_ready)
        for ms in (1100, 1200):
            self.assertEqual(detector.update(None, track(), ms).state, 'calibrating')
        self.assertEqual(detector.update(None, track(), 1300).state, 'seated')

    def test_repeated_cancel_restarts_manual_deadline(self):
        detector = SeatDetector()
        detector.cancel(0)
        detector.cancel(9000)
        self.assertEqual(detector.manual_remaining_seconds(9000), 10)
        self.assertTrue(detector.set_manual_roi((10, 10, 20, 20), 18000))

    def test_r_during_replacement_preserves_manual_window(self):
        detector = SeatDetector()
        detector.cancel(0)
        detector.update(None, track(), 100)
        detector.reset()
        self.assertEqual(detector.manual_remaining_seconds(100), 9.9)
        self.assertTrue(detector.set_manual_roi((10, 10, 20, 20), 100))

    def test_departure_only_uses_observed_box_return_needs_trusted_pose(self):
        detector, before = self.calibrate()
        self.assertEqual(detector.update(None, shifted(300), 300).state, 'away')
        missing = detector.update(None, shifted(observed=False), 400)
        self.assertEqual(missing.state, 'unknown')
        self.assertEqual(missing.seat_roi, before.seat_roi)
        self.assertEqual(detector.update(None, track(), 500).detail, 'awaiting_return_pose')
        rejected = pose()
        rejected.presence[11] = .69
        self.assertEqual(detector.update(rejected, track(), 600).state, 'away')
        self.assertEqual(detector.update(pose(), track(), 700).state, 'seated')

    def test_iou_half_boundary_does_not_trigger_departure(self):
        detector = SeatDetector(roi=(150, 150, 450, 450))
        at_half = track((250, 150, 550, 450))
        result = detector.update(None, at_half, 0)
        self.assertEqual(result.iou, .5)
        self.assertEqual(result.state, 'seated')
        self.assertEqual(detector.update(None, track((251, 150, 551, 450)), 100).state, 'away')

    def test_initial_cli_roi_is_direct_and_does_not_open_manual_window(self):
        detector = SeatDetector(roi=(0, 0, 100, 100))
        self.assertTrue(detector.baseline_ready)
        self.assertFalse(detector.manual_available(0))
        self.assertEqual(detector.update(None, track(), 0).reference_source, 'initial_manual')

    def test_full_reset_and_new_video_drop_reference_and_old_clock(self):
        detector, _ = self.calibrate(start=1000)
        detector.reset(preserve_reference=False)
        self.assertFalse(detector.baseline_ready)
        self.assertEqual(detector.update(None, track(), 0).state, 'calibrating')
        self.assertFalse(SeatDetector().baseline_ready)

    def test_invalid_geometry_breaks_pending_collection(self):
        for box in ((100, 100, 100, 200), (100, 100, np.nan, 200), (1, 2, 3)):
            detector = SeatDetector(self.config())
            detector.update(None, track(), 0)
            self.assertEqual(detector.update(None, track(box), 100).state, 'unknown')
            self.assertEqual(detector.update(None, track(), 200).calibration_samples, 1)
        detector = SeatDetector(self.config())
        self.assertEqual(detector.update(None, shifted(2000), 0,
                                         frame_shape=(600, 1000)).detail, 'person_box_outside_frame')

    def test_timestamps_and_configuration_are_validated(self):
        for value in (-1, True, np.bool_(True), 1.5):
            with self.assertRaises(ValueError):
                SeatDetector().update(None, track(), value)
        detector = SeatDetector()
        detector.update(None, track(), 100)
        for call in (lambda: detector.update(None, track(), 100),
                     lambda: detector.cancel(99)):
            with self.assertRaises(ValueError):
                call()
        detector.cancel(200)
        with self.assertRaises(ValueError):
            detector.update(None, track(), 150)
        for changes in ({'baseline_seconds': 0}, {'max_displacement_ratio': np.nan},
                        {'manual_window_seconds': -1}, {'min_samples': True}):
            with self.assertRaises(ValueError):
                SeatConfig(**changes)


class IntegratedBehaviorSeatTests(unittest.TestCase):
    shape = (600, 1000, 3)

    def engine(self):
        return BehaviorEngine(seat_config=SeatConfig(baseline_seconds=.2, min_samples=3))

    def calibrate(self, engine):
        for ms in (0, 100, 200):
            result = engine.update(analysis(ms), self.shape)
        self.assertEqual(result.statuses['seat_leave'], 'normal')
        return result

    def test_pose_free_seat_calibration_is_independent_of_motion_unknown(self):
        engine = self.engine()
        result = self.calibrate(engine)
        self.assertEqual(result.statuses['head_shake'], 'unknown')
        self.assertEqual(result.statuses['table_rest'], 'unknown')
        self.assertEqual(result.seat.seat_roi, (100, 100, 200, 300))

    def test_reselection_and_r_preserve_the_frozen_video_seat(self):
        engine = self.engine()
        before = self.calibrate(engine)
        missing = engine.update(analysis(300, target_id=None, generation=2), self.shape)
        self.assertEqual(missing.statuses['seat_leave'], 'unknown')
        self.assertEqual(missing.seat.seat_roi, before.seat.seat_roi)
        new_target = engine.update(analysis(400, target_id=8, generation=2, dx=300), self.shape)
        self.assertEqual(new_target.statuses['seat_leave'], 'active')
        self.assertEqual(new_target.seat.seat_roi, before.seat.seat_roi)
        engine.reset()
        result = engine.update(analysis(500, target_id=8, generation=3), self.shape)
        self.assertEqual(result.seat.seat_roi, before.seat.seat_roi)
        self.assertEqual(result.statuses['seat_leave'], 'normal')

    def test_generation_change_restarts_unfinished_calibration(self):
        engine = self.engine()
        engine.update(analysis(0), self.shape)
        engine.update(analysis(100), self.shape)
        result = engine.update(analysis(200, generation=2, dx=20), self.shape)
        self.assertEqual(result.seat.calibration_samples, 1)
        engine.update(analysis(300, generation=2, dx=20), self.shape)
        result = engine.update(analysis(400, generation=2, dx=20), self.shape)
        self.assertEqual(result.seat.seat_roi, (120, 100, 220, 300))

    def test_clear_and_set_controls_do_not_replace_motion_state(self):
        engine = self.engine()
        self.calibrate(engine)
        motion = engine.motion
        engine.clear_seat(200)
        self.assertIsNone(engine.seat_roi)
        self.assertIs(engine.motion, motion)
        self.assertTrue(engine.set_seat_roi((10, 10, 300, 400), 200, self.shape))
        self.assertEqual(engine.seat_roi, (10, 10, 300, 400))
        self.assertIs(engine.motion, motion)
        engine.reset()
        self.assertEqual(engine.seat_roi, (10, 10, 300, 400))


if __name__ == '__main__':
    unittest.main()
