"""Manual-ID/identity contracts with actual OC-SORT and fake model inference.

Only the new integration behavior is exercised here. No network downloads,
neural model construction, private videos, or saved output are required.
"""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from attention_engine import AttentionAnalyzer  # noqa: E402
from target_selection import TargetSelector, TrackObservation, crop_person  # noqa: E402
from tracking_pipeline import PipelineConfig, TrackingPipeline  # noqa: E402
from vendor.ocsort.trackers.ocsort_tracker.ocsort import OCSort  # noqa: E402
from test_pipeline import ADULT, CHILD, FakeDetector, FakePoseFactory  # noqa: E402


class ToyModel:
    def __init__(self):
        self.boxes = []
        self.inputs = []

    def __call__(self, frame, verbose=False):
        self.inputs.append(frame.copy())
        return [NS(boxes=[NS(xyxy=np.asarray([box], dtype=float)) for box in self.boxes])]


class HolisticFactory:
    def __init__(self):
        self.hands = True
        self.shoulder_offset = 0.0
        self.instances = []
        self.face = NS(landmark=[NS(x=.5, y=.25) for _ in range(478)])

    def __call__(self, **options):
        factory = self

        class FakeHolistic:
            def __init__(self):
                self.closed = False
                self.calls = 0
                self.options = options

            def process(self, _rgb):
                if self.closed:
                    raise AssertionError('Closed Holistic instance was reused')
                self.calls += 1
                points = [NS(x=.5, y=.2, visibility=1.0) for _ in range(33)]
                points[11] = NS(x=.3 + factory.shoulder_offset, y=.5, visibility=1.0)
                points[12] = NS(x=.7 + factory.shoulder_offset, y=.5, visibility=1.0)
                hand = NS(landmark=[NS(x=float(x), y=float(y)) for x, y in
                                   zip(np.linspace(.45, .55, 21), np.linspace(.7, .8, 21))])
                return NS(pose_landmarks=NS(landmark=points),
                          face_landmarks=factory.face,
                          left_hand_landmarks=hand if factory.hands else None,
                          right_hand_landmarks=None)

            def close(self):
                self.closed = True

        result = FakeHolistic()
        self.instances.append(result)
        return result


class ManualIdIntegrationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        model = Path(directory.name) / 'fixture.task'
        model.write_bytes(b'No model is loaded: PoseLandmarker construction is replaced')
        self.pose_factory = FakePoseFactory()
        factory = self.pose_factory
        pose_patch = patch.object(TrackingPipeline, '_create_pose',
                                  lambda _pipeline, mode: factory.create(mode))
        pose_patch.start()
        self.addCleanup(pose_patch.stop)
        self.detector = FakeDetector()
        self.pipeline = TrackingPipeline(
            self.detector, model, fps=10,
            config=PipelineConfig(init_seconds=.2, min_samples=2,
                                  sample_stride=1, stable_observations=2,
                                  auto_reset_seconds=0, retain_target_on_loss=False,
                                  require_posture_landmarks=False))
        self.addCleanup(self.pipeline.close)
        self.frame = np.zeros((400, 800, 3), dtype=np.uint8)
        self.timestamp = 0
        self.index = 0
        self.analyzer = None

    def step(self):
        if self.analyzer is None:
            result = self.pipeline.process(self.frame, self.timestamp, self.index)
        else:
            result = self.analyzer.process(self.frame, self.timestamp, self.index)
        self.timestamp += 100
        self.index += 1
        return result

    def lock(self):
        for _ in range(30):
            result = self.step()
            if result.target_id is not None:
                self.assertEqual(result.state, 'locked', result.detail)
                return result
        self.fail('Fake poses did not establish a selected target')

    @staticmethod
    def adult_id(tracks):
        return next(t.track_id for t in tracks if t.observed and t.bbox[2] - t.bbox[0] > 200)

    def add_attention(self, tracking):
        toys = ToyModel()
        holistic = HolisticFactory()
        target = next(t for t in tracking.tracks if t.track_id == tracking.target_id)
        _, (left, top, width, height) = crop_person(self.frame, target.bbox, .08)
        x, y = left + .5 * width, top + .75 * height
        toys.boxes = [[x - 3, y - 3, x + 3, y + 3]]
        self.analyzer = AttentionAnalyzer(toys, self.pipeline, fps=10,
                                          holistic_factory=holistic)
        self.addCleanup(self.analyzer.close)
        return toys, holistic

    def test_uses_actual_ocsort_while_all_model_construction_is_fake(self):
        result = self.lock()
        self.assertIsInstance(self.pipeline.tracker.tracker, OCSort)
        self.assertTrue(self.pose_factory.instances)
        self.assertEqual(len([t for t in result.tracks if t.observed]), 2)

    def test_manual_adult_id_overrides_automatic_smallest_shoulder(self):
        first = self.lock()
        adult = self.adult_id(first.tracks)
        self.assertNotEqual(first.target_id, adult)
        generation = self.pipeline.selection_generation
        tracker_frames = self.pipeline.tracker.frame_count
        self.assertEqual(self.pipeline.select_target(adult), adult)
        self.assertEqual(self.pipeline.selection_generation, generation + 1)
        self.assertEqual(self.pipeline.tracker.frame_count, tracker_frames)
        second = self.step()
        self.assertEqual(second.target_id, adult)
        self.assertEqual(second.detail, 'manual_id')
        self.assertEqual(second.state, 'locked')
        for _ in range(5):
            self.assertEqual(self.step().target_id, adult)
        self.assertEqual(self.pipeline.auto_reset_count, 0)

    def test_manual_selection_does_not_wait_for_shoulder_calibration(self):
        self.pose_factory.no_pose = True
        first = self.step()
        self.assertIsNone(first.target_id)
        adult = self.adult_id(first.tracks)
        self.assertEqual(self.pipeline.select_target(adult), adult)
        second = self.step()
        self.assertEqual(second.target_id, adult)
        self.assertEqual(second.state, 'pose_uncertain')
        self.assertIsNone(second.pose)

    def test_invalid_id_and_types_do_not_reset_the_existing_target(self):
        first = self.lock()
        generation = self.pipeline.selection_generation
        video = self.pose_factory.video_instances()[-1]
        tracker_frames = self.pipeline.tracker.frame_count
        for invalid in (99999, -1, True, np.bool_(True), '1', 1.5, None):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    self.pipeline.select_target(invalid)
                self.assertEqual(self.pipeline.selector.target_id, first.target_id)
                self.assertEqual(self.pipeline.selection_generation, generation)
                self.assertEqual(self.pipeline.tracker.frame_count, tracker_frames)
                self.assertFalse(video.closed)

    def test_select_before_any_observation_is_rejected(self):
        with self.assertRaises(ValueError):
            self.pipeline.select_target(1)

    def test_predicted_live_track_id_cannot_be_manually_selected(self):
        first = self.lock()
        adult = self.adult_id(first.tracks)
        self.detector.detections = np.asarray([CHILD], dtype=float)
        second = self.step()
        predicted = next(t for t in second.tracks if t.track_id == adult)
        self.assertFalse(predicted.observed)
        generation = self.pipeline.selection_generation
        with self.assertRaises(ValueError):
            self.pipeline.select_target(adult)
        self.assertEqual(self.pipeline.selection_generation, generation)
        self.assertEqual(self.pipeline.selector.target_id, first.target_id)

    def test_manual_reselecting_same_id_creates_a_new_generation(self):
        first = self.lock()
        generation = self.pipeline.selection_generation
        old_video = self.pose_factory.video_instances()[-1]
        frame_count = self.pipeline.tracker.frame_count
        source_time = self.pipeline._last_timestamp
        self.pipeline.select_target(np.int64(first.target_id))
        self.assertEqual(self.pipeline.selection_generation, generation + 1)
        self.assertEqual(self.pipeline.selector.target_id, first.target_id)
        self.assertTrue(old_video.closed)
        self.assertIsNone(self.pipeline._video_pose)
        self.assertIsNone(self.pipeline.last_anchor)
        self.assertIsNone(self.pipeline.last_anchor_time)
        self.assertFalse(self.pipeline.posture_detector.baseline_ready)
        self.assertEqual(self.pipeline.tracker.frame_count, frame_count)
        self.assertEqual(self.pipeline._last_timestamp, source_time)
        second = self.step()
        self.assertEqual(second.target_id, first.target_id)
        self.assertEqual(second.selection_generation, generation + 1)
        self.assertGreater(second.timestamp_ms, source_time)
        self.assertEqual(second.detail, 'manual_id')
        self.assertEqual(len(self.pose_factory.video_instances()), 2)

    def test_manual_switch_closes_pose_and_clears_old_posture_baseline(self):
        first = self.lock()
        for _ in range(25):
            first = self.step()
        self.assertTrue(self.pipeline.posture_detector.baseline_ready)
        old_detector = self.pipeline.posture_detector
        old_video = self.pose_factory.video_instances()[-1]
        adult = self.adult_id(first.tracks)
        self.pipeline.select_target(adult)
        self.assertTrue(old_video.closed)
        self.assertIsNot(self.pipeline.posture_detector, old_detector)
        self.assertFalse(self.pipeline.posture_detector.baseline_ready)
        self.assertIsNone(self.pipeline.last_anchor)
        self.assertEqual(self.step().target_id, adult)

    def test_first_missed_detection_immediately_restarts_selection(self):
        first = self.lock()
        generation = self.pipeline.selection_generation
        old_video = self.pose_factory.video_instances()[-1]
        tracker_frames = self.pipeline.tracker.frame_count
        detector_calls = self.detector.calls
        image_calls = sum(len(instance.calls) for instance in self.pose_factory.image_instances())
        self.detector.detections = np.asarray([ADULT], dtype=float)
        second = self.step()
        self.assertTrue(any(t.track_id == first.target_id and not t.observed for t in second.tracks))
        self.assertEqual(second.state, 'reinitializing')
        self.assertEqual(second.detail, 'auto_reset_tracking_lost')
        self.assertIsNone(second.target_id)
        self.assertIsNone(second.pose)
        self.assertEqual(second.statistics, {})
        self.assertEqual(second.selection_generation, generation + 1)
        self.assertEqual(second.auto_reset_count, 1)
        self.assertTrue(old_video.closed)
        self.assertEqual(self.pipeline.tracker.frame_count, tracker_frames + 1)
        self.assertEqual(self.detector.calls, detector_calls + 1)
        self.assertEqual(sum(len(instance.calls) for instance in self.pose_factory.image_instances()), image_calls)

    def test_attention_same_id_switch_clears_hand_model_and_toy_history(self):
        tracking = self.lock()
        toys, holistic = self.add_attention(tracking)
        first = self.step()
        self.assertTrue(first.playing)
        previous = holistic.instances[-1]
        old_video = self.pose_factory.video_instances()[-1]
        generation = first.tracking.selection_generation
        selected = self.analyzer.select_target(first.tracking.target_id)
        self.assertEqual(selected, first.tracking.target_id)
        self.assertTrue(previous.closed)
        self.assertTrue(old_video.closed)
        self.assertEqual(len(self.analyzer._history), 0)
        self.assertIsNone(self.analyzer._holistic)
        toys.boxes = []
        second = self.step()
        self.assertFalse(second.playing)
        self.assertEqual(second.tracking.selection_generation, generation + 1)
        self.assertEqual(len(holistic.instances), 2)

    def test_attention_manual_adult_switch_clears_history_and_rebinds_crop(self):
        tracking = self.lock()
        toys, holistic = self.add_attention(tracking)
        first = self.step()
        adult = self.adult_id(first.tracking.tracks)
        old = holistic.instances[-1]
        self.assertTrue(first.playing)
        self.analyzer.select_target(adult)
        self.assertTrue(old.closed)
        self.assertEqual(len(self.analyzer._history), 0)
        toys.boxes = []
        second = self.step()
        self.assertEqual(second.tracking.target_id, adult)
        self.assertFalse(second.playing)
        self.assertGreater(second.crop_rect[2], first.crop_rect[2])
        self.assertIsNotNone(second.holistic)

    def test_attention_invalid_manual_id_preserves_hand_graph_and_history(self):
        tracking = self.lock()
        toys, holistic = self.add_attention(tracking)
        first = self.step()
        old = holistic.instances[-1]
        history = list(self.analyzer._history)
        with self.assertRaises(ValueError):
            self.analyzer.select_target(99999)
        self.assertFalse(old.closed)
        self.assertEqual(list(self.analyzer._history), history)
        toys.boxes = []
        self.assertTrue(self.step().playing)
        self.assertEqual(self.pipeline.selection_generation, first.tracking.selection_generation)

    def test_no_hands_is_unknown_but_preserves_holistic_face_for_expression(self):
        tracking = self.lock()
        _, holistic = self.add_attention(tracking)
        holistic.hands = False
        result = self.step()
        self.assertIsNone(result.playing)
        self.assertEqual(result.status, 'unknown')
        self.assertEqual(result.detail, 'no_hands_detected')
        self.assertEqual(result.hand_boxes, [])
        self.assertIsNotNone(result.crop_rect)
        self.assertIs(result.holistic.face_landmarks, holistic.face)
        self.assertFalse(holistic.instances[-1].closed)

    def test_invalid_hand_geometry_preserves_verified_face_and_clears_toy_history(self):
        tracking = self.lock()
        _, holistic = self.add_attention(tracking)
        self.assertTrue(self.step().playing)
        with patch('attention_engine.get_hand_bbox', side_effect=ValueError('bad hand')):
            result = self.step()
        self.assertIsNone(result.playing)
        self.assertEqual(result.detail, 'invalid_hand_landmarks')
        self.assertIs(result.holistic.face_landmarks, holistic.face)
        self.assertIsNotNone(result.crop_rect)
        self.assertFalse(holistic.instances[-1].closed)
        self.assertEqual(len(self.analyzer._history), 0)

    def test_missing_target_clears_hands_and_history_on_first_lost_frame(self):
        tracking = self.lock()
        toys, holistic = self.add_attention(tracking)
        self.assertTrue(self.step().playing)
        old = holistic.instances[-1]
        self.detector.detections = np.asarray([ADULT], dtype=float)
        missing = self.step()
        self.assertEqual(missing.tracking.state, 'reinitializing')
        self.assertIsNone(missing.playing)
        self.assertIsNone(missing.holistic)
        self.assertEqual(missing.hand_boxes, [])
        self.assertTrue(old.closed)
        self.assertEqual(len(self.analyzer._history), 0)
        self.assertEqual(len(toys.inputs), 2)

    def test_hand_toy_overlap_uses_selected_crop_and_short_source_time_history(self):
        tracking = self.lock()
        toys, _ = self.add_attention(tracking)
        first = self.step()
        self.assertTrue(first.playing)
        self.assertEqual(first.toy_boxes[0][0], int(toys.boxes[0][0]))
        toys.boxes = []
        for _ in range(4):
            self.assertTrue(self.step().playing)
        self.assertFalse(self.step().playing)

    def test_holistic_shoulder_mismatch_rejects_hand_and_face_evidence(self):
        tracking = self.lock()
        _, holistic = self.add_attention(tracking)
        holistic.shoulder_offset = .8
        result = self.step()
        self.assertEqual(result.detail, 'holistic_target_shoulders_mismatch')
        self.assertIsNone(result.playing)
        self.assertIsNone(result.holistic)
        self.assertTrue(holistic.instances[-1].closed)
        self.assertEqual(len(self.analyzer._history), 0)


class ManualSelectorValidationTests(unittest.TestCase):
    def test_selector_rejects_missing_predicted_and_invalid_boxes(self):
        selector = TargetSelector()
        selector.observe([
            TrackObservation(1, np.array([0, 0, 100, 100]), True, 1, 0, 0),
            TrackObservation(2, np.array([0, 0, 100, 100]), False, 1, 0, 0),
            TrackObservation(3, np.array([0, 0, np.nan, 100]), True, 1, 0, 0),
        ], 0)
        selector.select_id(1)
        for invalid in (2, 3, 9, True, 1.5, '1'):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    selector.select_id(invalid)
                self.assertEqual(selector.target_id, 1)


if __name__ == '__main__':
    unittest.main()
