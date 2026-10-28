"""GUI controls and headless orchestration without windows or neural models."""
from contextlib import ExitStack
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

import numpy as np

import integrated_app as app
from behavior_engine import BehaviorEngine
from seat_detection import SeatConfig, SeatResult
from tracking_pipeline import PipelineConfig
from window_log import WindowLog


class Capture:
    def __init__(self, count=3):
        self.frames = [np.zeros((80, 120, 3), dtype=np.uint8) for _ in range(count)]
        self.index = 0
        self.release = Mock()

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == app.cv2.CAP_PROP_FPS:
            return 30.0
        if prop == app.cv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames)
        if prop == app.cv2.CAP_PROP_POS_MSEC:
            return max(0, self.index - 1) * 1000 / 30
        return 0

    def read(self):
        if self.index == len(self.frames):
            return False, None
        result = self.frames[self.index]
        self.index += 1
        return True, result


class Analyzer:
    def __init__(self, pipeline):
        self.pipeline = pipeline
        self.process = Mock(side_effect=self._process)
        self.select_target = Mock(side_effect=self._select)
        self.target_id = None
        self.close = Mock(side_effect=pipeline.close)

    def _select(self, track_id):
        self.target_id = track_id
        return track_id

    def _process(self, frame, timestamp, frame_index):
        tracking = NS(target_id=self.target_id, selection_generation=1,
                      timestamp_ms=timestamp, state='locked',
                      tracks=[NS(track_id=7, observed=True)])
        return NS(tracking=tracking, playing=True, detail='test_toy',
                  toy_boxes=[], hand_boxes=[])


class IntegratedAppTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def gui(self, keys, analyzer=None, engine=None, roi=(0, 0, 0, 0)):
        analyzer = analyzer or Mock()
        engine = engine or BehaviorEngine()
        controls = []
        render = Mock(return_value=np.zeros((1080, 1920, 3), dtype=np.uint8))
        with patch.object(app.cv2, 'imshow') as imshow, \
                patch.object(app.cv2, 'waitKey', side_effect=keys), \
                patch.object(app, 'window_visible', return_value=True), \
                patch.object(app.cv2, 'selectROI', return_value=roi) as select, \
                patch.object(app.cv2, 'destroyWindow'):
            action = app.interact(render, analyzer, engine, 1250, (480, 640, 3), controls, 1)
        return action, controls, analyzer, engine, render, imshow, select

    def test_roi_maps_letterboxed_image_and_rejects_sidebar(self):
        for shape in [(480, 640, 3), (1080, 1920, 3), (1000, 500, 3)]:
            with self.subTest(shape=shape):
                width, height, left, top = app.display.display_geometry(shape)
                actual = app.display_roi_to_source((left, top, width, height), shape)
                np.testing.assert_allclose(actual, (0, 0, shape[1], shape[0]))
                self.assertIsNone(app.display_roi_to_source((1501, 10, 100, 100), shape))
                self.assertIsNone(app.display_roi_to_source((left, top, 0, 100), shape))

    def test_pause_cancel_and_manual_roi_keep_the_same_source_time(self):
        action, controls, analyzer, engine, render, _, select = self.gui(
            [ord(' '), ord('c'), ord('s'), ord(' '), -1], roi=(300, 200, 600, 600))
        self.assertEqual(action, 'continue')
        self.assertEqual([item['action'] for item in controls], ['cancel_seat', 'manual_seat'])
        self.assertEqual([item['time_seconds'] for item in controls], [1.25, 1.25])
        self.assertTrue(engine.seat.baseline_ready)
        select.assert_called_once()
        analyzer.process.assert_not_called()
        self.assertTrue(any(call.args[1] for call in render.call_args_list))

    def test_manual_roi_cancel_does_not_set_a_seat(self):
        action, controls, _, engine, _, _, _ = self.gui([ord('c'), ord('s'), -1])
        self.assertEqual(action, 'continue')
        self.assertEqual([item['action'] for item in controls], ['cancel_seat'])
        self.assertFalse(engine.seat.baseline_ready)
        self.assertTrue(engine.seat.manual_available(1250))

    def test_id_editing_selects_exact_id_and_resets_behavior_once(self):
        analyzer, engine = Mock(), Mock()
        analyzer.select_target.return_value = 13
        action, controls, *_ = self.gui(
            [ord('i'), ord('1'), ord('2'), 8, ord('3'), 13, -1], analyzer, engine)
        self.assertEqual(action, 'continue')
        analyzer.select_target.assert_called_once_with(13)
        engine.reset.assert_called_once()
        self.assertEqual(controls, [{'time_seconds': 1.25, 'action': 'select_id', 'id': 13}])

    def test_empty_id_and_escape_cancel_do_not_reset_selection(self):
        action, controls, analyzer, engine, *_ = self.gui([ord('i'), 13, 27, -1])
        self.assertEqual(action, 'continue')
        self.assertFalse(controls)
        analyzer.select_target.assert_not_called()
        self.assertFalse(engine.seat.baseline_ready)

    def test_unknown_id_keeps_input_until_cancel(self):
        analyzer = Mock()
        analyzer.select_target.side_effect = ValueError('ID is not currently observed')
        action, controls, *_ = self.gui([ord('i'), ord('9'), 13, 27, -1], analyzer)
        self.assertEqual(action, 'continue')
        self.assertFalse(controls)
        analyzer.select_target.assert_called_once_with(9)

    def test_quit_and_next_work_from_pause(self):
        self.assertEqual(self.gui([ord(' '), ord('q')])[0], 'quit')
        self.assertEqual(self.gui([ord(' '), ord('n')])[0], 'next')

    def test_cli_settings_and_explicit_overrides(self):
        settings = self.root / 'settings.json'
        settings.write_text(json.dumps({'headless': True, 'target_id': 7, 'max_frames': 10}), encoding='utf-8')
        args = app.parse_args(['--settings', str(settings), '--max-frames', '2'])
        self.assertTrue(args.headless)
        self.assertEqual(args.target_id, 7)
        self.assertEqual(args.max_frames, 2)
        for options in [['--target-id', '0'], ['--max-frames', '0'], ['--start-seconds', 'nan']]:
            with self.subTest(options=options), patch('sys.stderr'), self.assertRaises(SystemExit):
                app.parse_args(['--settings', str(settings), *options])

    def scenario(self, *, count=3, frame_log=True, export=False, target_id=7):
        args = app.build_parser().parse_args([])
        args.output = self.root / 'logs'
        args.headless, args.frame_log, args.export_video = True, frame_log, export
        args.target_id, args.max_frames = target_id, count
        for name in ('toy_model', 'emotion_model', 'pose_model', 'detector_model'):
            model_path = self.root / name
            model_path.write_bytes(b'model fixture')
            setattr(args, name, model_path)
        capture = Capture(count)
        pipeline = NS(close=Mock())
        analyzer = Analyzer(pipeline)
        seat = NS(seat_roi=None)
        behavior = NS(seat=seat, reset=Mock())
        behavior.update = Mock(side_effect=lambda analysis, shape: NS(
            statuses=dict.fromkeys(app.BODY_FIELDS, 'normal'),
            seat=SeatResult('calibrating', 'test_seat', calibration_seconds=analysis.timestamp_ms/1000,
                            calibration_samples=1)))
        emotion = NS(device='cpu', process=Mock(return_value=NS(
            label=0, probability=.2, detail='neutral', face_bbox=None)))
        detector, toy = NS(provenance={'test': True}), NS(names={0: 'toy'})
        stack = ExitStack()
        self.addCleanup(stack.close)
        for name, value in [('TrackingPipeline', Mock(return_value=pipeline)),
                            ('AttentionAnalyzer', Mock(return_value=analyzer)),
                            ('BehaviorEngine', Mock(return_value=behavior)),
                            ('render_preview', Mock(return_value=np.zeros((1080, 1920, 3), dtype=np.uint8)))]:
            stack.enter_context(patch.object(app, name, value))
        stack.enter_context(patch.object(app.cv2, 'VideoCapture', return_value=capture))
        stack.enter_context(patch.object(app.cv2, 'imshow'))
        stack.enter_context(patch.object(app.cv2, 'waitKey', side_effect=AssertionError('headless called waitKey')))
        writer = Mock()
        writer.isOpened.return_value = True
        stack.enter_context(patch.object(app.cv2, 'VideoWriter', return_value=writer))
        stack.enter_context(patch('builtins.print'))
        run = lambda: app.play_video(self.root / 'fixture.mp4', detector, toy, emotion,
                                    args, PipelineConfig(), SeatConfig(baseline_seconds=3, max_displacement_ratio=.2))
        return NS(args=args, capture=capture, pipeline=pipeline, analyzer=analyzer,
                  behavior=behavior, emotion=emotion, writer=writer, run=run)

    def read_metadata(self):
        return json.loads(next((self.root/'logs').glob('*_metadata.json')).read_text(encoding='utf-8'))

    def test_headless_counts_fresh_frames_and_source_identity_once(self):
        scenario = self.scenario()
        self.assertTrue(scenario.run())
        self.assertEqual(scenario.analyzer.process.call_count, 3)
        self.assertEqual(scenario.emotion.process.call_count, 3)
        self.assertEqual(scenario.behavior.update.call_count, 3)
        scenario.analyzer.select_target.assert_called_once_with(7)
        scenario.capture.release.assert_called_once()
        scenario.pipeline.close.assert_called_once()
        metadata = self.read_metadata()
        self.assertEqual(metadata['frames'], 3)
        self.assertEqual(metadata['counts']['emotion_0'], 3)
        self.assertEqual(metadata['end_reason'], 'max_frames')
        self.assertEqual(metadata['toy_model_names'], {'0': 'toy'})
        self.assertIn('3 continuous', metadata['seat_rule'])
        self.assertIn('<=0.2', metadata['seat_rule'])
        with next((self.root/'logs').glob('*_frames.csv')).open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual([row['target_id'] for row in rows], ['', '7', '7'])
        self.assertEqual([int(row['frame_index']) for row in rows], [0, 1, 2])
        self.assertEqual(rows[0]['seat_calibration_samples'], '1')
        self.assertEqual(rows[0]['seat_detail'], 'test_seat')

    def test_metadata_preparation_failure_releases_capture(self):
        scenario = self.scenario()
        with patch.object(app, 'sha256', side_effect=OSError('hash failure')):
            with self.assertRaisesRegex(OSError, 'hash failure'):
                scenario.run()
        scenario.capture.release.assert_called_once()
        self.assertEqual(self.read_metadata()['end_reason'], 'error')

    def test_analyzer_constructor_failure_closes_unowned_pipeline(self):
        scenario = self.scenario()
        with patch.object(app, 'AttentionAnalyzer', side_effect=RuntimeError('analyzer startup')):
            with self.assertRaisesRegex(RuntimeError, 'analyzer startup'):
                scenario.run()
        scenario.pipeline.close.assert_called_once()
        scenario.capture.release.assert_called_once()

    def test_cleanup_failures_do_not_hide_inference_error_or_skip_resources(self):
        scenario = self.scenario(count=2, export=True)
        scenario.emotion.process.side_effect = [NS(label=0, probability=.2, detail='ok', face_bbox=None),
                                                ValueError('primary inference failure')]
        scenario.analyzer.close.side_effect = RuntimeError('analyzer close failure')
        scenario.capture.release.side_effect = RuntimeError('capture release failure')
        scenario.writer.release.side_effect = RuntimeError('writer release failure')
        loggers = []

        class FailingFinish(WindowLog):
            def __init__(self, stream):
                super().__init__(stream)
                loggers.append(self)

            def finish(self, **options):
                super().finish(**options)
                raise RuntimeError('finish failure')

        with patch.object(app, 'WindowLog', FailingFinish):
            with self.assertRaisesRegex(ValueError, 'primary inference failure'):
                scenario.run()
        scenario.pipeline.close.assert_called_once()
        scenario.capture.release.assert_called_once()
        scenario.writer.release.assert_called_once()
        self.assertTrue(loggers[0].stream.closed)
        metadata = self.read_metadata()
        self.assertIn('primary inference failure', metadata['error'])
        self.assertEqual({item['operation'] for item in metadata['cleanup_errors']},
                         {'finish_windows', 'close_analyzer', 'release_capture', 'release_writer'})
        # All CSV files are closed and can be opened exclusively on Windows.
        for path in (self.root/'logs').glob('*.csv'):
            with path.open('a', encoding='utf-8'):
                pass

    def test_cleanup_failure_after_success_is_reported(self):
        scenario = self.scenario()
        scenario.analyzer.close.side_effect = RuntimeError('close failure')
        with self.assertRaisesRegex(RuntimeError, 'close failure'):
            scenario.run()
        scenario.pipeline.close.assert_called_once()
        scenario.capture.release.assert_called_once()
        self.assertEqual(self.read_metadata()['end_reason'], 'error')

    def test_metadata_write_failure_does_not_replace_primary_error(self):
        scenario = self.scenario()
        scenario.emotion.process.side_effect = ValueError('inference failed')
        with patch.object(app, 'save_json', side_effect=OSError('metadata unavailable')):
            with self.assertRaisesRegex(ValueError, 'inference failed'):
                scenario.run()
        scenario.capture.release.assert_called_once()


if __name__ == '__main__':
    unittest.main()
