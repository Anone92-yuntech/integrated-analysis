"""One tracked subject, DAN expression and five-second behavior windows."""
from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import asdict, replace
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / '.cache' / 'ultralytics')
os.environ['MPLCONFIGDIR'] = str(ROOT / '.cache' / 'matplotlib')
os.environ['TORCH_HOME'] = str(ROOT / '.cache' / 'torch')

import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

from attention_engine import AttentionAnalyzer
from behavior_engine import BehaviorEngine
from emotion_inference import DanEmotionPredictor
from motion_detection import MotionConfig
from person_detector import PersonDetector
import pose_viewer as display
from seat_detection import SeatConfig
from tracking_pipeline import PipelineConfig, TrackingPipeline
from window_log import WindowLog, WINDOW_SPEC

WINDOW_NAME = 'Integrated expression and behavior analysis'
BODY_FIELDS = ('head_shake', 'both_wave', 'table_rest', 'seat_leave')
LABELS = {'head_shake': '搖頭', 'both_wave': '雙手同時揮舞',
          'table_rest': '疑似趴桌', 'seat_leave': '離開座位區'}
EMOTIONS = {0: '0 中性', 1: '1 正向', 2: '2 無偵測'}
STATES = {'active': '成立', 'normal': '未成立', 'unknown': '無法判斷',
          'calibrating': '校正中', 'candidate': '持續觀察'}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    temporary = Path(path).with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False, default=str), encoding='utf-8')
    temporary.replace(path)


def project_path(value):
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings', type=project_path, default=ROOT / 'settings.json')
    parser.add_argument('--input', type=project_path,
                        help='Video file or nonrecursive folder; omit in GUI mode to choose a file')
    parser.add_argument('--output', type=project_path, default=ROOT / 'logs')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--start-seconds', type=float, default=0.0)
    parser.add_argument('--target-id', type=int, help='Select this ID when it first has a real detection')
    parser.add_argument('--export-video', action='store_true', help='Save annotated FHD video')
    parser.add_argument('--frame-log', action='store_true', help='Also save per-frame diagnostic evidence')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto',
                        help='PyTorch device for toy and expression models')
    parser.add_argument('--detector-device', choices=('auto', 'cpu', 'cuda'), default='cpu')
    parser.add_argument('--toy-model', type=project_path, default=ROOT / 'models' / 'best.pt')
    parser.add_argument('--emotion-model', type=project_path, default=ROOT / 'models' / 'best_transfer_model.pth')
    parser.add_argument('--pose-model', type=project_path, default=ROOT / 'models' / 'pose_landmarker_lite.task')
    parser.add_argument('--detector-model', type=project_path, default=ROOT / 'models' / 'yolox_nano.onnx')
    parser.add_argument('--init-seconds', type=float, default=2.0)
    parser.add_argument('--min-samples', type=int, default=10)
    parser.add_argument('--landmark-confidence', type=float, default=.7)
    parser.add_argument('--pose-recovery-seconds', type=float, default=2.0)
    parser.add_argument('--pose-reselect-seconds', type=float, default=10.0)
    parser.add_argument('--seat-baseline-seconds', type=float, default=10.0)
    parser.add_argument('--seat-max-displacement-ratio', type=float, default=.25)
    parser.add_argument('--seat-iou-threshold', type=float, default=.5)
    parser.add_argument('--toy-iou-threshold', type=float, default=0.0)
    return parser


def parse_args(argv=None):
    parser = build_parser()
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument('--settings', type=project_path, default=ROOT / 'settings.json')
    known, _ = early.parse_known_args(argv)
    if known.settings.is_file():
        config = json.loads(known.settings.read_text(encoding='utf-8-sig'))
        if not isinstance(config, dict):
            parser.error('settings must be a flat JSON object')
        allowed = {action.dest: action for action in parser._actions if action.dest != 'help'}
        unknown = set(config) - set(allowed)
        if unknown:
            parser.error('Unknown settings: ' + ', '.join(sorted(unknown)))
        for key, value in config.items():
            action = allowed[key]
            if isinstance(action, argparse._StoreTrueAction):
                if not isinstance(value, bool):
                    parser.error(f'{key} must be true or false')
            elif action.type:
                try:
                    config[key] = action.type(value)
                except (TypeError, ValueError) as exc:
                    parser.error(f'Invalid {key}: {exc}')
            if action.choices and config[key] not in action.choices:
                parser.error(f'Invalid {key}: {config[key]}')
        parser.set_defaults(**config)
    elif known.settings != ROOT / 'settings.json':
        parser.error(f'Settings file missing: {known.settings}')
    args = parser.parse_args(argv)
    if args.max_frames is not None and args.max_frames < 1:
        parser.error('--max-frames must be positive')
    if not math.isfinite(args.start_seconds) or args.start_seconds < 0:
        parser.error('--start-seconds must be finite and nonnegative')
    if args.target_id is not None and args.target_id < 1:
        parser.error('--target-id must be positive')
    return args


def display_roi_to_source(rectangle, frame_shape):
    x, y, width, height = rectangle
    if width <= 0 or height <= 0:
        return None
    shown_w, shown_h, left, top = display.display_geometry(frame_shape)
    if x < left or y < top or x + width > left + shown_w or y + height > top + shown_h:
        return None
    source_h, source_w = frame_shape[:2]
    return ((x-left)*source_w/shown_w, (y-top)*source_h/shown_h,
            (x+width-left)*source_w/shown_w, (y+height-top)*source_h/shown_h)


def draw_box(canvas, box, frame_shape, color, label=''):
    points = np.rint(display.to_display(np.asarray(box).reshape(2, 2), frame_shape)).astype(int)
    cv2.rectangle(canvas, tuple(points[0]), tuple(points[1]), color, 2, cv2.LINE_AA)
    if label:
        cv2.putText(canvas, label, tuple(points[0] + [0, -5]),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, color, 1, cv2.LINE_AA)


def render_preview(frame, attention, behavior, emotion, path, fps, processing_fps,
                   engine, notice='', paused=False, id_text=None, latest_row=None):
    analysis = attention.tracking
    canvas = display.render_fhd(frame, analysis, sidebar=False)
    for box in attention.toy_boxes:
        draw_box(canvas, box, frame.shape, (80, 210, 80), 'TOY')
    for box in attention.hand_boxes:
        draw_box(canvas, box, frame.shape, (255, 140, 40), 'HAND')
    if emotion.face_bbox is not None:
        draw_box(canvas, emotion.face_bbox, frame.shape, (180, 180, 255), 'FACE')
    if behavior.seat.seat_roi is not None:
        draw_box(canvas, behavior.seat.seat_roi, frame.shape, (60, 220, 255), 'FIXED SEAT')
    panel = Image.new('RGB', (420, 1080), (22, 25, 31))
    painter = ImageDraw.Draw(panel)
    y = 16

    def text(value, size=22, color=(228, 232, 239), gap=8):
        nonlocal y
        for line in display._wrapped_lines(str(value), size, 388):
            painter.text((16, y), line, font=display._font(size), fill=color)
            y += size + gap

    text('表情與行為整合分析', 28, (110, 208, 255))
    text(path.name, 18)
    text(f'影片 {analysis.timestamp_ms/1000:.2f} 秒  / FPS {fps:.2f}', 18)
    text(f'處理 {processing_fps:.1f} FPS', 18)
    text(f'目標 ID {analysis.target_id} ／代次 {analysis.selection_generation}')
    observed = [t.track_id for t in analysis.tracks if t.observed]
    text('可選 ID：' + ', '.join(map(str, observed)), 18)
    text('表情：' + EMOTIONS[emotion.label], 24)
    if emotion.probability is not None:
        text(f'正向分數 {emotion.probability:.3f}', 18)
    text('玩具觸發：' + ('未知' if attention.playing is None else '是' if attention.playing else '否'))
    for name in BODY_FIELDS:
        state = behavior.statuses[name]
        text(f'{LABELS[name]}：{STATES.get(state, state)}', 22,
             (255, 194, 90) if state == 'active' else (228, 232, 239))
    if engine.seat.baseline_ready:
        text('座位：已固定', 20, (120, 235, 170))
    else:
        remaining = engine.seat.manual_remaining_seconds(analysis.timestamp_ms)
        text(f'座位：校正中／手動剩餘 {remaining:.1f} 秒', 18)
    if latest_row:
        text('已完成 5 秒窗口：' + str(latest_row.get('window_start_seconds', '')), 18)
    text('空白：暫停  I：輸入目標 ID', 18)
    text('R：重新選人  C：取消座位', 18)
    text('S：手動座位（取消後 10 秒內）', 18)
    text('N：下一部  Q / Esc：結束', 18)
    if paused:
        text('已暫停，影片時間不推進', 20, (255, 205, 100))
    if id_text is not None:
        text(f'輸入 ID：{id_text}_  Enter 確認', 22, (255, 205, 100))
    if notice:
        text(notice, 18, (255, 205, 100))
    canvas[:, 1500:] = cv2.cvtColor(np.asarray(panel), cv2.COLOR_RGB2BGR)
    return canvas


def window_visible():
    try:
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        return False


def interact(render, analyzer, engine, timestamp_ms, frame_shape, controls, delay_ms):
    """No frame/model updates during pause, ID input or manual ROI selection."""
    paused, id_text, notice = False, None, ''
    while True:
        canvas = render(notice, paused, id_text)
        cv2.imshow(WINDOW_NAME, canvas)
        raw_key = cv2.waitKey(30 if paused or id_text is not None else delay_ms)
        key = raw_key & 0xFF
        if not window_visible():
            return 'quit'
        if id_text is not None:
            if key == 27:
                id_text = None
                notice = '已取消輸入 ID'
            elif key in (10, 13):
                try:
                    selected = analyzer.select_target(int(id_text))
                except ValueError as exc:
                    notice = str(exc)
                else:
                    engine.reset()
                    controls.append({'time_seconds': timestamp_ms/1000, 'action': 'select_id', 'id': selected})
                    notice = f'已指定 ID {selected}，從下一影格開始'
                    id_text = None
            elif key in (8, 127):
                id_text = id_text[:-1]
            elif ord('0') <= key <= ord('9') and len(id_text) < 9:
                id_text += chr(key)
            continue
        if key in (27, ord('q'), ord('Q')):
            return 'quit'
        if key in (ord('n'), ord('N')):
            return 'next'
        if key == ord(' '):
            paused = not paused
            continue
        if key in (ord('i'), ord('I')):
            id_text, notice = '', ''
            continue
        if key in (ord('r'), ord('R')):
            analyzer.reset_selection()
            engine.reset()
            controls.append({'time_seconds': timestamp_ms/1000, 'action': 'reselect'})
            notice = '已重選，固定座位保留，下一影格開始收集'
            continue
        if key in (ord('c'), ord('C')):
            engine.clear_seat(timestamp_ms)
            controls.append({'time_seconds': timestamp_ms/1000, 'action': 'cancel_seat'})
            notice = '已取消座位；10 秒影片時間內可按 S 框選'
            continue
        if key in (ord('s'), ord('S')):
            if not engine.seat.manual_available(timestamp_ms):
                notice = '請先按 C 取消座位，並在 10 秒內按 S'
                continue
            name = 'Select seated PERSON box - ENTER confirm / ESC cancel'
            try:
                rectangle = cv2.selectROI(name, canvas, showCrosshair=True, fromCenter=False)
            finally:
                cv2.destroyWindow(name)
            roi = display_roi_to_source(rectangle, frame_shape)
            if roi is not None and engine.set_seat_roi(roi, timestamp_ms, frame_shape):
                controls.append({'time_seconds': timestamp_ms/1000, 'action': 'manual_seat', 'roi': roi})
                notice = '手動座位已立即固定'
            else:
                notice = '取消或無效框，繼續收集穩定追蹤框'
            continue
        if not paused:
            return 'continue'


def play_video(path, detector, toy_model, emotion_predictor, args, pipeline_config, seat_config):
    args.output.mkdir(parents=True, exist_ok=True)
    tag = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ') + '_' + uuid.uuid4().hex[:8]
    prefix = args.output / (path.stem + '_' + tag)
    csv_path = prefix.with_name(prefix.name + '_windows.csv')
    metadata_path = prefix.with_name(prefix.name + '_metadata.json')
    frame_path = prefix.with_name(prefix.name + '_frames.csv') if args.frame_log else None
    video_path = prefix.with_name(prefix.name + '_annotated.mp4') if args.export_video else None
    metadata = {'schema_version': 1, 'video': str(path), 'controls': [], 'end_reason': 'error'}
    capture, pipeline = None, None
    stream, frame_stream, writer, analyzer, logger = None, None, None, None, None
    processed, last_timestamp, last_frame_index = 0, None, None
    counts = Counter()
    timings = deque(maxlen=30)
    action, end_reason = 'continue', 'error'
    try:
        capture = cv2.VideoCapture(str(path))
        metadata.update({
                'pipeline': asdict(pipeline_config), 'seat': asdict(seat_config),
                'motion': asdict(MotionConfig(confidence=args.landmark_confidence)),
                'emotion_labels': {'0': 'neutral', '1': 'positive', '2': 'no_detection'},
                'emotion_missing_rule': 'valid frames must exceed half of each ID/generation group; ties are 2',
                'window_rule': 'fixed source [0,5),[5,10); rows per target_id and selection_generation',
                'window_specification': WINDOW_SPEC,
                'toy_rule': 'IoU>threshold; >=3 trigger frames in each source second; >=3 interactive seconds per window',
                'body_rule': '>=5 active frames; 0 means threshold not reached; inspect valid/unknown counts',
                'seat_rule': (f'{seat_config.baseline_seconds:g} continuous observed-box seconds, '
                              f'center distance <={seat_config.max_displacement_ratio:g} initial width; '
                              'arithmetic mean; frozen across reselections; '
                              f'IoU threshold {seat_config.iou_threshold:g}'),
                'toy_model_names': toy_model.names,
                'device': str(emotion_predictor.device), 'detector': detector.provenance,
                'models': {str(p.name): {'path': str(p), 'sha256': sha256(p)}
                           for p in (args.toy_model, args.emotion_model, args.pose_model, args.detector_model)},
                'python': sys.version, 'python_executable': sys.executable,
                'packages': {name: importlib.metadata.version(name) for name in
                             ('numpy', 'mediapipe', 'opencv-contrib-python', 'torch', 'torchvision', 'ultralytics')},
                'files': {'windows': str(csv_path), 'frames': str(frame_path) if frame_path else None,
                          'annotated_video': str(video_path) if video_path else None}})
        if not capture.isOpened():
            raise OSError(f'Cannot open video: {path}')
        fps = capture.get(cv2.CAP_PROP_FPS)
        fps = fps if math.isfinite(fps) and fps > 0 else 30.0
        metadata['source_fps'] = fps
        metadata['source_frames_reported'] = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if args.start_seconds and not capture.set(cv2.CAP_PROP_POS_MSEC, args.start_seconds*1000):
            raise OSError('Video backend cannot seek to requested position')
        position = capture.get(cv2.CAP_PROP_POS_FRAMES)
        frame_index = max(0, round(position)) if math.isfinite(position) else round(args.start_seconds*fps)
        if args.start_seconds and frame_index == 0:
            frame_index = round(args.start_seconds*fps)
        pipeline = TrackingPipeline(detector, args.pose_model, fps, pipeline_config)
        analyzer = AttentionAnalyzer(toy_model, pipeline, fps, iou_threshold=args.toy_iou_threshold)
        engine = BehaviorEngine(MotionConfig(confidence=args.landmark_confidence), seat_config)
        stream = csv_path.open('x', encoding='utf-8-sig', newline='')
        logger = WindowLog(stream)
        frame_fields = ['frame_index', 'time_seconds', 'target_id', 'selection_generation',
                        'toy_trigger', 'emotion', 'positive_probability', *BODY_FIELDS,
                        'seat_roi', 'seat_calibration_seconds', 'seat_calibration_samples',
                        'seat_displacement_ratio', 'seat_reference_source', 'seat_detail',
                        'tracking_state', 'emotion_detail', 'toy_detail']
        if frame_path:
            frame_stream = frame_path.open('x', encoding='utf-8-sig', newline='')
            frame_writer = csv.DictWriter(frame_stream, fieldnames=frame_fields)
            frame_writer.writeheader()
        pending_id = args.target_id
        while args.max_frames is None or processed < args.max_frames:
            started = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                if not processed:
                    raise OSError('No decodable frames')
                end_reason = 'end_video'
                break
            timestamp = capture.get(cv2.CAP_PROP_POS_MSEC)
            if (not math.isfinite(timestamp) or timestamp < 0
                    or (last_timestamp is None and frame_index > 0 and timestamp <= 0)
                    or (last_timestamp is not None and timestamp <= last_timestamp)):
                timestamp = frame_index*1000/fps
                counts['timestamp_fallback_frames'] += 1
            timestamp = max(0 if last_timestamp is None else last_timestamp+1, round(timestamp))
            attention = analyzer.process(frame, timestamp, frame_index)
            behavior = engine.update(attention.tracking, frame.shape)
            emotion = emotion_predictor.process(frame, attention)
            logger.add(timestamp, attention.tracking.target_id, attention.tracking.selection_generation,
                       attention.playing, emotion.label, behavior.statuses, source_frame_index=frame_index)
            counts[f'emotion_{emotion.label}'] += 1
            counts['toy_trigger_frames'] += attention.playing is True
            counts.update(name for name in BODY_FIELDS if behavior.statuses[name] == 'active')
            if frame_stream:
                frame_writer.writerow(dict(frame_index=frame_index, time_seconds=timestamp/1000,
                    target_id=attention.tracking.target_id, selection_generation=attention.tracking.selection_generation,
                    toy_trigger='' if attention.playing is None else int(attention.playing),
                    emotion=emotion.label, positive_probability=emotion.probability,
                    **{name: behavior.statuses[name] for name in BODY_FIELDS},
                    seat_roi=json.dumps(behavior.seat.seat_roi),
                    seat_calibration_seconds=behavior.seat.calibration_seconds,
                    seat_calibration_samples=behavior.seat.calibration_samples,
                    seat_displacement_ratio=behavior.seat.displacement_ratio,
                    seat_reference_source=behavior.seat.reference_source,
                    seat_detail=behavior.seat.detail, tracking_state=attention.tracking.state,
                    emotion_detail=emotion.detail, toy_detail=attention.detail))
            last_timestamp, last_frame_index = timestamp, frame_index
            processed += 1
            frame_index += 1
            if pending_id is not None and any(t.track_id == pending_id and t.observed for t in attention.tracking.tracks):
                analyzer.select_target(pending_id)
                engine.reset()
                metadata['controls'].append({'time_seconds': timestamp/1000, 'action': 'cli_select_id', 'id': pending_id})
                pending_id = None
            speed = len(timings)/sum(timings) if timings else 0.0

            def render(notice='', paused=False, id_text=None):
                # Seat controls apply immediately, while the other displayed
                # predictions remain the last actually analyzed frame.
                shown_behavior = behavior
                current_roi = engine.seat.seat_roi
                if current_roi != behavior.seat.seat_roi:
                    state = 'unknown' if engine.seat.baseline_ready else 'calibrating'
                    shown_behavior = replace(behavior, statuses=dict(behavior.statuses, seat_leave=state),
                        seat=replace(behavior.seat, seat_roi=current_roi))
                return render_preview(frame, attention, shown_behavior, emotion, path, fps, speed,
                                      engine, notice, paused, id_text, logger.rows[-1] if logger.rows else None)

            if video_path:
                if writer is None:
                    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (1920, 1080))
                    if not writer.isOpened():
                        raise OSError('Cannot create annotated video')
                writer.write(render())
            timings.append(max(1e-6, time.perf_counter()-started))
            if not args.headless:
                delay = max(1, math.ceil((1/fps-(time.perf_counter()-started))*1000))
                action = interact(render, analyzer, engine, timestamp, frame.shape, metadata['controls'], delay)
            if processed % 150 == 0:
                stream.flush()
                if frame_stream:
                    frame_stream.flush()
                print(f'[PROGRESS] {path.name}: {processed} frames, source {timestamp/1000:.2f}s', flush=True)
            if action in ('quit', 'next'):
                end_reason = action
                break
        else:
            end_reason = 'max_frames'
        if pending_id is not None:
            metadata['unresolved_requested_id'] = pending_id
        metadata['final_seat_roi'] = engine.seat.seat_roi
        return action != 'quit'
    except KeyboardInterrupt:
        end_reason = 'interrupted'
        raise
    except Exception as exc:
        metadata['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        primary_error = sys.exc_info()[1]
        cleanup_errors = []

        def cleanup(name, operation):
            # One failed close must not prevent any remaining resource from
            # being released, nor replace an inference/startup exception.
            try:
                operation()
                return True
            except BaseException as exc:
                cleanup_errors.append((name, exc))
                return False

        if logger is not None:
            cleanup('finish_windows', lambda: logger.finish(
                end_seconds=None if last_timestamp is None else last_timestamp/1000 + 1/fps))
        analyzer_closed = analyzer is not None and cleanup('close_analyzer', analyzer.close)
        # AttentionAnalyzer owns its pipeline on successful close. Keep a
        # separate fallback for constructor failures and interrupted closes.
        if pipeline is not None and not analyzer_closed:
            cleanup('close_pipeline', pipeline.close)
        if capture is not None:
            cleanup('release_capture', capture.release)
        if writer is not None:
            cleanup('release_writer', writer.release)
        if stream is not None:
            cleanup('close_windows_csv', stream.close)
        if frame_stream is not None:
            cleanup('close_frames_csv', frame_stream.close)
        if cleanup_errors:
            if primary_error is None:
                end_reason = 'error'
                metadata['error'] = f'{type(cleanup_errors[0][1]).__name__}: {cleanup_errors[0][1]}'
            metadata['cleanup_errors'] = [
                {'operation': name, 'error': f'{type(exc).__name__}: {exc}'}
                for name, exc in cleanup_errors]
        metadata.update(end_reason=end_reason, frames=processed, last_source_frame_index=last_frame_index,
                        last_source_seconds=None if last_timestamp is None else last_timestamp/1000,
                        window_rows=len(logger.rows) if logger is not None else 0, counts=dict(counts))
        cleanup('save_metadata', lambda: save_json(metadata_path, metadata))
        cleanup('print_result', lambda: print('[RESULT] ' + json.dumps({
            'csv': str(csv_path), 'metadata': str(metadata_path), 'frames': processed,
            'reason': end_reason}, ensure_ascii=True), flush=True))
        if cleanup_errors and primary_error is None:
            raise cleanup_errors[0][1]


def choose_input_path():
    """Choose a user's video without depending on sibling research projects."""
    import tkinter as tk
    from tkinter import filedialog
    chooser = tk.Tk()
    try:
        chooser.withdraw()
        selected = filedialog.askopenfilename(
            parent=chooser, title='Select input video',
            filetypes=[('Video files', '*.mp4 *.avi *.mov *.mkv *.wmv *.m4v *.webm *.mpg *.mpeg *.mts *.m2ts *.ts *.3gp'),
                       ('All files', '*.*')])
        return Path(selected).resolve() if selected else None
    finally:
        chooser.destroy()


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.input is None:
            if args.headless:
                raise ValueError('--input is required with --headless')
            args.input = choose_input_path()
            if args.input is None:
                return 0
        for path in (args.toy_model, args.emotion_model, args.pose_model, args.detector_model):
            if not path.is_file():
                raise OSError(f'Model missing: {path}')
        videos = display.find_videos(args.input)
        if not videos:
            raise ValueError('No videos found; specify --input')
        pipeline_config = PipelineConfig(init_seconds=args.init_seconds, min_samples=args.min_samples,
            landmark_confidence=args.landmark_confidence, auto_reset_seconds=0.0, retain_target_on_loss=False,
            pose_recovery_seconds=args.pose_recovery_seconds, pose_reselect_seconds=args.pose_reselect_seconds,
            require_posture_landmarks=False)
        seat_config = SeatConfig(baseline_seconds=args.seat_baseline_seconds,
            max_displacement_ratio=args.seat_max_displacement_ratio, iou_threshold=args.seat_iou_threshold,
            confidence=args.landmark_confidence)
        emotion = DanEmotionPredictor(args.emotion_model, device=args.device)
        toy = YOLO(str(args.toy_model))
        toy.to(str(emotion.device))
        detector = PersonDetector(args.detector_model, args.detector_device)
        if not args.headless:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            cv2.resizeWindow(WINDOW_NAME, 1920, 1080)
        failures = 0
        for path in videos:
            try:
                if not play_video(path, detector, toy, emotion, args, pipeline_config, seat_config):
                    break
            except (OSError, ValueError, RuntimeError, cv2.error) as exc:
                failures += 1
                print(f'[ERROR] {path.name}: {exc}', file=sys.stderr, flush=True)
        return 1 if failures else 0
    except KeyboardInterrupt:
        return 0
    except (OSError, ValueError, RuntimeError, cv2.error) as exc:
        print(f'[ERROR] {exc}', file=sys.stderr, flush=True)
        return 1
    finally:
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    raise SystemExit(main())
