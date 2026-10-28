import contextlib
import copy
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import verify_windows as audit


class VerifyWindowsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.frames_path = self.directory / "run_frames.csv"
        self.windows_path = self.directory / "run_windows.csv"

    def frames(self, count=5, target=7, generation=2, start=0, index=0):
        return [dict(frame_index=index + i, time_seconds=start + i / 10,
                     target_id=target, selection_generation=generation,
                     toy_trigger=1, emotion=1, **dict.fromkeys(audit.BODY_NAMES, "active"))
                for i in range(count)]

    def window(self, count=5, target=7, generation=2, first=0, last=.4, index=0,
               bucket=0, primary=1):
        # Explicit fixture: every frame is positive, toy-triggering and body-active;
        # all samples are in the one source-second bucket supplied here.
        second = int(first)
        row = dict(window_start_seconds=bucket * 5, window_end_seconds=(bucket + 1) * 5,
                   target_id=target, selection_generation=generation,
                   toy_interaction=0, emotion=primary,
                   observed_start_seconds=first, observed_end_seconds=last,
                   observed_span_seconds=last - first,
                   first_source_frame_index=index, last_source_frame_index=index + count - 1,
                   frame_count=count, Partial_Window=True, selection_partial=False,
                   unselected=False, source_gap=False, observation_segments_json="[]",
                   toy_trigger_frames=count, toy_valid_frames=count, toy_unknown_frames=0,
                   toy_interactive_seconds=int(count >= 3),
                   toy_second_counts_json=json.dumps({str(second): {
                       "trigger_frames": count, "valid_frames": count,
                       "unknown_frames": 0, "interactive": int(count >= 3)}}),
                   emotion_neutral_frames=0, emotion_positive_frames=count,
                   emotion_valid_frames=count, emotion_missing_frames=0, emotion_valid_ratio=1.0)
        for name in audit.BODY_NAMES:
            row.update({name: primary, name + "_active_frames": count,
                        name + "_valid_frames": count, name + "_unknown_frames": 0})
        return row

    def write(self, frames, windows):
        with self.frames_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=audit.FRAME_FIELDS)
            writer.writeheader()
            writer.writerows(frames)
        with self.windows_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[*audit.WINDOW_FIELDS,
                "Partial_Window", "source_gap", "observation_segments_json"])
            writer.writeheader()
            writer.writerows(windows)

    def verify(self, frames=None, windows=None):
        self.write(self.frames() if frames is None else frames,
                   [self.window()] if windows is None else windows)
        return audit.verify_windows(self.frames_path, self.windows_path)

    def test_explicit_fixture_passes_and_inputs_are_unchanged(self):
        self.write(self.frames(), [self.window()])
        before = self.frames_path.read_bytes(), self.windows_path.read_bytes()
        result = audit.verify_windows(self.frames_path, self.windows_path)
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(result["checked_frame_rows"], 5)
        self.assertEqual(result["checked_window_rows"], 1)
        self.assertEqual(before, (self.frames_path.read_bytes(), self.windows_path.read_bytes()))

    def test_nine_frames_one_second_cannot_make_toy_interaction(self):
        frames = self.frames(count=9)
        row = self.window(count=9, last=.8)
        self.assertEqual(self.verify(frames, [row])["status"], "passed")
        row["toy_interaction"] = 1
        result = self.verify(frames, [row])
        self.assertEqual(result["status"], "failed")
        self.assertTrue(any(error.get("field") == "toy_interaction" for error in result["errors"]))

    def test_three_seconds_each_three_frames_make_toy_interaction(self):
        frames = self.frames(count=9)
        for i, frame in enumerate(frames):
            frame["time_seconds"] = i // 3 + i % 3 / 10
        row = self.window(count=9, last=2.2)
        row.update(toy_interaction=1, toy_interactive_seconds=3,
                   toy_second_counts_json=json.dumps({str(s): {
                       "trigger_frames": 3, "valid_frames": 3,
                       "unknown_frames": 0, "interactive": 1} for s in range(3)}))
        self.assertEqual(self.verify(frames, [row])["status"], "passed")

    def test_exact_half_missing_beats_five_positive_frames(self):
        frames = self.frames(count=10)
        for frame in frames[5:]:
            frame["emotion"] = 2
        row = self.window(count=10, last=.9)
        row.update(emotion=2, emotion_positive_frames=5, emotion_missing_frames=5,
                   emotion_valid_frames=5, emotion_valid_ratio=.5)
        self.assertEqual(self.verify(frames, [row])["status"], "passed")
        row["emotion"] = 1
        self.assertEqual(self.verify(frames, [row])["status"], "failed")

    def test_more_than_half_valid_with_only_four_positives_is_neutral(self):
        frames = self.frames(count=10)
        for i, frame in enumerate(frames):
            frame["emotion"] = 1 if i < 4 else 0 if i < 6 else 2
        row = self.window(count=10, last=.9)
        row.update(emotion=0, emotion_positive_frames=4, emotion_neutral_frames=2,
                   emotion_missing_frames=4, emotion_valid_frames=6, emotion_valid_ratio=.6)
        self.assertEqual(self.verify(frames, [row])["status"], "passed")

    def test_five_positive_frames_have_priority_over_many_neutral(self):
        frames = self.frames(count=10)
        for frame in frames[5:]:
            frame["emotion"] = 0
        row = self.window(count=10, last=.9)
        row.update(emotion=1, emotion_positive_frames=5, emotion_neutral_frames=5)
        self.assertEqual(self.verify(frames, [row])["status"], "passed")

    def test_generations_never_pool_their_frame_counts(self):
        first = self.frames(count=3)
        second = self.frames(count=3, generation=3, start=.3, index=3)
        rows = [self.window(count=3, last=.2, primary=0),
                self.window(count=3, generation=3, first=.3, last=.5, index=3, primary=0)]
        for row in rows:
            row["selection_partial"] = True
        self.assertEqual(self.verify(first + second, rows)["status"], "passed")
        rows[0]["emotion"] = 1
        self.assertEqual(self.verify(first + second, rows)["status"], "failed")

    def test_different_ids_have_separate_denominators(self):
        frames = self.frames() + self.frames(target=8, start=.5, index=5)
        for frame in frames[5:]:
            frame["emotion"] = 2
        rows = [self.window(), self.window(target=8, first=.5, last=.9, index=5)]
        for row in rows:
            row["selection_partial"] = True
        rows[1].update(emotion=2, emotion_positive_frames=0, emotion_missing_frames=5,
                       emotion_valid_frames=0, emotion_valid_ratio=0)
        self.assertEqual(self.verify(frames, rows)["status"], "passed")

    def test_source_five_second_boundary_has_separate_windows(self):
        frames = self.frames(count=2, start=4.8) + self.frames(count=3, start=5, index=2)
        rows = [self.window(count=2, first=4.8, last=4.9, primary=0),
                self.window(count=3, first=5, last=5.2, index=2, bucket=1, primary=0)]
        self.assertEqual(self.verify(frames, rows)["status"], "passed")

    def test_body_valid_unknown_counts_and_both_wave_are_independent(self):
        frames = self.frames()
        for frame, state in zip(frames, ("active", "normal", "unknown", "calibrating", "candidate")):
            frame["both_wave"] = state
        row = self.window()
        row.update(both_wave=0, both_wave_active_frames=1,
                   both_wave_valid_frames=2, both_wave_unknown_frames=3)
        self.assertEqual(self.verify(frames, [row])["status"], "passed")

    def test_unselected_input_evidence_is_excluded(self):
        frames = self.frames(target=None)
        row = self.window(target=None)
        row.update(emotion=2, toy_trigger_frames=0, toy_valid_frames=0, toy_unknown_frames=5,
                   toy_interactive_seconds=0, emotion_positive_frames=0, emotion_valid_frames=0,
                   emotion_missing_frames=5, emotion_valid_ratio=0, unselected=True,
                   selection_partial=True,
                   toy_second_counts_json=json.dumps({"0": {"trigger_frames": 0,
                       "valid_frames": 0, "unknown_frames": 5, "interactive": 0}}))
        for name in audit.BODY_NAMES:
            row.update({name: 0, name + "_active_frames": 0,
                        name + "_valid_frames": 0, name + "_unknown_frames": 5})
        result = self.verify(frames, [row])
        self.assertEqual(result["status"], "passed", result)
        self.assertEqual(result["unselected_evidence_discarded_frames"], 5)

    def test_duplicate_or_out_of_order_source_frames_fail(self):
        for field, value in (("frame_index", 0), ("time_seconds", 0), ("frame_index", -1)):
            frames = self.frames()
            frames[1][field] = value
            with self.subTest(field=field, value=value):
                self.assertEqual(self.verify(frames)["status"], "failed")

    def test_invalid_input_labels_fail(self):
        for field, value in (("emotion", 3), ("toy_trigger", 2),
                             ("head_shake", "yes"), ("time_seconds", "nan")):
            frames = self.frames()
            frames[0][field] = value
            with self.subTest(field=field):
                self.assertEqual(self.verify(frames)["status"], "failed")

    def test_wrong_primary_and_diagnostic_values_are_detected(self):
        for field in ("toy_interaction", "emotion", *audit.BODY_NAMES, *audit.COUNT_FIELDS,
                      "first_source_frame_index", "last_source_frame_index",
                      "emotion_valid_ratio", "observed_end_seconds"):
            row = self.window()
            row[field] += 1
            with self.subTest(field=field):
                self.assertEqual(self.verify(windows=[row])["status"], "failed")

    def test_wrong_per_second_counts_are_detected(self):
        row = self.window()
        per_second = json.loads(row["toy_second_counts_json"])
        per_second["0"]["trigger_frames"] = 4
        row["toy_second_counts_json"] = json.dumps(per_second)
        self.assertEqual(self.verify(windows=[row])["status"], "failed")

    def test_missing_duplicate_and_extra_window_rows_fail(self):
        for rows in ([], [self.window(), self.window()],
                     [self.window(), self.window(target=99)]):
            with self.subTest(rows=len(rows)):
                self.assertEqual(self.verify(windows=rows)["status"], "failed")

    def test_window_alignment_must_be_exact(self):
        row = self.window()
        row["window_start_seconds"] = .1
        self.assertEqual(self.verify(windows=[row])["status"], "failed")

    def test_empty_inputs_are_valid_without_synthetic_rows(self):
        result = self.verify([], [])
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["expected_window_rows"], 0)

    def test_missing_input_is_json_failure(self):
        result = audit.verify_windows(self.frames_path, self.windows_path)
        self.assertEqual(result["status"], "failed")
        self.assertGreater(result["error_count"], 0)

    def test_missing_columns_are_reported(self):
        self.verify()
        self.frames_path.write_text("frame_index,time_seconds\n0,0\n", encoding="utf-8")
        result = audit.verify_windows(self.frames_path, self.windows_path)
        self.assertEqual(result["status"], "failed")
        self.assertIn("missing CSV columns", result["errors"][0]["message"])

    def test_cli_stdout_only_and_failure_exit_code(self):
        self.verify()
        before = sorted(self.directory.iterdir())
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = audit.main([str(self.frames_path), str(self.windows_path)])
        self.assertEqual(result, 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "passed")
        self.assertEqual(sorted(self.directory.iterdir()), before)
        self.windows_path.write_text("bad\n", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(audit.main([str(self.frames_path), str(self.windows_path)]), 1)

    def test_report_is_exclusive_and_limited_to_own_logs(self):
        self.verify()
        logs = self.directory / "logs"
        report = logs / "audit.json"
        args = [str(self.frames_path), str(self.windows_path), "--report", str(report)]
        with mock.patch.object(audit, "LOGS_ROOT", logs):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(audit.main(args), 0)
            original = report.read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(audit.main(args), 1)
            self.assertEqual(report.read_bytes(), original)
            outside = self.directory / "outside.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(audit.main([*args[:-1], str(outside)]), 1)
            self.assertFalse(outside.exists())


if __name__ == "__main__":
    unittest.main()
