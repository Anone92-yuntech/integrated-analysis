import csv
import io
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from window_log import BODY_FIELDS, FIELDNAMES, WINDOW_SPEC, WindowLog


class WindowLogTests(unittest.TestCase):
    def make(self, **kwargs):
        stream = io.StringIO(newline="")
        return WindowLog(stream, **kwargs), stream

    def add(self, log, t, *, target=1, generation=0, toy=None, emotion=2,
            states=None, index=None):
        return log.add(t, target, generation, toy, emotion, states or {},
                       source_frame_index=index)

    def test_three_interactive_seconds_required(self):
        log, _ = self.make()
        for second in range(3):
            for frame in range(3):
                self.add(log, second * 1000 + frame * 10, toy=True)
        row = log.finish(2.03)[0]
        self.assertEqual(row["toy_interaction"], 1)
        self.assertEqual(row["toy_trigger_frames"], 9)
        self.assertEqual(row["toy_interactive_seconds"], 3)
        self.assertTrue(row["Partial_Window"])

    def test_nine_frames_in_one_second_do_not_make_three_seconds(self):
        log, _ = self.make()
        for frame in range(9):
            self.add(log, frame * 10, toy=True)
        row = log.finish(.09)[0]
        self.assertEqual(row["toy_interaction"], 0)
        self.assertEqual(row["toy_interactive_seconds"], 1)

    def test_two_frames_in_each_second_do_not_trigger(self):
        log, _ = self.make()
        for second in range(5):
            for frame in range(2):
                self.add(log, second * 1000 + frame * 10, toy=True)
        row = log.finish(4.02)[0]
        self.assertEqual(row["toy_trigger_frames"], 10)
        self.assertEqual(row["toy_interactive_seconds"], 0)
        self.assertEqual(row["toy_interaction"], 0)

    def test_source_second_boundary_is_half_open(self):
        log, _ = self.make()
        for t in (0, 10, 999.9, 1000, 1010):
            self.add(log, t, toy=True)
        row = log.finish(1.02)[0]
        seconds = json.loads(row["toy_second_counts_json"])
        self.assertEqual(seconds["0"]["trigger_frames"], 3)
        self.assertEqual(seconds["1"]["trigger_frames"], 2)
        self.assertEqual(row["toy_interactive_seconds"], 1)

    def test_five_active_frames_are_inclusive_and_nonconsecutive(self):
        log, _ = self.make()
        for i in range(9):
            self.add(log, i * 100, states={name: i % 2 == 0 for name in BODY_FIELDS})
        row = log.finish(.9)[0]
        for name in BODY_FIELDS:
            self.assertEqual(row[name], 1)
            self.assertEqual(row[f"{name}_active_frames"], 5)
            self.assertEqual(row[f"{name}_valid_frames"], 9)

    def test_four_active_frames_do_not_trigger(self):
        log, _ = self.make()
        for i in range(4):
            self.add(log, i * 100, states={name: "active" for name in BODY_FIELDS})
        row = log.finish(.4)[0]
        self.assertTrue(all(row[name] == 0 for name in BODY_FIELDS))

    def test_left_and_right_states_cannot_accidentally_make_both(self):
        log, _ = self.make()
        for i in range(10):
            self.add(log, i * 100, states={"left_wave": "active", "right_wave": "active"})
        row = log.finish(1)[0]
        self.assertEqual(row["both_wave"], 0)
        self.assertEqual(row["both_wave_unknown_frames"], 10)

    def test_body_unknown_counts_are_preserved(self):
        log, _ = self.make()
        for i, status in enumerate(("active", "normal", "unknown", "calibrating", "candidate", None)):
            self.add(log, i * 100, states={"table_rest": status})
        row = log.finish(.6)[0]
        self.assertEqual(row["table_rest"], 0)
        self.assertEqual(row["table_rest_valid_frames"], 2)
        self.assertEqual(row["table_rest_unknown_frames"], 4)

    def test_positive_priority_does_not_require_majority(self):
        log, _ = self.make()
        for i in range(20):
            self.add(log, i * 100, emotion=int(i < 5))
        row = log.finish(2)[0]
        self.assertEqual(row["emotion"], 1)
        self.assertEqual(row["emotion_positive_frames"], 5)
        self.assertEqual(row["emotion_neutral_frames"], 15)

    def test_four_positive_frames_with_valid_majority_are_neutral(self):
        log, _ = self.make()
        for i, emotion in enumerate((1, 1, 1, 1, 0, 0, 2)):
            self.add(log, i * 100, emotion=emotion)
        row = log.finish(.7)[0]
        self.assertEqual(row["emotion"], 0)
        self.assertEqual(row["emotion_missing_frames"], 1)

    def test_exactly_half_valid_is_missing_even_with_five_positives(self):
        log, _ = self.make()
        for i in range(10):
            self.add(log, i * 100, emotion=1 if i < 5 else 2)
        row = log.finish(1)[0]
        self.assertEqual(row["emotion_valid_ratio"], .5)
        self.assertEqual(row["emotion"], 2)

    def test_just_over_half_valid_allows_positive(self):
        log, _ = self.make()
        for i, emotion in enumerate((1, 1, 1, 1, 1, 0, 2, 2, 2, 2)):
            self.add(log, i * 100, emotion=emotion)
        self.assertEqual(log.finish(1)[0]["emotion"], 1)

    def test_all_missing_is_not_neutral(self):
        log, _ = self.make()
        self.add(log, 0)
        row = log.finish(.04)[0]
        self.assertEqual(row["emotion"], 2)
        self.assertEqual(row["emotion_missing_frames"], 1)

    def test_id_and_generation_have_independent_denominators(self):
        log, _ = self.make()
        for i in range(5):
            self.add(log, i * 100, target=1, generation=0, emotion=1)
        for i in range(5, 10):
            self.add(log, i * 100, target=2, generation=0, emotion=2)
        for i in range(10, 14):
            self.add(log, i * 100, target=1, generation=1, emotion=1)
        rows = log.finish(1.4)
        self.assertEqual(len(rows), 3)
        self.assertEqual([row["emotion"] for row in rows], [1, 2, 0])
        self.assertEqual([row["frame_count"] for row in rows], [5, 5, 4])
        self.assertTrue(all(row["selection_partial"] for row in rows))

    def test_repeated_identity_has_one_row_and_disjoint_segments(self):
        log, _ = self.make()
        self.add(log, 0, target=1, index=0)
        self.add(log, 100, target=2, index=1)
        self.add(log, 200, target=1, index=2)
        rows = log.finish(.3)
        self.assertEqual(len(rows), 2)
        row = rows[0]
        self.assertEqual(row["observed_start_seconds"], 0)
        self.assertEqual(row["observed_end_seconds"], .2)
        self.assertEqual(row["first_source_frame_index"], 0)
        self.assertEqual(row["last_source_frame_index"], 2)
        segments = json.loads(row["observation_segments_json"])
        self.assertEqual(len(segments), 2)
        self.assertEqual([segment["frame_count"] for segment in segments], [1, 1])

    def test_unselected_group_cannot_accumulate_evidence(self):
        log, _ = self.make()
        for i in range(5):
            self.add(log, i * 100, target=None, toy=True, emotion=1,
                     states=dict.fromkeys(BODY_FIELDS, "active"))
        row = log.finish(.5)[0]
        self.assertEqual(row["emotion"], 2)
        self.assertEqual(row["toy_interaction"], 0)
        self.assertEqual(row["toy_unknown_frames"], 5)
        self.assertTrue(row["unselected"])
        for name in BODY_FIELDS:
            self.assertEqual(row[name], 0)
            self.assertEqual(row[f"{name}_unknown_frames"], 5)

    def test_five_second_boundary_never_double_counts(self):
        log, _ = self.make(max_gap_seconds=10)
        self.add(log, 0, emotion=1)
        self.add(log, 4999.9, emotion=1)
        emitted = self.add(log, 5000, emotion=1)
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0]["frame_count"], 2)
        self.assertEqual(emitted[0]["window_start_seconds"], 0)
        row = log.finish(5.1)[0]
        self.assertEqual(row["window_start_seconds"], 5)
        self.assertEqual(row["frame_count"], 1)

    def test_gaps_do_not_synthesize_windows(self):
        log, _ = self.make()
        self.add(log, 0, index=0)
        emitted = self.add(log, 16000, index=400)
        tail = log.finish(16.04)
        self.assertEqual([row["window_start_seconds"] for row in emitted + tail], [0, 15])
        self.assertTrue(all(row["source_gap"] for row in emitted + tail))
        self.assertTrue(all(row["Partial_Window"] for row in emitted + tail))

    def test_same_bucket_gap_keeps_actual_observation_segments(self):
        log, _ = self.make()
        self.add(log, 0)
        self.add(log, 2500)
        row = log.finish(2.54)[0]
        self.assertEqual(row["frame_count"], 2)
        self.assertTrue(row["source_gap"])
        self.assertEqual(len(json.loads(row["observation_segments_json"])), 2)

    def test_full_window_and_partial_clip_boundaries(self):
        log, _ = self.make()
        for i in range(50):
            self.add(log, i * 100)
        self.assertFalse(log.finish(5)[0]["Partial_Window"])
        log, _ = self.make()
        self.add(log, 8200)
        row = log.finish(8.3)[0]
        self.assertEqual(row["window_start_seconds"], 5)
        self.assertEqual(row["observed_start_seconds"], 8.2)
        self.assertTrue(row["Partial_Window"])

    def test_duplicate_and_backward_frames_rejected_before_counting(self):
        log, _ = self.make()
        self.add(log, 0, index=10)
        for t, index in ((0, 11), (-1, 11), (100, 10), (100, 9)):
            with self.subTest(t=t, index=index), self.assertRaises(ValueError):
                self.add(log, t, index=index)
        self.assertEqual(log.finish(.1)[0]["frame_count"], 1)

    def test_invalid_values_do_not_consume_source_frame(self):
        log, _ = self.make()
        bad_values = [dict(toy=3), dict(emotion=None), dict(emotion=3),
                      dict(states={"head_shake": "unsupported"}), dict(generation=-1)]
        for values in bad_values:
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.add(log, 0, **values)
        self.add(log, 0, emotion=0)
        self.assertEqual(log.finish(.1)[0]["frame_count"], 1)

    def test_finish_is_idempotent_and_stream_is_caller_owned(self):
        log, stream = self.make()
        self.add(log, 0)
        self.assertEqual(len(log.finish(.04)), 1)
        before = stream.getvalue()
        self.assertEqual(log.finish(.04), [])
        self.assertEqual(stream.getvalue(), before)
        self.assertFalse(stream.closed)
        with self.assertRaises(RuntimeError):
            self.add(log, 100)

    def test_empty_run_only_writes_header(self):
        log, stream = self.make()
        self.assertEqual(log.finish(), [])
        self.assertEqual(list(csv.DictReader(io.StringIO(stream.getvalue()))), [])

    def test_invalid_endpoint_can_be_corrected(self):
        log, _ = self.make()
        self.add(log, 1000)
        for endpoint in (.9, float("nan"), -1):
            with self.assertRaises(ValueError):
                log.finish(endpoint)
        self.assertEqual(len(log.finish(1.1)), 1)

    def test_csv_schema_and_json_are_round_trippable(self):
        log, stream = self.make()
        self.add(log, 0, toy=True, emotion=0, index=12)
        log.finish(.04)
        reader = csv.DictReader(io.StringIO(stream.getvalue()))
        self.assertEqual(tuple(reader.fieldnames), FIELDNAMES)
        row = next(reader)
        self.assertEqual(row["first_source_frame_index"], "12")
        self.assertEqual(json.loads(row["toy_second_counts_json"])["0"]["trigger_frames"], 1)
        self.assertEqual(WINDOW_SPEC["body_active_frames_per_window"], 5)


if __name__ == "__main__":
    unittest.main()
