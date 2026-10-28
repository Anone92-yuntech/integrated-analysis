"""Independently audit frame-log -> five-second CSV aggregation.

This module uses only the standard library and never imports the application or
its window logger. Inputs are read-only. A report is written only if explicitly
requested, as a new file inside this project's logs directory.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import sys


BASE_DIR = Path(__file__).resolve().parent
LOGS_ROOT = BASE_DIR / "logs"
BODY_NAMES = ("head_shake", "both_wave", "table_rest", "seat_leave")
FRAME_FIELDS = ("frame_index", "time_seconds", "target_id", "selection_generation",
                "toy_trigger", "emotion", *BODY_NAMES)
COUNT_FIELDS = (
    "frame_count", "toy_trigger_frames", "toy_valid_frames", "toy_unknown_frames",
    "toy_interactive_seconds", "emotion_neutral_frames", "emotion_positive_frames",
    "emotion_valid_frames", "emotion_missing_frames",
    *(field for name in BODY_NAMES for field in
      (name + "_active_frames", name + "_valid_frames", name + "_unknown_frames")),
)
VALUE_FIELDS = ("toy_interaction", "emotion", *BODY_NAMES, *COUNT_FIELDS,
                "observed_start_seconds", "observed_end_seconds", "observed_span_seconds",
                "first_source_frame_index", "last_source_frame_index", "emotion_valid_ratio",
                "selection_partial", "unselected", "toy_second_counts_json")
WINDOW_FIELDS = ("window_start_seconds", "window_end_seconds", "target_id",
                 "selection_generation", *VALUE_FIELDS)
KNOWN_STATES = {"active", "normal", "unknown", "calibrating", "candidate"}


class _Errors:
    def __init__(self):
        self.count = 0
        self.items = []

    def add(self, message, **details):
        self.count += 1
        if len(self.items) < 100:
            self.items.append({"message": message, **details})


def _integer(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field}: expected a nonnegative integer")
    try:
        result = int(value)
    except (ValueError, TypeError):
        raise ValueError(f"{field}: expected a nonnegative integer") from None
    if result < 0:
        raise ValueError(f"{field}: expected a nonnegative integer")
    return result


def _number(value, field):
    try:
        result = float(value)
    except (ValueError, TypeError):
        raise ValueError(f"{field}: expected a finite number") from None
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{field}: expected a finite nonnegative number")
    return result


def _target(value):
    return None if value == "" else _integer(value, "target_id")


def _boolean(value, field):
    if value in ("True", "true", "1"):
        return True
    if value in ("False", "false", "0"):
        return False
    raise ValueError(f"{field}: expected a boolean")


def _reader(path, required):
    stream = Path(path).open("r", encoding="utf-8-sig", newline="")
    reader = csv.DictReader(stream)
    missing = sorted(set(required) - set(reader.fieldnames or ()))
    if missing:
        stream.close()
        raise ValueError("missing CSV columns: " + ", ".join(missing))
    if len(reader.fieldnames) != len(set(reader.fieldnames)):
        stream.close()
        raise ValueError("duplicate CSV column names")
    return stream, reader


def _new_group():
    return {"frames": 0, "neutral": 0, "positive": 0, "missing": 0,
            "toy_seconds": defaultdict(Counter), "body_active": Counter(),
            "body_normal": Counter()}


def _read_frames(path, errors):
    groups = defaultdict(_new_group)
    bucket_totals = Counter()
    parsed_count = 0
    discarded_unselected = 0
    previous_index, previous_time = -1, -1.0
    stream, reader = _reader(path, FRAME_FIELDS)
    with stream:
        for line, raw in enumerate(reader, 2):
            try:
                if None in raw or any(raw[name] is None for name in FRAME_FIELDS):
                    raise ValueError("malformed CSV row")
                index = _integer(raw["frame_index"], "frame_index")
                time = _number(raw["time_seconds"], "time_seconds")
                target = _target(raw["target_id"])
                generation = _integer(raw["selection_generation"], "selection_generation")
                if raw["toy_trigger"] not in ("", "0", "1"):
                    raise ValueError("toy_trigger must be blank, 0 or 1")
                if raw["emotion"] not in ("0", "1", "2"):
                    raise ValueError("emotion must be 0, 1 or 2")
                if any(raw[name] not in KNOWN_STATES for name in BODY_NAMES):
                    raise ValueError("body status must be active/normal/unknown/calibrating/candidate")
                if index <= previous_index:
                    raise ValueError("frame_index is duplicated or not strictly increasing")
                if time <= previous_time:
                    raise ValueError("time_seconds is duplicated or not strictly increasing")
            except ValueError as exc:
                errors.add(str(exc), file="frames", line=line)
                continue
            previous_index, previous_time = index, time
            parsed_count += 1
            bucket = math.floor(time / 5)
            bucket_totals[bucket] += 1
            group = groups[bucket, target, generation]
            if group["frames"] == 0:
                group.update(first_time=time, first_index=index)
            group.update(last_time=time, last_index=index)
            group["frames"] += 1
            seconds = group["toy_seconds"][math.floor(time)]
            if target is None:
                group["missing"] += 1
                seconds["unknown_frames"] += 1
                if (raw["toy_trigger"] != "" or raw["emotion"] != "2"
                        or any(raw[name] in ("active", "normal") for name in BODY_NAMES)):
                    discarded_unselected += 1
                continue
            group[{"0": "neutral", "1": "positive", "2": "missing"}[raw["emotion"]]] += 1
            toy = raw["toy_trigger"]
            if toy == "":
                seconds["unknown_frames"] += 1
            else:
                seconds["valid_frames"] += 1
                if toy == "1":
                    seconds["trigger_frames"] += 1
            for name in BODY_NAMES:
                if raw[name] == "active":
                    group["body_active"][name] += 1
                elif raw[name] == "normal":
                    group["body_normal"][name] += 1
    return groups, bucket_totals, parsed_count, discarded_unselected


def _expected(group, target, total):
    count = group["frames"]
    per_second = {}
    for second, counts in group["toy_seconds"].items():
        per_second[str(second)] = {
            "trigger_frames": counts["trigger_frames"],
            "valid_frames": counts["valid_frames"],
            "unknown_frames": counts["unknown_frames"],
            "interactive": int(counts["trigger_frames"] >= 3),
        }
    active_seconds = sum(second["interactive"] for second in per_second.values())
    emotion = (2 if group["missing"] * 2 >= count else
               1 if group["positive"] >= 5 else 0)
    valid_emotions = count - group["missing"]
    result = {
        "toy_interaction": int(active_seconds >= 3), "emotion": emotion,
        "frame_count": count,
        "toy_trigger_frames": sum(second["trigger_frames"] for second in per_second.values()),
        "toy_valid_frames": sum(second["valid_frames"] for second in per_second.values()),
        "toy_unknown_frames": sum(second["unknown_frames"] for second in per_second.values()),
        "toy_interactive_seconds": active_seconds,
        "toy_second_counts_json": per_second,
        "emotion_positive_frames": group["positive"],
        "emotion_neutral_frames": group["neutral"],
        "emotion_missing_frames": group["missing"],
        "emotion_valid_frames": valid_emotions,
        "emotion_valid_ratio": valid_emotions / count,
        "observed_start_seconds": group["first_time"],
        "observed_end_seconds": group["last_time"],
        "observed_span_seconds": group["last_time"] - group["first_time"],
        "first_source_frame_index": group["first_index"],
        "last_source_frame_index": group["last_index"],
        "selection_partial": target is None or count != total,
        "unselected": target is None,
    }
    for name in BODY_NAMES:
        active = group["body_active"][name]
        valid = active + group["body_normal"][name]
        result.update({name: int(active >= 5), name + "_active_frames": active,
                       name + "_valid_frames": valid, name + "_unknown_frames": count - valid})
    return result


def _read_value(raw, field):
    if field == "toy_second_counts_json":
        value = json.loads(raw[field])
        if not isinstance(value, dict):
            raise ValueError("toy_second_counts_json must contain an object")
        for counts in value.values():
            if (not isinstance(counts, dict) or set(counts) !=
                    {"trigger_frames", "valid_frames", "unknown_frames", "interactive"}
                    or any(type(n) is not int or n < 0 for n in counts.values())):
                raise ValueError("toy_second_counts_json contains invalid counts")
        return value
    if field in ("selection_partial", "unselected"):
        return _boolean(raw[field], field)
    if field.endswith("_seconds") or field == "emotion_valid_ratio":
        return _number(raw[field], field)
    return _integer(raw[field], field)


def verify_windows(frames_csv, windows_csv):
    """Return a JSON-serializable audit; do not modify inputs or run inference."""
    errors = _Errors()
    report = {
        "status": "failed", "method": "independent standard-library recomputation from frame CSV",
        "frames_csv": str(Path(frames_csv).resolve()),
        "windows_csv": str(Path(windows_csv).resolve()),
        "checked_frame_rows": 0, "expected_window_rows": 0, "checked_window_rows": 0,
        "compared_fields": list(VALUE_FIELDS),
        "not_recomputed": {
            "Partial_Window": "exclusive clip endpoint is not included in frame CSV",
            "source_gap": "gap policy is not included in frame CSV",
            "observation_segments_json": "segment/gap policy is not included in frame CSV",
        },
    }
    try:
        groups, totals, frame_count, discarded = _read_frames(frames_csv, errors)
        report.update(checked_frame_rows=frame_count, expected_window_rows=len(groups),
                      unselected_evidence_discarded_frames=discarded)
        seen = set()
        previous_bucket = -1
        stream, reader = _reader(windows_csv, WINDOW_FIELDS)
        with stream:
            for line, raw in enumerate(reader, 2):
                report["checked_window_rows"] += 1
                try:
                    if None in raw or any(raw[name] is None for name in WINDOW_FIELDS):
                        raise ValueError("malformed CSV row")
                    start = _number(raw["window_start_seconds"], "window_start_seconds")
                    end = _number(raw["window_end_seconds"], "window_end_seconds")
                    bucket = math.floor(start / 5)
                    if start != bucket * 5 or end != start + 5:
                        raise ValueError("window must align to source-time [5k,5k+5)")
                    if bucket < previous_bucket:
                        raise ValueError("window rows are not in source-time order")
                    previous_bucket = bucket
                    target = _target(raw["target_id"])
                    generation = _integer(raw["selection_generation"], "selection_generation")
                    key = bucket, target, generation
                    if key in seen:
                        raise ValueError("duplicate window/target/generation row")
                    seen.add(key)
                    if key not in groups:
                        raise ValueError("window identity has no corresponding observed source frames")
                    expected = _expected(groups[key], target, totals[bucket])
                    for field, expected_value in expected.items():
                        try:
                            actual = _read_value(raw, field)
                            matches = (math.isclose(actual, expected_value, rel_tol=1e-9, abs_tol=1e-8)
                                       if isinstance(expected_value, float) else actual == expected_value)
                            if not matches:
                                errors.add("value mismatch", file="windows", line=line,
                                           field=field, expected=expected_value, actual=actual)
                        except (ValueError, TypeError, json.JSONDecodeError) as exc:
                            errors.add(str(exc), file="windows", line=line, field=field)
                    if expected["selection_partial"] and "Partial_Window" in raw:
                        if not _boolean(raw["Partial_Window"], "Partial_Window"):
                            errors.add("identity partial row must also have Partial_Window=True",
                                       file="windows", line=line)
                except (ValueError, TypeError) as exc:
                    errors.add(str(exc), file="windows", line=line)
        for key in groups.keys() - seen:
            errors.add("missing window identity row", bucket_start=key[0] * 5,
                       target_id=key[1], selection_generation=key[2])
    except (OSError, ValueError, csv.Error, UnicodeError) as exc:
        errors.add(str(exc))
    report.update(status="passed" if errors.count == 0 else "failed",
                  error_count=errors.count, errors=errors.items,
                  errors_truncated=errors.count > len(errors.items))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames_csv", type=Path)
    parser.add_argument("windows_csv", type=Path)
    parser.add_argument("--report", type=Path,
                        help="New JSON file under integrated_analysis/logs; relative to project folder")
    args = parser.parse_args(argv)
    report = verify_windows(args.frames_csv, args.windows_csv)
    if args.report is not None:
        try:
            destination = (args.report if args.report.is_absolute() else BASE_DIR / args.report).resolve()
            if not destination.is_relative_to(LOGS_ROOT.resolve()) or destination == LOGS_ROOT.resolve():
                raise ValueError("report must be a new file inside integrated_analysis/logs")
            if destination in (args.frames_csv.resolve(), args.windows_csv.resolve()):
                raise ValueError("report cannot replace an input CSV")
            destination.parent.mkdir(parents=True, exist_ok=True)
            report["report_path"] = str(destination)
            with destination.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
        except (OSError, ValueError) as exc:
            report["status"] = "failed"
            report["error_count"] += 1
            report["errors"].append({"message": str(exc), "file": "report"})
            report.pop("report_path", None)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
