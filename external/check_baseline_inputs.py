#!/usr/bin/env python3
"""Validate AMEC external-baseline inputs before running matched comparisons."""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

from external_common import (
    DEFAULT_CONFIG,
    config_path,
    git_commit,
    load_config,
    resolve_path,
    utc_timestamp,
)


REQUIRED_TRIAL_COLUMNS = [
    "trial_id",
    "enroll_utt",
    "test_utt",
    "enroll_speaker",
    "test_speaker",
    "enroll_language",
    "test_language",
    "language_condition",
    "enroll_duration_sec",
    "test_duration_sec",
]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--trial-table", type=Path)
    parser.add_argument("--calibration-split", type=Path)
    parser.add_argument("--test-split", type=Path)
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--report-output", type=Path, default=Path("external/outputs/audit_report.md"))
    return parser.parse_args(argv)


def csv_header(path: Path) -> list[str]:
    return pd.read_csv(path, nrows=0).columns.tolist()


def validate_header(path: Path) -> tuple[list[str], list[str]]:
    columns = csv_header(path)
    missing = [column for column in REQUIRED_TRIAL_COLUMNS if column not in columns]
    if "label" not in columns and "target" not in columns:
        missing.append("label_or_target")
    return columns, missing


def summarize_split(path: Path, chunksize: int) -> dict:
    columns = csv_header(path)
    usecols = [
        column
        for column in [
            "trial_id",
            "label",
            "target",
            "split",
            "enroll_speaker",
            "test_speaker",
            "language_condition",
        ]
        if column in columns
    ]
    rows = 0
    labels: Counter[str] = Counter()
    targets: Counter[str] = Counter()
    split_values: Counter[str] = Counter()
    language_conditions: Counter[str] = Counter()
    speakers: set[str] = set()
    trial_ids: set[str] = set()
    duplicate_trial_ids = 0

    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        rows += int(len(chunk))
        if "label" in chunk:
            labels.update(chunk["label"].astype(str))
        if "target" in chunk:
            targets.update(chunk["target"].astype(str))
        if "split" in chunk:
            split_values.update(chunk["split"].astype(str))
        if "language_condition" in chunk:
            language_conditions.update(chunk["language_condition"].astype(str))
        for speaker_col in ["enroll_speaker", "test_speaker"]:
            if speaker_col in chunk:
                speakers.update(chunk[speaker_col].dropna().astype(str))
        ids = chunk["trial_id"].astype(str)
        duplicate_trial_ids += int(ids.duplicated().sum())
        seen_overlap = set(ids) & trial_ids
        duplicate_trial_ids += len(seen_overlap)
        trial_ids.update(ids)

    return {
        "path": str(path),
        "rows": rows,
        "labels": dict(labels),
        "targets": dict(targets),
        "split_values": dict(split_values),
        "language_conditions": dict(language_conditions),
        "speaker_count": len(speakers),
        "trial_id_count": len(trial_ids),
        "duplicate_trial_ids": duplicate_trial_ids,
        "_speakers": speakers,
        "_trial_ids": trial_ids,
    }


def writeability(path: Path) -> tuple[bool, str]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write_probe"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def format_counter(mapping: dict) -> str:
    if not mapping:
        return "{}"
    return ", ".join(f"{key}: {value}" for key, value in sorted(mapping.items()))


def report_lines(
    *,
    trial_table: Path,
    calibration_split: Path,
    test_split: Path,
    audio_root: Path,
    output_root: Path,
    checkpoint_root: Path,
    headers: dict[str, tuple[list[str], list[str]]],
    calibration: dict,
    test: dict,
    writable: tuple[bool, str],
) -> list[str]:
    speaker_overlap = calibration["_speakers"] & test["_speakers"]
    trial_id_overlap = calibration["_trial_ids"] & test["_trial_ids"]
    missing_paths = [
        str(path)
        for path in [trial_table, calibration_split, test_split, audio_root, checkpoint_root]
        if path and not path.exists()
    ]

    lines = [
        "# External Baseline Repository Audit",
        "",
        f"- run_id: audit_{utc_timestamp().replace(':', '').replace('-', '')}",
        f"- timestamp_utc: {utc_timestamp()}",
        f"- git_commit: {git_commit()}",
        "",
        "## Configured Paths",
        "",
        f"- trial_table: `{trial_table}`",
        f"- calibration_split: `{calibration_split}`",
        f"- test_split: `{test_split}`",
        f"- audio_root: `{audio_root}`",
        f"- checkpoint_root: `{checkpoint_root}`",
        f"- output_root: `{output_root}`",
        "",
        "## Header Checks",
        "",
    ]
    for name, (_, missing) in headers.items():
        status = "ok" if not missing else f"missing: {missing}"
        lines.append(f"- {name}: {status}")
    lines.extend(
        [
            "",
            "## Split Summary",
            "",
            f"- calibration rows: {calibration['rows']}",
            f"- calibration speakers: {calibration['speaker_count']}",
            f"- calibration labels: {format_counter(calibration['labels'])}",
            f"- calibration language_condition: {format_counter(calibration['language_conditions'])}",
            f"- test rows: {test['rows']}",
            f"- test speakers: {test['speaker_count']}",
            f"- test labels: {format_counter(test['labels'])}",
            f"- test language_condition: {format_counter(test['language_conditions'])}",
            f"- calibration duplicate trial_id count: {calibration['duplicate_trial_ids']}",
            f"- test duplicate trial_id count: {test['duplicate_trial_ids']}",
            f"- calibration/test speaker overlap: {len(speaker_overlap)}",
            f"- calibration/test trial_id overlap: {len(trial_id_overlap)}",
            "",
            "## Output Checks",
            "",
            f"- output_root writable: {writable[0]} ({writable[1]})",
            "",
            "## Missing Local Artifacts",
            "",
        ]
    )
    if missing_paths:
        lines.extend(f"- `{path}`" for path in missing_paths)
    else:
        lines.append("- none detected among configured core paths")
    lines.extend(
        [
            "",
            "## Assumptions",
            "",
            "- Matched external baselines must align to the exact `trial_id` set in the configured test split.",
            "- Calibration, score normalization, threshold selection, and abstention thresholds must use only the calibration split.",
            "- Missing external checkpoints are blockers for inference mode, not a reason to fabricate metrics.",
            "- Imported score files are acceptable only when every dropped or unmatched trial is counted and logged.",
        ]
    )
    return lines


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    trial_table = resolve_path(args.trial_table or config_path(config, "TIDYVOICE_TRIAL_TABLE"))
    calibration_split = resolve_path(args.calibration_split or config_path(config, "TIDYVOICE_CALIBRATION_SPLIT"))
    test_split = resolve_path(args.test_split or config_path(config, "TIDYVOICE_TEST_SPLIT"))
    audio_root = resolve_path(args.audio_root or config_path(config, "TIDYVOICE_AUDIO_ROOT"))
    output_root = resolve_path(args.output_root or config_path(config, "OUTPUT_ROOT"))
    checkpoint_root = resolve_path(args.checkpoint_root or config_path(config, "CHECKPOINT_ROOT"))

    headers = {
        "trial_table": validate_header(trial_table),
        "calibration_split": validate_header(calibration_split),
        "test_split": validate_header(test_split),
    }
    calibration = summarize_split(calibration_split, args.chunksize)
    test = summarize_split(test_split, args.chunksize)
    writable = writeability(output_root)

    lines = report_lines(
        trial_table=trial_table,
        calibration_split=calibration_split,
        test_split=test_split,
        audio_root=audio_root,
        output_root=output_root,
        checkpoint_root=checkpoint_root,
        headers=headers,
        calibration=calibration,
        test=test,
        writable=writable,
    )
    output = resolve_path(args.report_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    hard_failures: list[str] = []
    for name, (_, missing) in headers.items():
        if missing:
            hard_failures.append(f"{name} missing required columns: {missing}")
    if calibration["duplicate_trial_ids"]:
        hard_failures.append("calibration split has duplicate trial_id values")
    if test["duplicate_trial_ids"]:
        hard_failures.append("test split has duplicate trial_id values")
    if calibration["_speakers"] & test["_speakers"]:
        hard_failures.append("calibration/test speaker overlap is nonzero")
    if calibration["_trial_ids"] & test["_trial_ids"]:
        hard_failures.append("calibration/test trial_id overlap is nonzero")
    if not writable[0]:
        hard_failures.append(f"output root is not writable: {writable[1]}")

    print(f"wrote {output}")
    if hard_failures:
        for failure in hard_failures:
            print(f"ERROR: {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
