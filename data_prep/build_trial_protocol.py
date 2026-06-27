#!/usr/bin/env python3
"""Build dataset-independent ASV trial protocol CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import wave
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TIDYVOICE_TRIALS = (
    PROJECT_ROOT
    / "data"
    / "tidyvoice"
    / "trial_pairs_dev"
    / "TidyVocieX_Dev_trialPairs.txt"
)
DEFAULT_TIDYVOICE_AUDIO_ROOT = (
    PROJECT_ROOT / "data" / "tidyvoice" / "TidyVoiceX_ASV" / "TidyVoiceX_Dev"
)
DEFAULT_VOXCELEB1B_TRIALS = PROJECT_ROOT / "data" / "voxceleb1b" / "list_test_bilingual.txt"
DEFAULT_VOXCELEB1B_LANGS = PROJECT_ROOT / "data" / "voxceleb1b" / "vox1_lang_label.csv"
DEFAULT_VOXCELEB1_AUDIO_ROOT = PROJECT_ROOT / "data" / "voxceleb1"

FIELDNAMES = [
    "dataset",
    "split",
    "trial_id",
    "label",
    "target",
    "enroll_utt",
    "test_utt",
    "enroll_speaker",
    "test_speaker",
    "enroll_language",
    "test_language",
    "language_pair",
    "language_condition",
    "enroll_duration_sec",
    "test_duration_sec",
    "enroll_audio_exists",
    "test_audio_exists",
]


def parse_trial_line(line: str, line_number: int) -> Tuple[str, str, str]:
    parts = line.strip().split()
    if len(parts) != 3:
        raise ValueError(f"Line {line_number}: expected 3 columns, got {len(parts)}")
    label, enroll_utt, test_utt = parts
    if label not in {"0", "1"}:
        raise ValueError(f"Line {line_number}: label must be 0 or 1, got {label!r}")
    return label, enroll_utt, test_utt


def parse_tidyvoice_utterance(utt: str) -> Tuple[str, str]:
    parts = Path(utt).parts
    if len(parts) < 3:
        return "", ""
    return parts[0], parts[1]


def parse_voxceleb_utterance(utt: str, language_labels: Dict[str, str]) -> Tuple[str, str]:
    parts = Path(utt).parts
    speaker = parts[0] if parts else ""
    return speaker, language_labels.get(utt, "")


def language_condition(enroll_language: str, test_language: str) -> str:
    if not enroll_language or not test_language:
        return "unknown"
    if enroll_language == test_language:
        return "same_language"
    return "cross_language"


def wav_duration(path: Path, cache: Dict[Path, Optional[float]]) -> Optional[float]:
    if path in cache:
        return cache[path]
    try:
        with wave.open(str(path), "rb") as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate) if rate else None
    except (wave.Error, EOFError, FileNotFoundError, OSError):
        duration = None
    cache[path] = duration
    return duration


def audio_metadata(
    path: Path,
    *,
    cache: Dict[Path, Tuple[bool, Optional[float]]],
    skip_durations: bool,
) -> Tuple[bool, Optional[float]]:
    if path in cache:
        return cache[path]

    exists = path.exists()
    duration = None
    if exists and not skip_durations:
        try:
            with wave.open(str(path), "rb") as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                duration = frames / float(rate) if rate else None
        except (wave.Error, EOFError, OSError):
            duration = None

    cache[path] = (exists, duration)
    return cache[path]


def format_duration(duration: Optional[float]) -> str:
    if duration is None:
        return ""
    return f"{duration:.3f}"


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def audio_exists_text(value: bool, skipped: bool) -> str:
    if skipped:
        return "unknown"
    return bool_text(value)


def load_voxceleb_language_labels(path: Path) -> Dict[str, str]:
    labels: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        required = {"filename", "label"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"{path} must contain columns: {sorted(required)}")
        for row in reader:
            filename = row["filename"].strip()
            label = row["label"].strip()
            if filename:
                labels[filename] = label
    return labels


def iter_protocol_rows(
    *,
    dataset: str,
    split: str,
    trial_list: Path,
    audio_root: Path,
    max_trials: Optional[int],
    skip_audio_checks: bool,
    skip_durations: bool,
    language_labels: Optional[Dict[str, str]] = None,
) -> Iterator[dict]:
    audio_cache: Dict[Path, Tuple[bool, Optional[float]]] = {}
    parse_utterance = parse_tidyvoice_utterance
    if dataset == "voxceleb1b":
        if language_labels is None:
            language_labels = {}

        def parse_utterance(utt: str) -> Tuple[str, str]:  # type: ignore[no-redef]
            return parse_voxceleb_utterance(utt, language_labels or {})

    with trial_list.open("r", encoding="utf-8") as file_obj:
        for index, line in enumerate(file_obj, start=1):
            if max_trials is not None and index > max_trials:
                break
            label, enroll_utt, test_utt = parse_trial_line(line, index)
            enroll_speaker, enroll_language = parse_utterance(enroll_utt)
            test_speaker, test_language = parse_utterance(test_utt)

            enroll_audio_path = audio_root / enroll_utt
            test_audio_path = audio_root / test_utt
            enroll_exists = False
            test_exists = False
            enroll_duration = None
            test_duration = None
            if not skip_audio_checks:
                enroll_exists, enroll_duration = audio_metadata(
                    enroll_audio_path,
                    cache=audio_cache,
                    skip_durations=skip_durations,
                )
                test_exists, test_duration = audio_metadata(
                    test_audio_path,
                    cache=audio_cache,
                    skip_durations=skip_durations,
                )

            condition = language_condition(enroll_language, test_language)
            language_pair = ""
            if enroll_language and test_language:
                language_pair = f"{enroll_language}--{test_language}"

            yield {
                "dataset": dataset,
                "split": split,
                "trial_id": f"{dataset}:{split}:{index:08d}",
                "label": label,
                "target": "target" if label == "1" else "nontarget",
                "enroll_utt": enroll_utt,
                "test_utt": test_utt,
                "enroll_speaker": enroll_speaker,
                "test_speaker": test_speaker,
                "enroll_language": enroll_language,
                "test_language": test_language,
                "language_pair": language_pair,
                "language_condition": condition,
                "enroll_duration_sec": format_duration(enroll_duration),
                "test_duration_sec": format_duration(test_duration),
                "enroll_audio_exists": audio_exists_text(enroll_exists, skip_audio_checks),
                "test_audio_exists": audio_exists_text(test_exists, skip_audio_checks),
            }


def update_summary(summary: dict, row: dict) -> None:
    summary["rows"] += 1
    summary["labels"].update([row["label"]])
    summary["targets"].update([row["target"]])
    summary["language_conditions"].update([row["language_condition"]])
    if row["enroll_language"]:
        summary["languages"].update([row["enroll_language"]])
    if row["test_language"]:
        summary["languages"].update([row["test_language"]])
    if row["language_pair"]:
        summary["language_pairs"].update([row["language_pair"]])
    if row["enroll_speaker"]:
        summary["speakers"].add(row["enroll_speaker"])
    if row["test_speaker"]:
        summary["speakers"].add(row["test_speaker"])
    if row["enroll_audio_exists"] == "false":
        summary["missing_enroll_audio"] += 1
    if row["test_audio_exists"] == "false":
        summary["missing_test_audio"] += 1


def serializable_summary(summary: dict) -> dict:
    return {
        "dataset": summary["dataset"],
        "split": summary["split"],
        "rows": summary["rows"],
        "labels": dict(summary["labels"]),
        "targets": dict(summary["targets"]),
        "language_conditions": dict(summary["language_conditions"]),
        "num_languages": len(summary["languages"]),
        "top_languages": summary["languages"].most_common(25),
        "num_language_pairs": len(summary["language_pairs"]),
        "top_language_pairs": summary["language_pairs"].most_common(25),
        "num_speakers": len(summary["speakers"]),
        "missing_enroll_audio": summary["missing_enroll_audio"],
        "missing_test_audio": summary["missing_test_audio"],
    }


def write_protocol(
    rows: Iterable[dict],
    output: Optional[Path],
    summary_output: Optional[Path],
    dataset: str,
    split: str,
    summary_only: bool,
    progress_every: int,
) -> None:
    summary = {
        "dataset": dataset,
        "split": split,
        "rows": 0,
        "labels": Counter(),
        "targets": Counter(),
        "language_conditions": Counter(),
        "languages": Counter(),
        "language_pairs": Counter(),
        "speakers": set(),
        "missing_enroll_audio": 0,
        "missing_test_audio": 0,
    }

    if summary_only:
        for row in rows:
            update_summary(summary, row)
            if progress_every and summary["rows"] % progress_every == 0:
                print(f"processed_rows={summary['rows']}", file=sys.stderr, flush=True)
    else:
        if output is None:
            raise ValueError("output is required unless summary_only is enabled")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=FIELDNAMES)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
                update_summary(summary, row)
                if progress_every and summary["rows"] % progress_every == 0:
                    print(f"processed_rows={summary['rows']}", file=sys.stderr, flush=True)

    final_summary = serializable_summary(summary)
    if summary_output:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        with summary_output.open("w", encoding="utf-8") as file_obj:
            json.dump(final_summary, file_obj, indent=2, sort_keys=True)
            file_obj.write("\n")

    print(json.dumps(final_summary, indent=2, sort_keys=True), file=sys.stderr)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, help="Output protocol CSV path.")
    parser.add_argument("--summary-output", type=Path, help="Optional JSON summary path.")
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only write/read the summary; do not write a protocol CSV.",
    )
    parser.add_argument("--max-trials", type=int, help="Limit rows for inspection/debugging.")
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    parser.add_argument(
        "--skip-audio-checks",
        action="store_true",
        help="Do not check whether audio files exist. Useful for fast protocol summaries.",
    )
    parser.add_argument(
        "--skip-durations",
        action="store_true",
        help="Do not read WAV headers to compute utterance durations.",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    tidyvoice = subparsers.add_parser("tidyvoice", help="Build a TidyVoiceX-ASV trial table.")
    add_common_args(tidyvoice)
    tidyvoice.add_argument("--split", default="dev")
    tidyvoice.add_argument("--trial-list", type=Path, default=DEFAULT_TIDYVOICE_TRIALS)
    tidyvoice.add_argument("--audio-root", type=Path, default=DEFAULT_TIDYVOICE_AUDIO_ROOT)

    voxceleb1b = subparsers.add_parser("voxceleb1b", help="Build a VoxCeleb1-B trial table.")
    add_common_args(voxceleb1b)
    voxceleb1b.add_argument("--split", default="test")
    voxceleb1b.add_argument("--trial-list", type=Path, default=DEFAULT_VOXCELEB1B_TRIALS)
    voxceleb1b.add_argument("--language-labels", type=Path, default=DEFAULT_VOXCELEB1B_LANGS)
    voxceleb1b.add_argument("--audio-root", type=Path, default=DEFAULT_VOXCELEB1_AUDIO_ROOT)

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    dataset_id = "tidyvoice_asv" if args.dataset == "tidyvoice" else args.dataset

    if not args.trial_list.exists():
        raise SystemExit(f"Trial list not found: {args.trial_list}")
    if args.max_trials is not None and args.max_trials <= 0:
        raise SystemExit("--max-trials must be positive")
    if not args.summary_only and args.output is None:
        raise SystemExit("--output is required unless --summary-only is used")
    if args.summary_only and args.summary_output is None:
        raise SystemExit("--summary-output is required when --summary-only is used")

    language_labels = None
    if args.dataset == "voxceleb1b":
        if not args.language_labels.exists():
            raise SystemExit(f"Language-label CSV not found: {args.language_labels}")
        language_labels = load_voxceleb_language_labels(args.language_labels)

    rows = iter_protocol_rows(
        dataset=dataset_id,
        split=args.split,
        trial_list=args.trial_list,
        audio_root=args.audio_root,
        max_trials=args.max_trials,
        skip_audio_checks=args.skip_audio_checks,
        skip_durations=args.skip_durations,
        language_labels=language_labels,
    )
    write_protocol(
        rows,
        args.output,
        args.summary_output,
        dataset_id,
        args.split,
        args.summary_only,
        args.progress_every,
    )


if __name__ == "__main__":
    main()
