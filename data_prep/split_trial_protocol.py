#!/usr/bin/env python3
"""Create deterministic speaker-disjoint protocol splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "protocols" / "tidyvoice_dev.csv"
DEFAULT_MANIFEST = PROJECT_ROOT / "protocols" / "tidyvoice_speaker_split_manifest.csv"
DEFAULT_CALIBRATION = PROJECT_ROOT / "protocols" / "tidyvoice_calibration.csv"
DEFAULT_VALIDATION = PROJECT_ROOT / "protocols" / "tidyvoice_validation.csv"
DEFAULT_TEST = PROJECT_ROOT / "protocols" / "tidyvoice_test.csv"
DEFAULT_UNUSED = PROJECT_ROOT / "protocols" / "tidyvoice_unused_cross_split.csv"
DEFAULT_SUMMARY = PROJECT_ROOT / "protocols" / "tidyvoice_split_summary.json"
DEFAULT_SEED = "tidyvoice_asv_speaker_split_v1"


def stable_hash(seed: str, value: str) -> str:
    payload = f"{seed}\0{value}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--calibration-output", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--validation-output", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--test-output", type=Path, default=DEFAULT_TEST)
    parser.add_argument("--unused-output", type=Path, default=DEFAULT_UNUSED)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--calibration-ratio", type=float, default=0.60)
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.0,
        help=(
            "Optional speaker-level validation ratio. The remaining speakers are "
            "assigned to test. Use 0.20 with --calibration-ratio 0.60 for a "
            "60/20/20 speaker-disjoint split."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=1_000_000)
    return parser.parse_args(argv)


def collect_speakers(protocol: Path) -> tuple[list[str], dict]:
    speakers: set[str] = set()
    source_splits = Counter()
    datasets = Counter()
    rows = 0

    with protocol.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        required = {"dataset", "split", "enroll_speaker", "test_speaker"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"{protocol} must contain columns: {sorted(required)}")

        for row in reader:
            rows += 1
            datasets.update([row["dataset"]])
            source_splits.update([row["split"]])
            if row["enroll_speaker"]:
                speakers.add(row["enroll_speaker"])
            if row["test_speaker"]:
                speakers.add(row["test_speaker"])

    return sorted(speakers), {
        "rows": rows,
        "datasets": dict(datasets),
        "source_splits": dict(source_splits),
    }


def assign_speakers(
    speakers: list[str],
    *,
    seed: str,
    calibration_ratio: float,
    validation_ratio: float,
) -> tuple[dict[str, str], list[dict]]:
    if not 0.0 < calibration_ratio < 1.0:
        raise ValueError("--calibration-ratio must be between 0 and 1")
    if not 0.0 <= validation_ratio < 1.0:
        raise ValueError("--validation-ratio must be between 0 and 1")
    if calibration_ratio + validation_ratio >= 1.0:
        raise ValueError("--calibration-ratio + --validation-ratio must be less than 1")
    if len(speakers) < 2:
        raise ValueError("At least two speakers are required for speaker-disjoint splitting")

    ranked = sorted((stable_hash(seed, speaker), speaker) for speaker in speakers)
    split_ratios = {"calibration": calibration_ratio, "test": 1.0 - calibration_ratio}
    if validation_ratio > 0.0:
        split_ratios = {
            "calibration": calibration_ratio,
            "validation": validation_ratio,
            "test": 1.0 - calibration_ratio - validation_ratio,
        }
    split_counts = speaker_split_counts(len(ranked), split_ratios)

    assignments: dict[str, str] = {}
    manifest_rows: list[dict] = []
    cumulative_counts: list[tuple[str, int]] = []
    total = 0
    for split in ("calibration", "validation", "test"):
        count = split_counts.get(split, 0)
        if count <= 0:
            continue
        total += count
        cumulative_counts.append((split, total))

    for rank, (hash_hex, speaker) in enumerate(ranked, start=1):
        split = next(
            split_name
            for split_name, upper_rank in cumulative_counts
            if rank <= upper_rank
        )
        assignments[speaker] = split
        manifest_rows.append(
            {
                "speaker_id": speaker,
                "speaker_split": split,
                "split_seed": seed,
                "hash_rank": rank,
                "hash_hex": hash_hex,
            }
        )

    return assignments, manifest_rows


def speaker_split_counts(num_speakers: int, split_ratios: dict[str, float]) -> dict[str, int]:
    raw_counts = {split: num_speakers * ratio for split, ratio in split_ratios.items()}
    counts = {split: int(value) for split, value in raw_counts.items()}
    remaining = num_speakers - sum(counts.values())
    ranked_remainders = sorted(
        split_ratios,
        key=lambda split: (raw_counts[split] - counts[split], split == "calibration"),
        reverse=True,
    )
    for split in ranked_remainders[:remaining]:
        counts[split] += 1

    active_splits = [split for split, ratio in split_ratios.items() if ratio > 0.0]
    for split in active_splits:
        if counts[split] == 0:
            donor = max(counts, key=counts.get)
            if counts[donor] <= 1:
                raise ValueError("Not enough speakers to allocate all requested splits")
            counts[donor] -= 1
            counts[split] = 1
    return counts


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["speaker_id", "speaker_split", "split_seed", "hash_rank", "hash_hex"]
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def empty_summary() -> dict:
    return {
        "rows": 0,
        "labels": Counter(),
        "targets": Counter(),
        "language_conditions": Counter(),
        "language_pairs": Counter(),
        "speakers": set(),
    }


def update_summary(summary: dict, row: dict) -> None:
    summary["rows"] += 1
    summary["labels"].update([row.get("label", "")])
    summary["targets"].update([row.get("target", "")])
    summary["language_conditions"].update([row.get("language_condition", "")])
    if row.get("language_pair"):
        summary["language_pairs"].update([row["language_pair"]])
    if row.get("enroll_speaker"):
        summary["speakers"].add(row["enroll_speaker"])
    if row.get("test_speaker"):
        summary["speakers"].add(row["test_speaker"])


def serialize_summary(summary: dict) -> dict:
    return {
        "rows": summary["rows"],
        "labels": {k: v for k, v in summary["labels"].items() if k},
        "targets": {k: v for k, v in summary["targets"].items() if k},
        "language_conditions": {
            k: v for k, v in summary["language_conditions"].items() if k
        },
        "num_language_pairs": len(summary["language_pairs"]),
        "top_language_pairs": summary["language_pairs"].most_common(25),
        "num_speakers": len(summary["speakers"]),
    }


def split_protocol(
    *,
    protocol: Path,
    assignments: dict[str, str],
    output_paths: dict[str, Path],
    progress_every: int,
) -> dict:
    for path in output_paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    split_summaries = {split: empty_summary() for split in output_paths}
    unused_reasons = Counter()

    with protocol.open("r", encoding="utf-8", newline="") as input_obj:
        reader = csv.DictReader(input_obj)
        required = {"split", "enroll_speaker", "test_speaker"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"{protocol} must contain columns: {sorted(required)}")
        fieldnames = list(reader.fieldnames or [])

        output_objs = {
            split: path.open("w", encoding="utf-8", newline="")
            for split, path in output_paths.items()
        }
        try:
            writers = {
                split: csv.DictWriter(
                    output_obj,
                    fieldnames=fieldnames,
                    lineterminator="\n",
                )
                for split, output_obj in output_objs.items()
            }
            for writer in writers.values():
                writer.writeheader()

            for row_index, row in enumerate(reader, start=1):
                enroll_split = assignments.get(row["enroll_speaker"])
                test_split = assignments.get(row["test_speaker"])
                if enroll_split and test_split and enroll_split == test_split:
                    output_split = enroll_split
                else:
                    output_split = "unused_cross_split"
                    reason = "missing_speaker_split"
                    if enroll_split and test_split and enroll_split != test_split:
                        reason = "cross_split_speakers"
                    unused_reasons.update([reason])

                output_row = dict(row)
                output_row["split"] = output_split
                writers[output_split].writerow(output_row)
                update_summary(split_summaries[output_split], output_row)

                if progress_every and row_index % progress_every == 0:
                    counts = {
                        split: summary["rows"]
                        for split, summary in split_summaries.items()
                    }
                    print(f"processed_rows={row_index} split_rows={counts}", file=sys.stderr)
        finally:
            for output_obj in output_objs.values():
                output_obj.close()

    real_splits = [split for split in output_paths if split != "unused_cross_split"]
    speaker_overlap: dict[str, int] = {}
    for left_index, left in enumerate(real_splits):
        for right in real_splits[left_index + 1 :]:
            speaker_overlap[f"{left}_{right}"] = len(
                split_summaries[left]["speakers"] & split_summaries[right]["speakers"]
            )
    return {
        "splits": {
            split: serialize_summary(summary)
            for split, summary in split_summaries.items()
        },
        "unused_reasons": dict(unused_reasons),
        "speaker_overlap": speaker_overlap,
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if not args.protocol.exists():
        raise SystemExit(f"Protocol not found: {args.protocol}")

    speakers, source_summary = collect_speakers(args.protocol)
    assignments, manifest_rows = assign_speakers(
        speakers,
        seed=args.seed,
        calibration_ratio=args.calibration_ratio,
        validation_ratio=args.validation_ratio,
    )
    write_manifest(args.manifest_output, manifest_rows)
    output_paths = {
        "calibration": args.calibration_output,
        "test": args.test_output,
        "unused_cross_split": args.unused_output,
    }
    if args.validation_ratio > 0.0:
        output_paths = {
            "calibration": args.calibration_output,
            "validation": args.validation_output,
            "test": args.test_output,
            "unused_cross_split": args.unused_output,
        }

    split_result = split_protocol(
        protocol=args.protocol,
        assignments=assignments,
        output_paths=output_paths,
        progress_every=args.progress_every,
    )

    speaker_counts = Counter(assignments.values())
    summary = {
        "protocol": str(args.protocol),
        "split_seed": args.seed,
        "calibration_ratio": args.calibration_ratio,
        "validation_ratio": args.validation_ratio,
        "test_ratio": 1.0 - args.calibration_ratio - args.validation_ratio,
        "source": source_summary,
        "num_speakers": len(speakers),
        "speaker_split_counts": dict(speaker_counts),
        "outputs": {
            "speaker_manifest": str(args.manifest_output),
            **{split: str(path) for split, path in output_paths.items()},
        },
        **split_result,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_output.open("w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2, sort_keys=True)
        file_obj.write("\n")

    print(json.dumps(summary, indent=2, sort_keys=True), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
