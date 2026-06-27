#!/usr/bin/env python3
"""Create calibration-internal speaker-disjoint splits for similarity experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = PROJECT_ROOT / "protocols" / "tidyvoice_calibration.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "similarity" / "outputs" / "internal_splits"
SPLITS = ("backend_fit", "calibrator_train", "selection_val")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", default="similarity_internal_speaker_split_v1")
    parser.add_argument("--backend-fit-ratio", type=float, default=0.60)
    parser.add_argument("--calibrator-train-ratio", type=float, default=0.25)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def stable_hash(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode("utf-8")).hexdigest()


def collect_speakers(protocol: Path, chunksize: int) -> list[str]:
    speakers: set[str] = set()
    for chunk in pd.read_csv(protocol, usecols=["enroll_speaker", "test_speaker"], chunksize=chunksize):
        speakers.update(chunk["enroll_speaker"].dropna().astype(str).unique())
        speakers.update(chunk["test_speaker"].dropna().astype(str).unique())
    return sorted(speakers)


def assign_speakers(speakers: list[str], seed: str, backend_ratio: float, calibrator_ratio: float) -> dict[str, str]:
    if not 0.0 < backend_ratio < 1.0 or not 0.0 < calibrator_ratio < 1.0 or backend_ratio + calibrator_ratio >= 1.0:
        raise ValueError("Ratios must be positive and sum to less than 1.")
    ranked = sorted((stable_hash(seed, speaker), speaker) for speaker in speakers)
    n = len(ranked)
    backend_cut = round(n * backend_ratio)
    calibrator_cut = backend_cut + round(n * calibrator_ratio)
    mapping: dict[str, str] = {}
    for idx, (_, speaker) in enumerate(ranked):
        if idx < backend_cut:
            split = "backend_fit"
        elif idx < calibrator_cut:
            split = "calibrator_train"
        else:
            split = "selection_val"
        mapping[speaker] = split
    return mapping


def write_manifest(output_dir: Path, mapping: dict[str, str], seed: str) -> Path:
    path = output_dir / "speaker_split_manifest.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["speaker_id", "speaker_split", "split_seed", "hash_hex"])
        writer.writeheader()
        for speaker, split in sorted(mapping.items(), key=lambda item: (item[1], item[0])):
            writer.writerow(
                {
                    "speaker_id": speaker,
                    "speaker_split": split,
                    "split_seed": seed,
                    "hash_hex": stable_hash(seed, speaker),
                }
            )
    return path


def split_protocol(protocol: Path, output_dir: Path, mapping: dict[str, str], chunksize: int, overwrite: bool) -> dict[str, object]:
    paths = {split: output_dir / f"tidyvoice_similarity_{split}.csv" for split in SPLITS}
    for path in paths.values():
        if path.exists() and not overwrite:
            raise SystemExit(f"Output already exists: {path}. Use --overwrite to replace it.")
    handles = {split: paths[split].open("w", newline="", encoding="utf-8") for split in SPLITS}
    writers: dict[str, csv.DictWriter] = {}
    counts = {split: 0 for split in SPLITS}
    dropped_cross_partition = 0
    try:
        for chunk in pd.read_csv(protocol, chunksize=chunksize):
            if not writers:
                for split, handle in handles.items():
                    writers[split] = csv.DictWriter(handle, fieldnames=list(chunk.columns))
                    writers[split].writeheader()
            for row in chunk.to_dict(orient="records"):
                enroll_split = mapping.get(str(row["enroll_speaker"]))
                test_split = mapping.get(str(row["test_speaker"]))
                if enroll_split is None or test_split is None or enroll_split != test_split:
                    dropped_cross_partition += 1
                    continue
                output_row = dict(row)
                output_row["split"] = enroll_split
                writers[enroll_split].writerow(output_row)
                counts[enroll_split] += 1
    finally:
        for handle in handles.values():
            handle.close()
    return {
        "paths": {split: str(path) for split, path in paths.items()},
        "rows_by_split": counts,
        "dropped_cross_partition_trials": dropped_cross_partition,
    }


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    speakers = collect_speakers(args.protocol, args.chunksize)
    mapping = assign_speakers(speakers, args.seed, args.backend_fit_ratio, args.calibrator_train_ratio)
    manifest_path = write_manifest(output_dir, mapping, args.seed)
    split_summary = split_protocol(args.protocol, output_dir, mapping, args.chunksize, args.overwrite)
    summary = {
        "protocol": str(args.protocol),
        "seed": args.seed,
        "speaker_counts": {split: sum(1 for value in mapping.values() if value == split) for split in SPLITS},
        "speaker_manifest": str(manifest_path),
        **split_summary,
    }
    summary_path = output_dir / "split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"wrote internal split summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

