#!/usr/bin/env python3
"""Import or score LI-MSV-TidyVoice2026 outputs on AMEC trial IDs."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from external_common import (
    DEFAULT_CONFIG,
    SCORE_OUTPUT_COLUMNS,
    align_scores_to_trials,
    config_path,
    git_commit,
    load_config,
    load_trial_table,
    make_run_id,
    read_table,
    resolve_path,
    utc_timestamp,
    write_mismatch_log,
    write_table,
)


SOURCE_SYSTEM = "limsv_w2vbert"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("infer", "import_embeddings", "import_scores"), default="import_scores")
    parser.add_argument("--trial-table", type=Path)
    parser.add_argument("--split", default=None)
    parser.add_argument("--score-file", type=Path)
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--match-key", choices=("trial_id", "utterance_pair"), default="trial_id")
    parser.add_argument("--embedding-file", type=Path)
    parser.add_argument("--utterance-column", default="utt")
    parser.add_argument("--vector-column", help="Column containing JSON/comma/space encoded embedding vectors.")
    parser.add_argument("--vector-prefix", default="emb_", help="Prefix for numeric vector columns when --vector-column is omitted.")
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--checkpoint-id", default="")
    parser.add_argument("--inference-command", help="Shell command template for released LI-MSV inference code.")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output-prefix", type=Path, default=Path("external/outputs/limsv_w2vbert_raw"))
    parser.add_argument("--mismatch-log", type=Path, default=Path("external/outputs/limsv_w2vbert_mismatches.json"))
    return parser.parse_args(argv)


def configured_checkpoint(config: dict, args: argparse.Namespace) -> Path:
    if args.checkpoint_path:
        return resolve_path(args.checkpoint_path)
    system_cfg = (config.get("external_systems") or {}).get(SOURCE_SYSTEM, {})
    value = system_cfg.get("checkpoint_path")
    if value:
        return resolve_path(value)
    root = config_path(config, "CHECKPOINT_ROOT", "external/checkpoints")
    return resolve_path(root / SOURCE_SYSTEM)


def parse_vector(value: object) -> np.ndarray:
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=np.float64)
    text = str(value).strip()
    if not text:
        return np.asarray([], dtype=np.float64)
    if text.startswith("["):
        return np.asarray(json.loads(text), dtype=np.float64)
    text = text.replace(",", " ")
    return np.fromstring(text, sep=" ", dtype=np.float64)


def load_embedding_map(args: argparse.Namespace) -> dict[str, np.ndarray]:
    if not args.embedding_file:
        raise SystemExit("--mode import_embeddings requires --embedding-file")
    frame = read_table(args.embedding_file)
    if args.utterance_column not in frame.columns:
        raise ValueError(f"Embedding file must contain {args.utterance_column}")
    if args.vector_column:
        if args.vector_column not in frame.columns:
            raise ValueError(f"Embedding file must contain {args.vector_column}")
        vectors = [parse_vector(value) for value in frame[args.vector_column]]
    else:
        vector_columns = [column for column in frame.columns if column.startswith(args.vector_prefix)]
        if not vector_columns:
            raise ValueError(f"No vector columns found with prefix {args.vector_prefix}")
        vectors = frame[vector_columns].to_numpy(dtype=np.float64)
    mapping: dict[str, np.ndarray] = {}
    for utt, vector in zip(frame[args.utterance_column].astype(str), vectors):
        arr = np.asarray(vector, dtype=np.float64)
        if arr.size == 0 or not np.all(np.isfinite(arr)):
            continue
        mapping[str(utt)] = arr
    return mapping


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0 or not math.isfinite(denom):
        return math.nan
    return float(np.dot(a, b) / denom)


def score_embeddings(args: argparse.Namespace, config: dict) -> Path:
    trial_table = resolve_path(args.trial_table or config_path(config, "TIDYVOICE_TEST_SPLIT"))
    trials = load_trial_table(trial_table, split=args.split)
    mapping = load_embedding_map(args)
    scores: list[float] = []
    missing = 0
    for enroll_utt, test_utt in zip(trials["enroll_utt"].astype(str), trials["test_utt"].astype(str)):
        enroll_vec = mapping.get(enroll_utt)
        test_vec = mapping.get(test_utt)
        if enroll_vec is None or test_vec is None:
            scores.append(math.nan)
            missing += 1
        else:
            scores.append(cosine(enroll_vec, test_vec))
    if missing and not args.allow_partial:
        raise SystemExit(f"Missing embeddings for {missing} trials. Use --allow-partial for diagnostics.")
    raw_scores = trials[["trial_id"]].copy()
    raw_scores["score"] = scores
    if args.allow_partial:
        raw_scores = raw_scores.dropna(subset=["score"])
    path = resolve_path(str(args.output_prefix) + "_embedding_scores.csv")
    write_table(raw_scores, path)
    return path


def run_inference_wrapper(args: argparse.Namespace, config: dict) -> Path:
    checkpoint = configured_checkpoint(config, args)
    if not checkpoint.exists():
        raise SystemExit(
            f"LI-MSV checkpoint not found: {checkpoint}. "
            "Use import_scores/import_embeddings or configure the released checkpoint path."
        )
    if not args.inference_command:
        raise SystemExit("--mode infer requires --inference-command for the released LI-MSV code.")
    generated_scores = resolve_path(str(args.output_prefix) + "_recipe_scores.csv")
    trial_table = resolve_path(args.trial_table or config_path(config, "TIDYVOICE_TEST_SPLIT"))
    command = args.inference_command.format(
        checkpoint=str(checkpoint),
        trial_table=str(trial_table),
        output_scores=str(generated_scores),
    )
    subprocess.run(command, cwd=Path.cwd(), shell=True, check=True)
    if not generated_scores.exists():
        raise SystemExit(f"Inference command completed but did not create {generated_scores}")
    return generated_scores


def import_scores(args: argparse.Namespace, config: dict, score_file: Path) -> tuple[Path, Path]:
    trial_table = resolve_path(args.trial_table or config_path(config, "TIDYVOICE_TEST_SPLIT"))
    trials = load_trial_table(trial_table, split=args.split)
    scores = read_table(score_file)
    checkpoint = configured_checkpoint(config, args)
    checkpoint_id = args.checkpoint_id or (checkpoint.name if checkpoint.exists() else "missing_checkpoint")
    run_id = make_run_id(SOURCE_SYSTEM)
    aligned, mismatch = align_scores_to_trials(
        trials,
        scores,
        score_column=args.score_column,
        match_key=args.match_key,
        source_system=SOURCE_SYSTEM,
        checkpoint_id=checkpoint_id,
        run_id=run_id,
        allow_partial=args.allow_partial,
    )
    mismatch.update(
        {
            "source_system": SOURCE_SYSTEM,
            "checkpoint_path": str(checkpoint),
            "checkpoint_exists": checkpoint.exists(),
            "run_id": run_id,
            "external_training_data_disclosure_required": True,
        }
    )
    write_mismatch_log(mismatch, args.mismatch_log)
    csv_path = resolve_path(str(args.output_prefix) + "_scores.csv")
    parquet_path = resolve_path(str(args.output_prefix) + "_scores.parquet")
    write_table(aligned[SCORE_OUTPUT_COLUMNS], csv_path)
    write_table(aligned[SCORE_OUTPUT_COLUMNS], parquet_path)
    return csv_path, parquet_path


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.mode == "infer":
        score_file = run_inference_wrapper(args, config)
    elif args.mode == "import_embeddings":
        score_file = score_embeddings(args, config)
    else:
        if not args.score_file:
            raise SystemExit("--mode import_scores requires --score-file")
        score_file = resolve_path(args.score_file)
    csv_path, parquet_path = import_scores(args, config, score_file)
    print(f"wrote {csv_path}")
    print(f"wrote {parquet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
