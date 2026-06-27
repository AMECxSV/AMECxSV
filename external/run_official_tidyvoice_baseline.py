#!/usr/bin/env python3
"""Import or wrap inference for the official TidyVoice SimAM-ResNet34 baseline."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Iterable

from external_common import (
    DEFAULT_CONFIG,
    align_scores_to_trials,
    config_path,
    load_config,
    load_trial_table,
    make_run_id,
    read_table,
    resolve_path,
    write_mismatch_log,
    write_table,
)


SOURCE_SYSTEM = "official_tidyvoice_simam_resnet34"


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--mode", choices=("import_scores", "infer"), default="import_scores")
    parser.add_argument("--trial-table", type=Path)
    parser.add_argument("--split", default=None, help="Optional split filter if --trial-table contains multiple splits.")
    parser.add_argument("--score-file", type=Path, help="Existing official score file for import_scores mode.")
    parser.add_argument("--score-column", default="score")
    parser.add_argument("--match-key", choices=("trial_id", "utterance_pair"), default="trial_id")
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--checkpoint-id", default="")
    parser.add_argument("--inference-command", help="Shell command template for official recipe inference mode.")
    parser.add_argument("--allow-partial", action="store_true", help="Write aligned subset while logging dropped trials.")
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("external/outputs/official_tidyvoice_simam_resnet34"),
    )
    parser.add_argument(
        "--mismatch-log",
        type=Path,
        default=Path("external/outputs/official_tidyvoice_simam_resnet34_mismatches.json"),
    )
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
        }
    )
    write_mismatch_log(mismatch, args.mismatch_log)

    csv_path = resolve_path(str(args.output_prefix) + "_scores.csv")
    parquet_path = resolve_path(str(args.output_prefix) + "_scores.parquet")
    write_table(aligned, csv_path)
    write_table(aligned, parquet_path)
    return csv_path, parquet_path


def run_inference_wrapper(args: argparse.Namespace, config: dict) -> Path:
    checkpoint = configured_checkpoint(config, args)
    if not checkpoint.exists():
        raise SystemExit(
            f"Official checkpoint not found: {checkpoint}. "
            "Use --mode import_scores with an existing score file or configure CHECKPOINT_ROOT."
        )
    if not args.inference_command:
        raise SystemExit(
            "--mode infer requires --inference-command because the official recipe is external to this repo."
        )
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


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.mode == "infer":
        score_file = run_inference_wrapper(args, config)
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
