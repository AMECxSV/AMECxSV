#!/usr/bin/env python3
"""Run final similarity evaluation from a frozen selection config."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from similarity_scores import trainable_parameter_count


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_MODEL_DIR = PROJECT_ROOT / "experiments"
if str(BASELINE_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_MODEL_DIR))

from common import Experiment, balanced_sample_weights, make_model, metric_block  # noqa: E402


DEFAULT_INPUT = PROJECT_ROOT / "similarity" / "outputs" / "tidyvoice_similarity_scores_full.parquet"
DEFAULT_FROZEN = PROJECT_ROOT / "similarity" / "frozen_similarity_selection.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "similarity" / "outputs" / "final_similarity_comparison.csv"
MLP_SETTING = {
    "hidden_dim": 128,
    "dropout": 0.15,
    "learning_rate": 5.0e-4,
    "epochs": 120,
    "batch_size": 262_144,
}


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-table", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--frozen-selection", type=Path, default=DEFAULT_FROZEN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-split", default="calibration")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--mlp-epochs", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def select_device(name: str):
    import torch

    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    if name == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def write_row(path: Path, row: dict[str, object], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"Output already exists: {path}. Use --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, index=False)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if not args.frozen_selection.exists():
        raise SystemExit(f"Frozen selection is required before final test evaluation: {args.frozen_selection}")
    frozen = json.loads(args.frozen_selection.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen":
        raise SystemExit(f"Frozen selection status is not 'frozen': {frozen.get('status')}")
    feature_columns = [column for column in frozen.get("feature_columns", []) if column]
    if not feature_columns:
        raise SystemExit("Frozen selection contains no feature_columns.")
    frame = pd.read_parquet(args.input_table) if args.input_table.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(args.input_table)
    missing_columns = sorted(set(feature_columns) - set(frame.columns))
    if missing_columns:
        raise SystemExit(f"Input table is missing frozen feature columns: {missing_columns}")
    train = frame[frame["split"].astype(str) == args.train_split].reset_index(drop=True)
    test = frame[frame["split"].astype(str) == args.eval_split].reset_index(drop=True)
    if train.empty or test.empty:
        raise SystemExit(f"Missing train/eval rows for {args.train_split}/{args.eval_split}")
    train_x = train[feature_columns].to_numpy(dtype=np.float64, copy=False)
    test_x = test[feature_columns].to_numpy(dtype=np.float64, copy=False)
    train_y = train["label"].to_numpy(dtype=np.int8, copy=False)
    test_y = test["label"].to_numpy(dtype=np.int8, copy=False)
    if not np.isfinite(train_x).all() or not np.isfinite(test_x).all():
        raise SystemExit("Frozen feature columns contain non-finite values; regenerate/audit scores before final test.")
    model_name = str(frozen.get("selected_model", "logistic"))
    seed = int(frozen.get("selected_seed", 0))
    if model_name == "logistic":
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=seed),
        )
        model.fit(train_x, train_y)
        llrs = model.decision_function(test_x).astype(np.float64)
        parameter_count = len(feature_columns) + 1
    elif model_name == "mlp":
        device = select_device(args.device)
        setting = dict(MLP_SETTING)
        setting["epochs"] = args.mlp_epochs
        experiment = Experiment("SIM-FINAL", str(frozen.get("selected_score_feature_set")), 10.0, seed=seed, **setting)
        model = make_model(experiment, device)
        model.fit(train_x, train_y, balanced_sample_weights(train_y), desc="SIM final")
        llrs = model.decision_function(test_x)
        parameter_count = trainable_parameter_count(len(feature_columns))
    else:
        raise SystemExit(f"Unsupported frozen model: {model_name}")
    row: dict[str, object] = {
        "system": "SIM-FINAL",
        "score_feature_set": frozen.get("selected_score_feature_set"),
        "metadata_features": "as_frozen",
        "coverage": 1.0,
        "feature_count": len(feature_columns),
        "parameter_count": parameter_count,
        "seed": seed,
        "fit_split": args.train_split,
        "evaluation_split": args.eval_split,
        "model": model_name,
        "frozen_selection": str(args.frozen_selection),
    }
    row.update(metric_block(llrs, llrs, test_y))
    write_row(args.output, row, overwrite=args.overwrite)
    print(f"wrote final result: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

