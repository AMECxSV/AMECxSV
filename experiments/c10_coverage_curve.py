from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from c10 import (
    C_VALUE,
    CHUNKSIZE,
    FEATURE_MODE,
    MAX_EVAL_ROWS_PER_CLASS,
    MAX_TRAIN_ROWS_PER_CLASS,
    MODEL_SETTING,
    binary_mlp_outputs,
    feature_matrix,
)
from selective import (
    confidence_from_llr,
    confidence_threshold,
    decision_metrics,
    decisions_from_confidence,
)
from common import (
    BASELINE_DIR,
    DATASET_NAME,
    EMBEDDINGS,
    add_metric_ci,
    bootstrap_decision_ci95,
    load_fixed_splits,
    metric_block,
    require_cuda,
)


OUTPUT_CSV = BASELINE_DIR / "tidyvoice_c10_coverage_curve.csv"

COVERAGE_TARGETS = [
    1.00,
    0.98,
    0.95,
    0.90,
    0.85,
    0.80,
    0.75,
    0.70,
    0.60,
    0.50,
]


def risk_summary(confidence: np.ndarray, accepted: np.ndarray) -> dict[str, float]:
    rejected = ~accepted
    row = {
        "confidence_mean": float(np.mean(confidence)),
        "confidence_accepted_mean": float(np.mean(confidence[accepted])) if np.any(accepted) else math.nan,
        "confidence_rejected_mean": float(np.mean(confidence[rejected])) if np.any(rejected) else math.nan,
        "confidence_accepted_min": float(np.min(confidence[accepted])) if np.any(accepted) else math.nan,
        "confidence_rejected_max": float(np.max(confidence[rejected])) if np.any(rejected) else math.nan,
    }
    return row


def curve_row(
    coverage_target: float,
    threshold: float | None,
    selection_split: str,
    labels: np.ndarray,
    llrs: np.ndarray,
    posterior_target: np.ndarray,
    confidence: np.ndarray,
) -> dict[str, float | int | str]:
    decisions = decisions_from_confidence(posterior_target, confidence, threshold)
    accepted = decisions != "reject"

    row: dict[str, float | int | str] = {
        "dataset": DATASET_NAME,
        "split": "test",
        "embedding": "six_embedding_fusion",
        "calib": "C10-CURVE",
        "feature_mode": FEATURE_MODE,
        "model": "rich_feature_mlp",
        "selection_split": selection_split,
        "hidden_dim": MODEL_SETTING["hidden_dim"],
        "dropout": MODEL_SETTING["dropout"],
        "learning_rate": MODEL_SETTING["learning_rate"],
        "epochs": MODEL_SETTING["epochs"],
        "batch_size": MODEL_SETTING["batch_size"],
        "C": C_VALUE,
        "coverage_target": coverage_target,
        "confidence_threshold": math.nan if threshold is None else threshold,
    }
    row.update(decision_metrics(labels, decisions))
    row.update(
        bootstrap_decision_ci95(
            labels,
            decisions,
            desc=f"C10 coverage={coverage_target:g} decisions",
        )
    )
    row.update(risk_summary(confidence, accepted))

    if np.any(accepted):
        accepted_metrics = metric_block(llrs[accepted], llrs[accepted], labels[accepted])
        for key, value in accepted_metrics.items():
            row[f"accepted_{key}"] = value
        add_metric_ci(
            row,
            llrs[accepted],
            llrs[accepted],
            labels[accepted],
            desc=f"C10 coverage={coverage_target:g} accepted",
            prefix="accepted_",
        )
    else:
        for key in (
            "n",
            "target_n",
            "nontarget_n",
            "eer_pct",
            "eer_threshold",
            "Cllr",
            "minDCF_p001",
            "actDCF_p001",
            "minDCF_p01",
            "actDCF_p01",
            "accuracy",
            "precision",
            "recall",
            "f1",
        ):
            row[f"accepted_{key}"] = math.nan

    labels_bool = labels.astype(bool)
    rejected = ~accepted
    if np.any(rejected):
        binary_target = posterior_target >= 0.5
        row["rejected_n"] = int(np.sum(rejected))
        row["rejected_error_rate"] = float(np.mean(binary_target[rejected] != labels_bool[rejected]))
        row["rejected_target_rate"] = float(np.mean(labels_bool[rejected]))
        row["rejected_target_n"] = int(np.sum(labels_bool[rejected]))
        row["rejected_nontarget_n"] = int(np.sum(~labels_bool[rejected]))
    else:
        row["rejected_error_rate"] = math.nan
        row["rejected_target_rate"] = math.nan
        row["rejected_target_n"] = 0
        row["rejected_nontarget_n"] = 0
    row["accepted_error_rate"] = math.nan if not np.any(accepted) else 1.0 - float(row["covered_acc"])
    return row


def write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    device = require_cuda()
    train, validation, test = load_fixed_splits(
        EMBEDDINGS,
        CHUNKSIZE,
        MAX_TRAIN_ROWS_PER_CLASS,
        MAX_EVAL_ROWS_PER_CLASS,
        return_validation=True,
    )

    train_features = feature_matrix(train)
    validation_features = None if validation is None else feature_matrix(validation)
    test_features = feature_matrix(test)
    train_labels = train["label"].to_numpy(dtype=np.int8, copy=False)
    validation_labels = None if validation is None else validation["label"].to_numpy(dtype=np.int8, copy=False)
    test_labels = test["label"].to_numpy(dtype=np.int8, copy=False)
    selection_split = "train" if validation is None else "validation"

    _, train_llrs, validation_llrs, test_llrs = binary_mlp_outputs(
        train_features,
        validation_features,
        test_features,
        train_labels,
        validation_labels,
        device,
    )

    selection_llrs = train_llrs if validation_llrs is None else validation_llrs
    _, selection_confidence = confidence_from_llr(selection_llrs)
    posterior_target, confidence = confidence_from_llr(test_llrs)

    rows: list[dict[str, float | int | str]] = []
    with tqdm(COVERAGE_TARGETS, desc="C10 coverage curve", unit="coverage") as progress:
        for coverage_target in progress:
            threshold = None if coverage_target >= 1.0 else confidence_threshold(selection_confidence, coverage_target)
            rows.append(curve_row(coverage_target, threshold, selection_split, test_labels, test_llrs, posterior_target, confidence))

    write_rows(OUTPUT_CSV, rows)
    print(f"wrote {OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
