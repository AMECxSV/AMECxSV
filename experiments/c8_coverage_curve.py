from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

from selective import (
    confidence_from_llr,
    confidence_threshold,
    decision_metrics,
    decisions_from_confidence,
)
from c8 import (
    C_VALUE,
    CHUNKSIZE,
    MAX_EVAL_ROWS_PER_CLASS,
    MAX_TRAIN_ROWS_PER_CLASS,
    MODEL_SETTING,
    feature_matrix,
    fit_score_stats,
    individual_weights,
    score_matrix,
)
from common import (
    BASELINE_DIR,
    DATASET_NAME,
    EMBEDDINGS,
    Experiment,
    add_metric_ci,
    balanced_sample_weights,
    bootstrap_decision_ci95,
    load_fixed_splits,
    make_rich_model,
    metric_block,
    require_cuda,
)


OUTPUT_CSV = BASELINE_DIR / "tidyvoice_c8_coverage_curve.csv"
FEATURE_MODE = "c8_3_entropy_fusion"

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
    return {
        "confidence_mean": float(np.mean(confidence)),
        "confidence_accepted_mean": float(np.mean(confidence[accepted])) if np.any(accepted) else math.nan,
        "confidence_rejected_mean": float(np.mean(confidence[rejected])) if np.any(rejected) else math.nan,
        "confidence_accepted_min": float(np.min(confidence[accepted])) if np.any(accepted) else math.nan,
        "confidence_rejected_max": float(np.max(confidence[rejected])) if np.any(rejected) else math.nan,
    }


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
        "calib": "C8-CURVE",
        "feature_mode": f"{FEATURE_MODE}_conf_reject",
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
            desc=f"C8 coverage={coverage_target:g} decisions",
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
            desc=f"C8 coverage={coverage_target:g} accepted",
            prefix="accepted_",
        )

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

    train_scores = score_matrix(train)
    validation_scores = None if validation is None else score_matrix(validation)
    test_scores = score_matrix(test)
    train_labels = train["label"].to_numpy(dtype=np.int8, copy=False)
    validation_labels = None if validation is None else validation["label"].to_numpy(dtype=np.int8, copy=False)
    test_labels = test["label"].to_numpy(dtype=np.int8, copy=False)
    selection_split = "train" if validation is None else "validation"

    mean, std = fit_score_stats(train_scores)
    weights = individual_weights(train_scores, train_labels)

    train_features = feature_matrix(FEATURE_MODE, train_scores, mean, std, weights)
    validation_features = None if validation_scores is None else feature_matrix(FEATURE_MODE, validation_scores, mean, std, weights)
    test_features = feature_matrix(FEATURE_MODE, test_scores, mean, std, weights)

    experiment = Experiment("C8-2", FEATURE_MODE, C_VALUE, **MODEL_SETTING)
    model = make_rich_model(experiment, device, num_scores=len(EMBEDDINGS))
    model.fit(
        train_features,
        train_labels,
        balanced_sample_weights(train_labels),
        desc=f"train {FEATURE_MODE}",
        validation_features=validation_features,
        validation_labels=validation_labels,
    )

    train_llrs = model.decision_function(train_features)
    validation_llrs = None if validation_features is None else model.decision_function(validation_features)
    test_llrs = model.decision_function(test_features)

    selection_llrs = train_llrs if validation_llrs is None else validation_llrs
    _, selection_confidence = confidence_from_llr(selection_llrs)
    posterior_target, confidence = confidence_from_llr(test_llrs)

    rows: list[dict[str, float | int | str]] = []
    with tqdm(COVERAGE_TARGETS, desc="C8 coverage curve", unit="coverage") as progress:
        for coverage_target in progress:
            threshold = None if coverage_target >= 1.0 else confidence_threshold(selection_confidence, coverage_target)
            rows.append(curve_row(coverage_target, threshold, selection_split, test_labels, test_llrs, posterior_target, confidence))

    write_rows(OUTPUT_CSV, rows)
    print(f"wrote {OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
