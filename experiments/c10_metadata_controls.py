from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from selective import (
    confidence_from_llr,
    confidence_threshold,
    decision_metrics,
    decisions_from_confidence,
)
from c8 import threshold_for_max_accuracy, threshold_metrics
from common import (
    BASELINE_DIR,
    BEST_MLP_SETTING,
    C5_COLS,
    DATASET_NAME,
    EMBEDDINGS,
    Experiment,
    balanced_sample_weights,
    load_fixed_splits,
    make_rich_model,
    metric_block,
    require_cuda,
)


OUTPUT_CSV = BASELINE_DIR / "tidyvoice_c10_metadata_controls.csv"
SUMMARY_CSV = BASELINE_DIR / "tidyvoice_c8_c10_coverage_auc_summary.csv"
CHUNKSIZE = 250_000
MAX_TRAIN_ROWS_PER_CLASS = None
MAX_EVAL_ROWS_PER_CLASS = None
COVERAGE_TARGET = 0.80
MODEL_SETTING = {key: value for key, value in BEST_MLP_SETTING.items() if key != "c_value"}
C_VALUE = BEST_MLP_SETTING["c_value"]
SEED = 20260625


def score_columns() -> list[str]:
    return [f"score_{embedding}" for embedding in EMBEDDINGS]


def metadata_matrix(frame) -> np.ndarray:
    language = frame["target"].to_numpy(dtype=np.float64, copy=False)[:, None]
    reliability = frame[C5_COLS].to_numpy(dtype=np.float64, copy=False)
    return np.concatenate([language, reliability], axis=1)


def full_feature_matrix(frame) -> np.ndarray:
    scores = frame[score_columns()].to_numpy(dtype=np.float64, copy=False)
    return np.concatenate([scores, metadata_matrix(frame)], axis=1)


def shuffled_metadata_feature_matrix(frame, rng: np.random.Generator) -> np.ndarray:
    scores = frame[score_columns()].to_numpy(dtype=np.float64, copy=False)
    metadata = metadata_matrix(frame)
    order = rng.permutation(metadata.shape[0])
    return np.concatenate([scores, metadata[order]], axis=1)


def metadata_only_feature_matrix(frame) -> np.ndarray:
    return metadata_matrix(frame)


def row_prefix(mode: str, calib: str, feature_count: int, num_scores: int) -> dict[str, float | int | str]:
    return {
        "dataset": DATASET_NAME,
        "split": "test",
        "embedding": "six_embedding_fusion",
        "calib": calib,
        "feature_mode": mode,
        "model": "rich_feature_mlp",
        "hidden_dim": MODEL_SETTING["hidden_dim"],
        "dropout": MODEL_SETTING["dropout"],
        "learning_rate": MODEL_SETTING["learning_rate"],
        "epochs": MODEL_SETTING["epochs"],
        "batch_size": MODEL_SETTING["batch_size"],
        "C": C_VALUE,
        "feature_count": feature_count,
        "num_scores_for_rich_expansion": num_scores,
    }


def fit_outputs(
    mode: str,
    num_scores: int,
    train_features: np.ndarray,
    validation_features: np.ndarray | None,
    test_features: np.ndarray,
    train_labels: np.ndarray,
    validation_labels: np.ndarray | None,
    device,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    experiment = Experiment("C10_CTRL", mode, C_VALUE, **MODEL_SETTING)
    model = make_rich_model(experiment, device, num_scores=num_scores)
    model.fit(
        train_features,
        train_labels,
        balanced_sample_weights(train_labels),
        desc=f"train {mode}",
        validation_features=validation_features,
        validation_labels=validation_labels,
    )
    train_llrs = model.decision_function(train_features)
    validation_llrs = None if validation_features is None else model.decision_function(validation_features)
    test_llrs = model.decision_function(test_features)
    return train_llrs, validation_llrs, test_llrs


def binary_row(
    mode: str,
    feature_count: int,
    num_scores: int,
    selection_llrs: np.ndarray,
    test_llrs: np.ndarray,
    selection_labels: np.ndarray,
    test_labels: np.ndarray,
    selection_split: str,
) -> dict[str, float | int | str]:
    threshold = threshold_for_max_accuracy(selection_llrs, selection_labels)
    row = row_prefix(mode, "C10-CTRL-2", feature_count, num_scores)
    row["selection_split"] = selection_split
    row.update(metric_block(test_llrs, test_llrs, test_labels))
    selected_metrics = threshold_metrics(test_llrs, test_labels, threshold)
    row["final_accuracy"] = selected_metrics["accuracy"]
    row.update({f"threshold_{key}": value for key, value in selected_metrics.items()})
    return row


def reject_row(
    mode: str,
    feature_count: int,
    num_scores: int,
    selection_llrs: np.ndarray,
    test_llrs: np.ndarray,
    selection_split: str,
    test_labels: np.ndarray,
) -> dict[str, float | int | str]:
    _, selection_confidence = confidence_from_llr(selection_llrs)
    posterior_target, confidence = confidence_from_llr(test_llrs)
    threshold = confidence_threshold(selection_confidence, COVERAGE_TARGET)
    decisions = decisions_from_confidence(posterior_target, confidence, threshold)
    accepted = decisions != "reject"

    row = row_prefix(f"{mode}_conf_reject", "C10-CTRL-3", feature_count, num_scores)
    row["selection_split"] = selection_split
    row["coverage_target"] = COVERAGE_TARGET
    row["confidence_threshold"] = threshold
    row["confidence_mean"] = float(np.mean(confidence))
    row["confidence_accepted_mean"] = float(np.mean(confidence[accepted])) if np.any(accepted) else math.nan
    row["confidence_rejected_mean"] = float(np.mean(confidence[~accepted])) if np.any(~accepted) else math.nan
    row["posterior_target_mean"] = float(np.mean(posterior_target))
    row.update(decision_metrics(test_labels, decisions))
    row["final_accuracy"] = row["accuracy"]

    accepted_metrics = metric_block(test_llrs[accepted], test_llrs[accepted], test_labels[accepted])
    for key, value in accepted_metrics.items():
        row[f"accepted_{key}"] = value
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


def normalized_auc(x: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    width = float(x_sorted[-1] - x_sorted[0])
    if width <= 0.0:
        return math.nan
    return float(np.trapezoid(y_sorted, x_sorted) / width)


def coverage_auc_rows() -> list[dict[str, float | str]]:
    import pandas as pd

    rows: list[dict[str, float | str]] = []
    for system, path in (
        ("C8-3", BASELINE_DIR / "tidyvoice_c8_coverage_curve.csv"),
        ("C10-3", BASELINE_DIR / "tidyvoice_c10_coverage_curve.csv"),
    ):
        frame = pd.read_csv(path)
        coverage = frame["coverage"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "system": system,
                "coverage_min": float(np.min(coverage)),
                "coverage_max": float(np.max(coverage)),
                "accepted_eer_auc": normalized_auc(coverage, frame["accepted_eer_pct"].to_numpy(dtype=np.float64)),
                "accepted_cllr_auc": normalized_auc(coverage, frame["accepted_Cllr"].to_numpy(dtype=np.float64)),
                "accepted_actdcf_p01_auc": normalized_auc(
                    coverage,
                    frame["accepted_actDCF_p01"].to_numpy(dtype=np.float64),
                ),
            }
        )
    return rows


def main() -> None:
    device = require_cuda()
    train, validation, test = load_fixed_splits(
        EMBEDDINGS,
        CHUNKSIZE,
        MAX_TRAIN_ROWS_PER_CLASS,
        MAX_EVAL_ROWS_PER_CLASS,
        return_validation=True,
    )

    train_labels = train["label"].to_numpy(dtype=np.int8, copy=False)
    validation_labels = None if validation is None else validation["label"].to_numpy(dtype=np.int8, copy=False)
    test_labels = test["label"].to_numpy(dtype=np.int8, copy=False)
    selection_split = "train" if validation is None else "validation"

    rng_train = np.random.default_rng(SEED + 1)
    rng_validation = np.random.default_rng(SEED + 2)
    rng_test = np.random.default_rng(SEED + 3)

    control_specs = [
        (
            "c10_shuffled_metadata",
            len(EMBEDDINGS),
            shuffled_metadata_feature_matrix(train, rng_train),
            None if validation is None else shuffled_metadata_feature_matrix(validation, rng_validation),
            shuffled_metadata_feature_matrix(test, rng_test),
        ),
        (
            "c10_metadata_only",
            4,
            metadata_only_feature_matrix(train),
            None if validation is None else metadata_only_feature_matrix(validation),
            metadata_only_feature_matrix(test),
        ),
    ]

    rows: list[dict[str, float | int | str]] = []
    for mode, num_scores, train_features, validation_features, test_features in control_specs:
        train_llrs, validation_llrs, test_llrs = fit_outputs(
            mode,
            num_scores,
            train_features,
            validation_features,
            test_features,
            train_labels,
            validation_labels,
            device,
        )
        selection_llrs = train_llrs if validation_llrs is None else validation_llrs
        selection_labels = train_labels if validation_labels is None else validation_labels
        rows.append(
            binary_row(
                mode,
                train_features.shape[1],
                num_scores,
                selection_llrs,
                test_llrs,
                selection_labels,
                test_labels,
                selection_split,
            )
        )
        rows.append(
            reject_row(
                mode,
                train_features.shape[1],
                num_scores,
                selection_llrs,
                test_llrs,
                selection_split,
                test_labels,
            )
        )

    write_rows(OUTPUT_CSV, rows)
    write_rows(SUMMARY_CSV, coverage_auc_rows())
    print(f"wrote {OUTPUT_CSV}", flush=True)
    print(f"wrote {SUMMARY_CSV}", flush=True)


if __name__ == "__main__":
    main()
