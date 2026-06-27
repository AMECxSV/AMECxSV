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
from c8 import threshold_for_max_accuracy, threshold_metrics
from common import (
    BASELINE_DIR,
    BEST_MLP_SETTING,
    C5_COLS,
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
OUTPUT_CSV = BASELINE_DIR / "tidyvoice_c10_results.csv"
CHUNKSIZE = 250_000
MAX_TRAIN_ROWS_PER_CLASS = None
MAX_EVAL_ROWS_PER_CLASS = None
C10_3_COVERAGE_TARGET = 0.80
MODEL_SETTING = {key: value for key, value in BEST_MLP_SETTING.items() if key != "c_value"}
C_VALUE = BEST_MLP_SETTING["c_value"]
FEATURE_MODE = "c10_six_score_lang_reliability"


def score_columns() -> list[str]:
    return [f"score_{embedding}" for embedding in EMBEDDINGS]


def feature_matrix(frame) -> np.ndarray:
    scores = frame[score_columns()].to_numpy(dtype=np.float64, copy=False)
    language = frame["target"].to_numpy(dtype=np.float64, copy=False)[:, None]
    reliability = frame[C5_COLS].to_numpy(dtype=np.float64, copy=False)
    return np.concatenate([scores, language, reliability], axis=1)


def row_prefix(calib: str, feature_mode: str, model: str) -> dict[str, float | int | str]:
    return {
        "dataset": DATASET_NAME,
        "split": "test",
        "embedding": "six_embedding_fusion",
        "calib": calib,
        "feature_mode": feature_mode,
        "model": model,
        "hidden_dim": MODEL_SETTING["hidden_dim"],
        "dropout": MODEL_SETTING["dropout"],
        "learning_rate": MODEL_SETTING["learning_rate"],
        "epochs": MODEL_SETTING["epochs"],
        "batch_size": MODEL_SETTING["batch_size"],
        "C": C_VALUE,
        "feature_count": len(EMBEDDINGS) + 1 + len(C5_COLS),
    }


def binary_mlp_outputs(
    train_features: np.ndarray,
    validation_features: np.ndarray | None,
    test_features: np.ndarray,
    train_labels: np.ndarray,
    validation_labels: np.ndarray | None,
    device,
) -> tuple[object, np.ndarray, np.ndarray | None, np.ndarray]:
    experiment = Experiment("C10-2", FEATURE_MODE, C_VALUE, **MODEL_SETTING)
    model = make_rich_model(experiment, device, num_scores=len(EMBEDDINGS))
    model.fit(
        train_features,
        train_labels,
        balanced_sample_weights(train_labels),
        desc="train C10-2",
        validation_features=validation_features,
        validation_labels=validation_labels,
    )
    train_llrs = model.decision_function(train_features)
    validation_llrs = None if validation_features is None else model.decision_function(validation_features)
    test_llrs = model.decision_function(test_features)
    return model, train_llrs, validation_llrs, test_llrs


def binary_row(
    selection_llrs: np.ndarray,
    test_llrs: np.ndarray,
    selection_labels: np.ndarray,
    test_labels: np.ndarray,
    selection_split: str,
) -> dict[str, float | int | str]:
    threshold = threshold_for_max_accuracy(selection_llrs, selection_labels)
    row = row_prefix("C10-2", FEATURE_MODE, "rich_feature_mlp")
    row["selection_split"] = selection_split
    row.update(metric_block(test_llrs, test_llrs, test_labels))
    add_metric_ci(row, test_llrs, test_llrs, test_labels, desc="C10-2")
    selected_metrics = threshold_metrics(test_llrs, test_labels, threshold)
    row["final_accuracy"] = selected_metrics["accuracy"]
    row.update({f"threshold_{key}": value for key, value in selected_metrics.items()})
    return row


def reject_row(
    selection_llrs: np.ndarray,
    test_llrs: np.ndarray,
    selection_split: str,
    test_labels: np.ndarray,
) -> dict[str, float | int | str]:
    _, selection_confidence = confidence_from_llr(selection_llrs)
    posterior_target, confidence = confidence_from_llr(test_llrs)
    threshold = confidence_threshold(selection_confidence, C10_3_COVERAGE_TARGET)
    decisions = decisions_from_confidence(posterior_target, confidence, threshold)
    accepted = decisions != "reject"

    row = row_prefix("C10-3", f"{FEATURE_MODE}_conf_reject", "rich_feature_mlp")
    row["selection_split"] = selection_split
    row["coverage_target"] = C10_3_COVERAGE_TARGET
    row["confidence_threshold"] = threshold
    row["confidence_mean"] = float(np.mean(confidence))
    row["confidence_accepted_mean"] = float(np.mean(confidence[accepted])) if np.any(accepted) else math.nan
    row["confidence_rejected_mean"] = float(np.mean(confidence[~accepted])) if np.any(~accepted) else math.nan
    row["posterior_target_mean"] = float(np.mean(posterior_target))
    row.update(decision_metrics(test_labels, decisions))
    row["final_accuracy"] = row["accuracy"]
    row.update(bootstrap_decision_ci95(test_labels, decisions, desc="C10-3 decisions"))

    accepted_metrics = metric_block(test_llrs[accepted], test_llrs[accepted], test_labels[accepted])
    for key, value in accepted_metrics.items():
        row[f"accepted_{key}"] = value
    add_metric_ci(
        row,
        test_llrs[accepted],
        test_llrs[accepted],
        test_labels[accepted],
        desc="C10-3 accepted",
        prefix="accepted_",
    )
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

    rows: list[dict[str, float | int | str]] = []
    train_llrs: np.ndarray | None = None
    validation_llrs: np.ndarray | None = None
    test_llrs: np.ndarray | None = None

    tasks = ["c10_2_binary", "c10_3_conf_reject"]
    with tqdm(tasks, desc="C10 experiments", unit="task") as progress:
        for task in progress:
            if task == "c10_2_binary":
                _, train_llrs, validation_llrs, test_llrs = binary_mlp_outputs(
                    train_features,
                    validation_features,
                    test_features,
                    train_labels,
                    validation_labels,
                    device,
                )
                selection_llrs = train_llrs if validation_llrs is None else validation_llrs
                selection_labels = train_labels if validation_labels is None else validation_labels
                rows.append(binary_row(selection_llrs, test_llrs, selection_labels, test_labels, selection_split))
            elif task == "c10_3_conf_reject":
                if train_llrs is None or test_llrs is None:
                    raise RuntimeError("C10-2 must run before C10-3")
                selection_llrs = train_llrs if validation_llrs is None else validation_llrs
                rows.append(reject_row(selection_llrs, test_llrs, selection_split, test_labels))
            else:
                raise ValueError(f"Unknown task: {task}")

    write_rows(OUTPUT_CSV, rows)
    print(f"wrote {OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
