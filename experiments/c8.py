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
from common import (
    BASELINE_DIR,
    BEST_MLP_SETTING,
    DATASET_NAME,
    EMBEDDINGS,
    Experiment,
    add_metric_ci,
    balanced_sample_weights,
    bootstrap_decision_ci95,
    classification_metrics,
    load_fixed_splits,
    make_rich_model,
    metric_block,
    require_cuda,
)


OUTPUT_CSV = BASELINE_DIR / "tidyvoice_c8_results.csv"
CHUNKSIZE = 250_000
MAX_TRAIN_ROWS_PER_CLASS = None
MAX_EVAL_ROWS_PER_CLASS = None
MODEL_SETTING = {key: value for key, value in BEST_MLP_SETTING.items() if key != "c_value"}
C_VALUE = BEST_MLP_SETTING["c_value"]
C8_3_COVERAGE_TARGET = 0.80
EPS = 1.0e-12


def score_columns() -> list[str]:
    return [f"score_{embedding}" for embedding in EMBEDDINGS]


def score_matrix(frame) -> np.ndarray:
    return frame[score_columns()].to_numpy(dtype=np.float64, copy=False)


def fit_score_stats(train_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(train_scores, axis=0, keepdims=True)
    std = np.std(train_scores, axis=0, keepdims=True)
    std = np.where(std <= 0.0, 1.0, std)
    return mean, std


def normalize_scores(scores: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (scores - mean) / std


def softmax_rows(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def score_entropy_features(z_scores: np.ndarray) -> np.ndarray:
    probs = np.clip(softmax_rows(z_scores), EPS, 1.0)
    entropy = -np.sum(probs * np.log(probs), axis=1, keepdims=True) / math.log(probs.shape[1])
    sorted_probs = np.sort(probs, axis=1)
    top_margin = (sorted_probs[:, -1] - sorted_probs[:, -2])[:, None]
    max_prob = sorted_probs[:, -1][:, None]
    return np.concatenate([entropy, top_margin, max_prob], axis=1)


def threshold_for_max_accuracy(scores: np.ndarray, labels: np.ndarray) -> float:
    labels_bool = labels.astype(bool)
    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels_bool[order]
    target_accepts = np.cumsum(sorted_labels, dtype=np.int64)
    nontarget_accepts = np.cumsum(~sorted_labels, dtype=np.int64)
    target_total = int(np.sum(labels_bool))
    nontarget_total = int(labels_bool.shape[0] - target_total)

    unique_ends = np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:])
    unique_ends = np.concatenate([unique_ends, np.asarray([sorted_scores.shape[0] - 1], dtype=np.int64)])
    correct = target_accepts[unique_ends] + (nontarget_total - nontarget_accepts[unique_ends])
    best = int(unique_ends[int(np.argmax(correct))])
    return float(sorted_scores[best])


def threshold_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float | int]:
    labels_bool = labels.astype(bool)
    pred = scores >= threshold
    target_n = int(np.sum(labels_bool))
    nontarget_n = int(labels_bool.shape[0] - target_n)
    row = classification_metrics(scores, labels, threshold=threshold)
    row["threshold"] = threshold
    row["FAR"] = float(np.sum(pred & ~labels_bool) / nontarget_n) if nontarget_n else math.nan
    row["FRR"] = float(np.sum(~pred & labels_bool) / target_n) if target_n else math.nan
    return row


def individual_weights(train_scores: np.ndarray, labels: np.ndarray) -> np.ndarray:
    eers = []
    for idx in range(train_scores.shape[1]):
        metrics = metric_block(train_scores[:, idx], train_scores[:, idx], labels)
        eers.append(float(metrics["eer_pct"]) / 100.0)
    inv = 1.0 / np.maximum(np.asarray(eers, dtype=np.float64), 1.0e-4)
    return inv / np.sum(inv)


def feature_matrix(mode: str, scores: np.ndarray, mean: np.ndarray, std: np.ndarray, weights: np.ndarray) -> np.ndarray:
    z_scores = normalize_scores(scores, mean, std)
    if mode == "c8_1_raw_scores":
        return scores

    z_mean = np.mean(z_scores, axis=1, keepdims=True)
    z_std = np.std(z_scores, axis=1, keepdims=True)
    z_max = np.max(z_scores, axis=1, keepdims=True)
    z_min = np.min(z_scores, axis=1, keepdims=True)
    z_range = z_max - z_min
    z_weighted = (z_scores @ weights)[:, None]
    summary = np.concatenate([z_scores, z_mean, z_std, z_max, z_min, z_range, z_weighted], axis=1)
    if mode == "c8_2_norm_fusion":
        return summary
    if mode == "c8_3_entropy_fusion":
        return np.concatenate([summary, score_entropy_features(z_scores)], axis=1)
    raise ValueError(f"Unknown C8 feature mode: {mode}")


def row_prefix(calib: str, mode: str, model: str) -> dict[str, float | int | str]:
    return {
        "dataset": DATASET_NAME,
        "split": "test",
        "embedding": "six_embedding_fusion",
        "calib": calib,
        "feature_mode": mode,
        "model": model,
        "hidden_dim": MODEL_SETTING["hidden_dim"],
        "dropout": MODEL_SETTING["dropout"],
        "learning_rate": MODEL_SETTING["learning_rate"],
        "epochs": MODEL_SETTING["epochs"],
        "batch_size": MODEL_SETTING["batch_size"],
        "C": C_VALUE,
    }


def raw_single_rows(
    selection_scores: np.ndarray,
    test_scores: np.ndarray,
    selection_labels: np.ndarray,
    test_labels: np.ndarray,
    selection_split: str,
) -> list[dict[str, float | int | str]]:
    rows = []
    for idx, embedding in enumerate(EMBEDDINGS):
        threshold = threshold_for_max_accuracy(selection_scores[:, idx], selection_labels)
        row = row_prefix("C8-RAW", "single_raw_score", "raw_threshold")
        row["embedding"] = embedding
        row["selection_split"] = selection_split
        row.update(metric_block(test_scores[:, idx], test_scores[:, idx], test_labels))
        add_metric_ci(row, test_scores[:, idx], test_scores[:, idx], test_labels, desc=f"C8-RAW {embedding}")
        row.update({f"threshold_{k}": v for k, v in threshold_metrics(test_scores[:, idx], test_labels, threshold).items()})
        rows.append(row)
    return rows


def hand_fusion_rows(
    selection_scores: np.ndarray,
    test_scores: np.ndarray,
    selection_labels: np.ndarray,
    test_labels: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    weights: np.ndarray,
    selection_split: str,
) -> list[dict[str, float | int | str]]:
    z_selection = normalize_scores(selection_scores, mean, std)
    z_test = normalize_scores(test_scores, mean, std)
    fused = {
        "raw_mean": (np.mean(selection_scores, axis=1), np.mean(test_scores, axis=1)),
        "z_mean": (np.mean(z_selection, axis=1), np.mean(z_test, axis=1)),
        "z_weighted": (z_selection @ weights, z_test @ weights),
        "z_max": (np.max(z_selection, axis=1), np.max(z_test, axis=1)),
    }
    rows = []
    for mode, (selection_values, test_values) in fused.items():
        threshold = threshold_for_max_accuracy(selection_values, selection_labels)
        row = row_prefix("C8-FUSION", mode, "score_fusion_threshold")
        row["selection_split"] = selection_split
        row.update(metric_block(test_values, test_values, test_labels))
        add_metric_ci(row, test_values, test_values, test_labels, desc=f"C8-FUSION {mode}")
        row.update({f"threshold_{k}": v for k, v in threshold_metrics(test_values, test_labels, threshold).items()})
        if mode == "z_weighted":
            for embedding, weight in zip(EMBEDDINGS, weights):
                row[f"weight_{embedding}"] = float(weight)
        rows.append(row)
    return rows


def binary_and_reject_rows(
    mode: str,
    train_features: np.ndarray,
    selection_features: np.ndarray,
    test_features: np.ndarray,
    train_labels: np.ndarray,
    selection_labels: np.ndarray,
    test_labels: np.ndarray,
    selection_split: str,
    device,
) -> list[dict[str, float | int | str]]:
    experiment = Experiment("C8-2", mode, C_VALUE, **MODEL_SETTING)
    model = make_rich_model(experiment, device, num_scores=len(EMBEDDINGS))
    use_validation = selection_features is not train_features
    model.fit(
        train_features,
        train_labels,
        balanced_sample_weights(train_labels),
        desc=f"train {mode}",
        validation_features=selection_features if use_validation else None,
        validation_labels=selection_labels if use_validation else None,
    )
    model_name = "rich_feature_mlp"
    train_llrs = model.decision_function(train_features)
    selection_llrs = train_llrs if selection_features is train_features else model.decision_function(selection_features)
    llrs = model.decision_function(test_features)
    threshold = threshold_for_max_accuracy(selection_llrs, selection_labels)

    binary = row_prefix("C8-2", mode, model_name)
    binary["selection_split"] = selection_split
    binary.update(metric_block(llrs, llrs, test_labels))
    add_metric_ci(binary, llrs, llrs, test_labels, desc=f"C8-2 {mode}")
    binary.update({f"threshold_{k}": v for k, v in threshold_metrics(llrs, test_labels, threshold).items()})

    _, selection_confidence = confidence_from_llr(selection_llrs)
    posterior_target, confidence = confidence_from_llr(llrs)
    reject_threshold = confidence_threshold(selection_confidence, C8_3_COVERAGE_TARGET)
    decisions = decisions_from_confidence(posterior_target, confidence, reject_threshold)
    accepted = decisions != "reject"

    reject = row_prefix("C8-3", f"{mode}_conf_reject", model_name)
    reject["selection_split"] = selection_split
    reject["coverage_target"] = C8_3_COVERAGE_TARGET
    reject["confidence_threshold"] = reject_threshold
    reject["confidence_mean"] = float(np.mean(confidence))
    reject["confidence_accepted_mean"] = float(np.mean(confidence[accepted])) if np.any(accepted) else math.nan
    reject["confidence_rejected_mean"] = float(np.mean(confidence[~accepted])) if np.any(~accepted) else math.nan
    reject["posterior_target_mean"] = float(np.mean(posterior_target))
    reject.update(decision_metrics(test_labels, decisions))
    reject.update(bootstrap_decision_ci95(test_labels, decisions, desc=f"C8-3 {mode} decisions"))
    accepted_metrics = metric_block(llrs[accepted], llrs[accepted], test_labels[accepted])
    for key, value in accepted_metrics.items():
        reject[f"accepted_{key}"] = value
    add_metric_ci(
        reject,
        llrs[accepted],
        llrs[accepted],
        test_labels[accepted],
        desc=f"C8-3 {mode} accepted",
        prefix="accepted_",
    )
    return [binary, reject]


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
    selection_frame = train if validation is None else validation
    selection_scores = train_scores if validation is None else score_matrix(validation)
    test_scores = score_matrix(test)
    train_labels = train["label"].to_numpy(dtype=np.int8, copy=False)
    selection_labels = train_labels if validation is None else validation["label"].to_numpy(dtype=np.int8, copy=False)
    test_labels = test["label"].to_numpy(dtype=np.int8, copy=False)
    selection_split = "train" if validation is None else "validation"
    mean, std = fit_score_stats(train_scores)
    weights = individual_weights(train_scores, train_labels)

    rows: list[dict[str, float | int | str]] = []
    tasks = [
        "single_raw_scores",
        "hand_fusions",
        "c8_1_raw_scores",
        "c8_2_norm_fusion",
        "c8_3_entropy_fusion",
    ]
    with tqdm(tasks, desc="C8 experiments", unit="task") as progress:
        for task in progress:
            if task == "single_raw_scores":
                rows.extend(raw_single_rows(selection_scores, test_scores, selection_labels, test_labels, selection_split))
            elif task == "hand_fusions":
                rows.extend(hand_fusion_rows(selection_scores, test_scores, selection_labels, test_labels, mean, std, weights, selection_split))
            else:
                train_features = feature_matrix(task, train_scores, mean, std, weights)
                selection_features = train_features if validation is None else feature_matrix(task, selection_scores, mean, std, weights)
                test_features = feature_matrix(task, test_scores, mean, std, weights)
                rows.extend(
                    binary_and_reject_rows(
                        task,
                        train_features,
                        selection_features,
                        test_features,
                        train_labels,
                        selection_labels,
                        test_labels,
                        selection_split,
                        device,
                    )
                )

    write_rows(OUTPUT_CSV, rows)
    print(f"wrote {OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
