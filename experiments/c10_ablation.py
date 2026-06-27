from __future__ import annotations

import csv
import math
import os
from concurrent.futures import ThreadPoolExecutor
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
    PRIORS,
    balanced_sample_weights,
    load_fixed_splits,
    make_rich_model,
    metric_block,
    normalized_dcf,
    require_cuda,
)
OUTPUT_CSV = BASELINE_DIR / "tidyvoice_c10_ablation_results.csv"
CHUNKSIZE = 250_000
MAX_TRAIN_ROWS_PER_CLASS = None
MAX_EVAL_ROWS_PER_CLASS = None
COVERAGE_TARGET = 0.80
MODEL_SETTING = {key: value for key, value in BEST_MLP_SETTING.items() if key != "c_value"}
C_VALUE = BEST_MLP_SETTING["c_value"]
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20260608
BOOTSTRAP_WORKERS = min(8, os.cpu_count() or 1)

FULL_CI_KEYS = (
    "eer_pct",
    "Cllr",
    "minDCF_p001",
    "actDCF_p001",
    "minDCF_p01",
    "actDCF_p01",
    "accuracy",
    "precision",
    "recall",
    "f1",
)
REJECT_CI_KEYS = (
    "accuracy",
    "effective_acc",
    "covered_acc",
    "coverage",
    "FAR",
    "FRR",
)

ABLATION_MODES = [
    "c10_full",
    "c10_score_only",
    "c10_no_language",
    "c10_no_reliability",
    "c10_language_only",
    "c10_reliability_only",
]
RICH_FEATURE_MODES = set(ABLATION_MODES)


def num_scores(mode: str) -> int:
    mode = mode.replace("_conf_reject", "")
    if mode == "c10_language_only":
        return 1
    if mode == "c10_reliability_only":
        return len(C5_COLS)
    return len(EMBEDDINGS)


def score_columns() -> list[str]:
    return [f"score_{embedding}" for embedding in EMBEDDINGS]


def feature_matrix(frame, mode: str) -> np.ndarray:
    scores = frame[score_columns()].to_numpy(dtype=np.float64, copy=False)
    language = frame["target"].to_numpy(dtype=np.float64, copy=False)[:, None]
    reliability = frame[C5_COLS].to_numpy(dtype=np.float64, copy=False)

    if mode == "c10_full":
        return np.concatenate([scores, language, reliability], axis=1)
    if mode == "c10_score_only":
        return scores
    if mode == "c10_no_language":
        return np.concatenate([scores, reliability], axis=1)
    if mode == "c10_no_reliability":
        return np.concatenate([scores, language], axis=1)
    if mode == "c10_language_only":
        return language
    if mode == "c10_reliability_only":
        return reliability
    raise ValueError(f"Unknown C10 ablation mode: {mode}")


def feature_count(mode: str) -> int:
    mode = mode.replace("_conf_reject", "")
    if mode == "c10_full":
        return len(EMBEDDINGS) + 1 + len(C5_COLS)
    if mode == "c10_score_only":
        return len(EMBEDDINGS)
    if mode == "c10_no_language":
        return len(EMBEDDINGS) + len(C5_COLS)
    if mode == "c10_no_reliability":
        return len(EMBEDDINGS) + 1
    if mode == "c10_language_only":
        return 1
    if mode == "c10_reliability_only":
        return len(C5_COLS)
    raise ValueError(f"Unknown C10 ablation mode: {mode}")


def row_prefix(mode: str) -> dict[str, float | int | str]:
    base_mode = mode.replace("_conf_reject", "")
    return {
        "dataset": DATASET_NAME,
        "split": "test",
        "embedding": "six_embedding_fusion",
        "calib": "C10_ABL",
        "feature_mode": mode,
        "model": "rich_feature_mlp" if base_mode in RICH_FEATURE_MODES else "cuda_mlp_binary",
        "hidden_dim": MODEL_SETTING["hidden_dim"],
        "dropout": MODEL_SETTING["dropout"],
        "learning_rate": MODEL_SETTING["learning_rate"],
        "epochs": MODEL_SETTING["epochs"],
        "batch_size": MODEL_SETTING["batch_size"],
        "C": C_VALUE,
        "feature_count": feature_count(mode),
    }


def eer_from_rates(thresholds: np.ndarray, pmiss: np.ndarray, pfa: np.ndarray) -> tuple[float, float]:
    if thresholds.size == 0:
        return math.nan, math.nan
    diff = pfa - pmiss
    crossing = np.flatnonzero(diff >= 0.0)
    if crossing.size == 0:
        best = int(np.argmin(np.abs(diff)))
        return float((pmiss[best] + pfa[best]) / 2.0), float(thresholds[best])
    right = int(crossing[0])
    if right == 0:
        return float((pmiss[0] + pfa[0]) / 2.0), float(thresholds[0])
    left = right - 1
    denom = diff[left] - diff[right]
    alpha = 0.0 if denom == 0.0 else float(np.clip(diff[left] / denom, 0.0, 1.0))
    eer_value = pfa[left] + alpha * (pfa[right] - pfa[left])
    threshold = thresholds[right]
    if math.isfinite(thresholds[left]) and math.isfinite(thresholds[right]):
        threshold = thresholds[left] + alpha * (thresholds[right] - thresholds[left])
    return float(eer_value), float(threshold)


def sorted_score_context(scores: np.ndarray, llrs: np.ndarray, labels: np.ndarray) -> dict[str, np.ndarray]:
    labels_bool = labels.astype(bool)
    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_scores = scores[order]
    unique_ends = np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:])
    unique_ends = np.concatenate([unique_ends, np.asarray([sorted_scores.shape[0] - 1], dtype=np.int64)])
    return {
        "scores": scores,
        "llrs": llrs,
        "labels_bool": labels_bool,
        "order": order,
        "sorted_labels_bool": labels_bool[order],
        "unique_ends": unique_ends,
        "thresholds": np.concatenate([np.asarray([np.inf]), sorted_scores[unique_ends]]),
        "target_loss": np.logaddexp(0.0, -llrs),
        "nontarget_loss": np.logaddexp(0.0, llrs),
    }


def weighted_classification_metrics(
    llrs: np.ndarray,
    labels_bool: np.ndarray,
    counts: np.ndarray,
    threshold: float = 0.0,
) -> dict[str, float]:
    pred = llrs >= threshold
    tp = int(np.sum(counts[pred & labels_bool]))
    fp = int(np.sum(counts[pred & ~labels_bool]))
    tn = int(np.sum(counts[~pred & ~labels_bool]))
    fn = int(np.sum(counts[~pred & labels_bool]))
    total = int(np.sum(counts))
    precision = math.nan if tp + fp == 0 else tp / (tp + fp)
    recall = math.nan if tp + fn == 0 else tp / (tp + fn)
    f1 = math.nan if not math.isfinite(precision) or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "accuracy": math.nan if total == 0 else (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def weighted_rates(
    context: dict[str, np.ndarray],
    counts: np.ndarray,
    target_n: int,
    nontarget_n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = context["order"]
    sorted_labels_bool = context["sorted_labels_bool"].astype(bool, copy=False)
    sorted_counts = counts[order]
    target_cumsum = np.cumsum(sorted_counts * sorted_labels_bool, dtype=np.int64)
    nontarget_cumsum = np.cumsum(sorted_counts * ~sorted_labels_bool, dtype=np.int64)
    unique_ends = context["unique_ends"]
    target_accepts = target_cumsum[unique_ends]
    nontarget_accepts = nontarget_cumsum[unique_ends]
    pmiss = (target_n - target_accepts).astype(np.float64) / target_n
    pfa = nontarget_accepts.astype(np.float64) / nontarget_n
    return (
        context["thresholds"],
        np.concatenate([np.asarray([1.0]), pmiss]),
        np.concatenate([np.asarray([0.0]), pfa]),
    )


def weighted_cllr(context: dict[str, np.ndarray], counts: np.ndarray, target_n: int, nontarget_n: int) -> float:
    labels_bool = context["labels_bool"].astype(bool, copy=False)
    target_loss = np.sum(counts[labels_bool] * context["target_loss"][labels_bool]) / target_n
    nontarget_loss = np.sum(counts[~labels_bool] * context["nontarget_loss"][~labels_bool]) / nontarget_n
    return float(0.5 * (target_loss + nontarget_loss) / math.log(2.0))


def weighted_act_dcf(
    llrs: np.ndarray,
    labels_bool: np.ndarray,
    counts: np.ndarray,
    target_n: int,
    nontarget_n: int,
    prior: float,
) -> float:
    threshold = math.log((1.0 - prior) / prior)
    accept = llrs >= threshold
    pmiss = np.sum(counts[labels_bool & ~accept]) / target_n
    pfa = np.sum(counts[~labels_bool & accept]) / nontarget_n
    return float(normalized_dcf(np.asarray([pmiss]), np.asarray([pfa]), prior)[0])


def weighted_verification_metric_block(context: dict[str, np.ndarray], counts: np.ndarray) -> dict[str, float | int]:
    labels_bool = context["labels_bool"].astype(bool, copy=False)
    target_n = int(np.sum(counts[labels_bool]))
    total_n = int(np.sum(counts))
    nontarget_n = total_n - target_n
    row: dict[str, float | int] = {
        "n": total_n,
        "target_n": target_n,
        "nontarget_n": nontarget_n,
    }
    if total_n == 0 or target_n == 0 or nontarget_n == 0:
        row.update({"eer_pct": math.nan, "eer_threshold": math.nan, "Cllr": math.nan})
        for suffix in PRIORS:
            row[f"minDCF_{suffix}"] = math.nan
            row[f"actDCF_{suffix}"] = math.nan
        row.update(weighted_classification_metrics(context["llrs"], labels_bool, counts))
        return row

    thresholds, pmiss, pfa = weighted_rates(context, counts, target_n, nontarget_n)
    eer_value, eer_threshold = eer_from_rates(thresholds, pmiss, pfa)
    row.update({"eer_pct": 100.0 * eer_value, "eer_threshold": eer_threshold, "Cllr": weighted_cllr(context, counts, target_n, nontarget_n)})
    for suffix, prior in PRIORS.items():
        row[f"minDCF_{suffix}"] = float(np.min(normalized_dcf(pmiss, pfa, prior)))
        row[f"actDCF_{suffix}"] = weighted_act_dcf(context["llrs"], labels_bool, counts, target_n, nontarget_n, prior)
    row.update(weighted_classification_metrics(context["llrs"], labels_bool, counts))
    return row


def weighted_decision_metrics(labels: np.ndarray, decisions: np.ndarray, counts: np.ndarray) -> dict[str, float | int]:
    labels_bool = labels.astype(bool)
    target_decision = decisions == "target"
    nontarget_decision = decisions == "nontarget"
    reject_decision = decisions == "reject"
    accepted = ~reject_decision

    total_n = int(np.sum(counts))
    target_n = int(np.sum(counts[labels_bool]))
    nontarget_n = total_n - target_n
    accepted_n = int(np.sum(counts[accepted]))
    rejected_n = int(np.sum(counts[reject_decision]))
    correct_accepted = int(np.sum(counts[(target_decision & labels_bool) | (nontarget_decision & ~labels_bool)]))
    correct_all = correct_accepted

    coverage = accepted_n / total_n if total_n else math.nan
    covered_acc = correct_accepted / accepted_n if accepted_n else math.nan
    effective_acc = correct_accepted / total_n if total_n else math.nan
    accuracy = correct_all / total_n if total_n else math.nan
    far = float(np.sum(counts[target_decision & ~labels_bool]) / nontarget_n) if nontarget_n else math.nan
    frr = (
        float(np.sum(counts[(nontarget_decision | reject_decision) & labels_bool]) / target_n)
        if target_n
        else math.nan
    )

    return {
        "n": total_n,
        "target_n": target_n,
        "nontarget_n": nontarget_n,
        "accepted_n": accepted_n,
        "rejected_n": rejected_n,
        "coverage": coverage,
        "accuracy": accuracy,
        "effective_acc": effective_acc,
        "covered_acc": covered_acc,
        "FAR": far,
        "FRR": frr,
    }


def percentile_ci95(values: list[float]) -> float:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return math.nan
    low, high = np.percentile(finite, [2.5, 97.5])
    return float((high - low) / 2.0)


def bootstrap_metric_ci95(
    scores: np.ndarray,
    llrs: np.ndarray,
    labels: np.ndarray,
    keys: tuple[str, ...],
    rng: np.random.Generator,
    desc: str,
) -> dict[str, float]:
    if BOOTSTRAP_SAMPLES <= 0 or labels.shape[0] == 0:
        return {f"{key}_ci95": math.nan for key in keys}

    samples: dict[str, list[float]] = {key: [] for key in keys}
    n = int(labels.shape[0])
    context = sorted_score_context(scores, llrs, labels)

    seeds = rng.integers(0, np.iinfo(np.uint32).max, size=BOOTSTRAP_SAMPLES, dtype=np.uint32)

    def one_sample(seed: np.uint32) -> dict[str, float]:
        local_rng = np.random.default_rng(int(seed))
        idx = local_rng.integers(0, n, size=n, dtype=np.int64)
        counts = np.bincount(idx, minlength=n)
        metrics = weighted_verification_metric_block(context, counts)
        return {key: float(metrics[key]) for key in keys}

    with ThreadPoolExecutor(max_workers=BOOTSTRAP_WORKERS) as pool:
        iterator = pool.map(one_sample, seeds)
        for metrics in tqdm(iterator, total=BOOTSTRAP_SAMPLES, desc=f"bootstrap {desc}", unit="sample", leave=False):
            for key in keys:
                samples[key].append(metrics[key])
    return {f"{key}_ci95": percentile_ci95(values) for key, values in samples.items()}


def bootstrap_decision_ci95(
    labels: np.ndarray,
    decisions: np.ndarray,
    keys: tuple[str, ...],
    rng: np.random.Generator,
    desc: str,
) -> dict[str, float]:
    if BOOTSTRAP_SAMPLES <= 0 or labels.shape[0] == 0:
        return {f"{key}_ci95": math.nan for key in keys}

    samples: dict[str, list[float]] = {key: [] for key in keys}
    n = int(labels.shape[0])

    seeds = rng.integers(0, np.iinfo(np.uint32).max, size=BOOTSTRAP_SAMPLES, dtype=np.uint32)

    def one_sample(seed: np.uint32) -> dict[str, float]:
        local_rng = np.random.default_rng(int(seed))
        idx = local_rng.integers(0, n, size=n, dtype=np.int64)
        counts = np.bincount(idx, minlength=n)
        metrics = weighted_decision_metrics(labels, decisions, counts)
        return {key: float(metrics[key]) for key in keys}

    with ThreadPoolExecutor(max_workers=BOOTSTRAP_WORKERS) as pool:
        iterator = pool.map(one_sample, seeds)
        for metrics in tqdm(iterator, total=BOOTSTRAP_SAMPLES, desc=f"bootstrap {desc}", unit="sample", leave=False):
            for key in keys:
                samples[key].append(metrics[key])
    return {f"{key}_ci95": percentile_ci95(values) for key, values in samples.items()}


def fit_binary_llrs(
    mode: str,
    train_features: np.ndarray,
    validation_features: np.ndarray | None,
    test_features: np.ndarray,
    train_labels: np.ndarray,
    validation_labels: np.ndarray | None,
    device,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    experiment = Experiment("C10_ABL", mode, C_VALUE, **MODEL_SETTING)
    model = make_rich_model(experiment, device, num_scores=num_scores(mode))
    model.fit(
        train_features,
        train_labels,
        balanced_sample_weights(train_labels),
        desc=f"train {mode}",
        validation_features=validation_features,
        validation_labels=validation_labels,
    )
    validation_llrs = None if validation_features is None else model.decision_function(validation_features)
    train_llrs = model.decision_function(train_features)
    test_llrs = model.decision_function(test_features)
    return train_llrs, validation_llrs, test_llrs


def ablation_row(
    mode: str,
    selection_llrs: np.ndarray,
    test_llrs: np.ndarray,
    selection_labels: np.ndarray,
    test_labels: np.ndarray,
    selection_split: str,
    rng: np.random.Generator,
) -> dict[str, float | int | str]:
    threshold = threshold_for_max_accuracy(selection_llrs, selection_labels)
    row = row_prefix(mode)
    row["selection_split"] = selection_split
    row.update(metric_block(test_llrs, test_llrs, test_labels))
    if BOOTSTRAP_SAMPLES > 0:
        row.update(bootstrap_metric_ci95(test_llrs, test_llrs, test_labels, FULL_CI_KEYS, rng, mode))
        row["ci95"] = row["accuracy_ci95"]
    selected_metrics = threshold_metrics(test_llrs, test_labels, threshold)
    row["final_accuracy"] = selected_metrics["accuracy"]
    row.update({f"threshold_{key}": value for key, value in selected_metrics.items()})
    return row


def reject_row(
    mode: str,
    selection_llrs: np.ndarray,
    test_llrs: np.ndarray,
    selection_split: str,
    test_labels: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float | int | str]:
    _, selection_confidence = confidence_from_llr(selection_llrs)
    posterior_target, confidence = confidence_from_llr(test_llrs)
    threshold = confidence_threshold(selection_confidence, COVERAGE_TARGET)
    decisions = decisions_from_confidence(posterior_target, confidence, threshold)
    accepted = decisions != "reject"

    row = row_prefix(f"{mode}_conf_reject")
    row["selection_split"] = selection_split
    row["coverage_target"] = COVERAGE_TARGET
    row["confidence_threshold"] = threshold
    row["confidence_mean"] = float(np.mean(confidence))
    row["confidence_accepted_mean"] = float(np.mean(confidence[accepted])) if np.any(accepted) else math.nan
    row["confidence_rejected_mean"] = float(np.mean(confidence[~accepted])) if np.any(~accepted) else math.nan
    row["posterior_target_mean"] = float(np.mean(posterior_target))
    row.update(decision_metrics(test_labels, decisions))
    row["final_accuracy"] = row["accuracy"]
    if BOOTSTRAP_SAMPLES > 0:
        row.update(bootstrap_decision_ci95(test_labels, decisions, REJECT_CI_KEYS, rng, f"{mode}_reject"))
        row["ci95"] = row["covered_acc_ci95"]

    accepted_metrics = metric_block(test_llrs[accepted], test_llrs[accepted], test_labels[accepted])
    for key, value in accepted_metrics.items():
        row[f"accepted_{key}"] = value
    if BOOTSTRAP_SAMPLES > 0:
        accepted_ci = bootstrap_metric_ci95(
            test_llrs[accepted],
            test_llrs[accepted],
            test_labels[accepted],
            FULL_CI_KEYS,
            rng,
            f"{mode}_accepted",
        )
        for key, value in accepted_ci.items():
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
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    rows: list[dict[str, float | int | str]] = []
    with tqdm(ABLATION_MODES, desc="C10 ablation", unit="mode") as progress:
        for mode in progress:
            train_features = feature_matrix(train, mode)
            validation_features = None if validation is None else feature_matrix(validation, mode)
            test_features = feature_matrix(test, mode)
            train_llrs, validation_llrs, test_llrs = fit_binary_llrs(
                mode,
                train_features,
                validation_features,
                test_features,
                train_labels,
                validation_labels,
                device,
            )
            selection_llrs = train_llrs if validation_llrs is None else validation_llrs
            selection_labels = train_labels if validation_labels is None else validation_labels
            rows.append(ablation_row(mode, selection_llrs, test_llrs, selection_labels, test_labels, selection_split, rng))
            rows.append(reject_row(mode, selection_llrs, test_llrs, selection_split, test_labels, rng))

    write_rows(OUTPUT_CSV, rows)
    print(f"wrote {OUTPUT_CSV}", flush=True)


if __name__ == "__main__":
    main()
