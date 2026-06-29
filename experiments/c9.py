from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import (
    BASELINE_DIR,
    EMBEDDINGS,
    add_metric_ci,
    balanced_sample_weights,
    bootstrap_decision_ci95,
    metric_block,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
C9_DATASET_DIR = PROJECT_ROOT / "dataset" / "c9"
C9_DATASET_NAME = os.environ.get("AMECXSV_C9_DATASET_NAME", "voxceleb1b")
C9_DATASET_PREFIX = os.environ.get("AMECXSV_C9_PREFIX", "voxceleb1b_c9")
FEATURES_PARQUET = Path(
    os.environ.get("AMECXSV_C9_FEATURES", C9_DATASET_DIR / f"{C9_DATASET_PREFIX}_features.parquet")
)
TRANSFORMS_JSON = Path(
    os.environ.get("AMECXSV_C9_TRANSFORMS", C9_DATASET_DIR / f"{C9_DATASET_PREFIX}_feature_transforms.json")
)
OUTPUT_CSV = BASELINE_DIR / "tidyvoice_c9_results.csv"
PREDICTION_CSV = BASELINE_DIR / "tidyvoice_c9_predictions.csv"
WEIGHTS_DIR = BASELINE_DIR / "weights" / "c9"

C9_3_COVERAGE_TARGET = 0.80
MODEL_C = 10.0
MODEL_MAX_ITER = 2000

FEATURE_SUFFIXES = [
    "score_sorted_1",
    "score_sorted_2",
    "score_sorted_3",
    "score_sorted_4",
    "score_sorted_5",
    "score_mean",
    "score_median",
    "score_min",
    "score_max",
    "score_std",
    "vote_frac",
    "vote_entropy",
    "post_mean",
    "post_std",
    "post_entropy_mean",
    "post_entropy_max",
]

RISK_WEIGHTS = {
    "model_uncertainty": 1.00,
    "score_std": 0.50,
    "post_std": 0.75,
    "vote_entropy": 0.75,
    "post_entropy_mean": 0.50,
    "post_entropy_max": 0.25,
    "vote_margin_risk": 0.75,
}


@dataclass(frozen=True)
class SklearnBinaryBackend:
    scaler: StandardScaler
    model: LogisticRegression
    feature_columns: list[str]

    def decision_function(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[self.feature_columns].to_numpy(dtype=np.float64, copy=False)
        return self.model.decision_function(self.scaler.transform(x)).astype(np.float64, copy=False)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values64 = values.astype(np.float64, copy=False)
    return 1.0 / (1.0 + np.exp(-values64))


def confidence_from_llr(llrs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    posterior_target = sigmoid(llrs)
    confidence = np.maximum(posterior_target, 1.0 - posterior_target)
    return posterior_target, confidence


def feature_columns_for_embedding(frame: pd.DataFrame, embedding: str) -> list[str]:
    columns = [f"{embedding}__{suffix}" for suffix in FEATURE_SUFFIXES]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing C9 feature columns for {embedding}: {missing}")
    return columns


def fit_backend(train: pd.DataFrame, embedding: str) -> SklearnBinaryBackend:
    feature_columns = feature_columns_for_embedding(train, embedding)
    x_train = train[feature_columns].to_numpy(dtype=np.float64, copy=False)
    labels = train["label"].to_numpy(dtype=np.int8, copy=False)
    weights = balanced_sample_weights(labels)

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)
    model = LogisticRegression(C=MODEL_C, max_iter=MODEL_MAX_ITER, solver="lbfgs")
    model.fit(x_scaled, labels, sample_weight=weights)
    return SklearnBinaryBackend(scaler=scaler, model=model, feature_columns=feature_columns)


def decision_metrics(labels: np.ndarray, decisions: np.ndarray) -> dict[str, float | int]:
    labels_bool = labels.astype(bool)
    target_decision = decisions == "target"
    nontarget_decision = decisions == "nontarget"
    reject_decision = decisions == "reject"
    accepted = ~reject_decision

    total_n = int(labels.shape[0])
    target_n = int(np.sum(labels_bool))
    nontarget_n = int(np.sum(~labels_bool))
    accepted_n = int(np.sum(accepted))
    correct_accepted = int(np.sum((target_decision & labels_bool) | (nontarget_decision & ~labels_bool)))

    coverage = accepted_n / total_n if total_n else math.nan
    covered_acc = correct_accepted / accepted_n if accepted_n else math.nan
    effective_acc = correct_accepted / total_n if total_n else math.nan
    accuracy = effective_acc if np.any(reject_decision) else covered_acc
    far = float(np.sum(target_decision & ~labels_bool) / nontarget_n) if nontarget_n else math.nan
    frr = (
        float(np.sum((nontarget_decision | reject_decision) & labels_bool) / target_n)
        if target_n
        else math.nan
    )

    return {
        "n": total_n,
        "target_n": target_n,
        "nontarget_n": nontarget_n,
        "accepted_n": accepted_n,
        "rejected_n": int(np.sum(reject_decision)),
        "coverage": coverage,
        "accuracy": accuracy,
        "effective_acc": effective_acc,
        "covered_acc": covered_acc,
        "FAR": far,
        "FRR": frr,
    }


def risk_components(frame: pd.DataFrame, embedding: str, confidence: np.ndarray) -> dict[str, np.ndarray]:
    vote_frac = frame[f"{embedding}__vote_frac"].to_numpy(dtype=np.float64, copy=False)
    return {
        "model_uncertainty": 1.0 - confidence,
        "score_std": frame[f"{embedding}__score_std"].to_numpy(dtype=np.float64, copy=False),
        "post_std": frame[f"{embedding}__post_std"].to_numpy(dtype=np.float64, copy=False),
        "vote_entropy": frame[f"{embedding}__vote_entropy"].to_numpy(dtype=np.float64, copy=False),
        "post_entropy_mean": frame[f"{embedding}__post_entropy_mean"].to_numpy(dtype=np.float64, copy=False),
        "post_entropy_max": frame[f"{embedding}__post_entropy_max"].to_numpy(dtype=np.float64, copy=False),
        "vote_margin_risk": 1.0 - np.abs(2.0 * vote_frac - 1.0),
    }


def fit_risk_stats(components: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for name, values in components.items():
        mean = float(np.mean(values))
        std = float(np.std(values))
        stats[name] = {"mean": mean, "std": std}
    return stats


def zscore(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    if not math.isfinite(std) or std <= 0.0:
        return np.zeros_like(values, dtype=np.float64)
    return (values - mean) / std


def reject_risk(components: dict[str, np.ndarray], stats: dict[str, dict[str, float]]) -> np.ndarray:
    risk = np.zeros_like(next(iter(components.values())), dtype=np.float64)
    for name, weight in RISK_WEIGHTS.items():
        values = components[name]
        risk += weight * zscore(values, stats[name]["mean"], stats[name]["std"])
    return risk


def decisions_from_scores(
    posterior_target: np.ndarray,
    risk: np.ndarray | None,
    risk_threshold: float | None,
) -> np.ndarray:
    decisions = np.where(posterior_target >= 0.5, "target", "nontarget").astype(object)
    if risk is not None and risk_threshold is not None:
        decisions[risk > risk_threshold] = "reject"
    return decisions


def summary_row(
    frame: pd.DataFrame,
    embedding: str,
    calib: str,
    feature_mode: str,
    llrs: np.ndarray,
    posterior_target: np.ndarray,
    confidence: np.ndarray,
    risk: np.ndarray | None,
    decisions: np.ndarray,
    risk_threshold: float | None,
    coverage_target: float,
    weight_path: Path,
) -> dict[str, float | int | str]:
    labels = frame["label"].to_numpy(dtype=np.int8, copy=False)
    accepted = decisions != "reject"
    binary_decisions = np.where(posterior_target >= 0.5, "target", "nontarget")
    rejected = ~accepted

    row: dict[str, float | int | str] = {
        "dataset": C9_DATASET_NAME,
        "split": "test",
        "embedding": embedding,
        "calib": calib,
        "feature_mode": feature_mode,
        "param_name": f"{feature_mode}|sklearn_logreg|C={MODEL_C:g}",
        "model": "sklearn_logistic_regression",
        "C": MODEL_C,
        "feature_count": len(feature_columns_for_embedding(frame, embedding)),
        "coverage_target": coverage_target,
        "risk_threshold": math.nan if risk_threshold is None else float(risk_threshold),
        "confidence_mean": float(np.mean(confidence)),
        "confidence_accepted_mean": float(np.mean(confidence[accepted])) if np.any(accepted) else math.nan,
        "confidence_rejected_mean": float(np.mean(confidence[rejected])) if np.any(rejected) else math.nan,
        "risk_mean": math.nan if risk is None else float(np.mean(risk)),
        "risk_accepted_mean": math.nan if risk is None or not np.any(accepted) else float(np.mean(risk[accepted])),
        "risk_rejected_mean": math.nan if risk is None or not np.any(rejected) else float(np.mean(risk[rejected])),
        "posterior_target_mean": float(np.mean(posterior_target)),
        "weight_path": str(weight_path),
    }
    row.update(decision_metrics(labels, decisions))
    row.update(bootstrap_decision_ci95(labels, decisions, desc=f"{embedding} {calib} decisions"))

    accepted_metrics = metric_block(llrs[accepted], llrs[accepted], labels[accepted])
    for key, value in accepted_metrics.items():
        row[f"accepted_{key}"] = value
    add_metric_ci(
        row,
        llrs[accepted],
        llrs[accepted],
        labels[accepted],
        desc=f"{embedding} {calib} accepted",
        prefix="accepted_",
    )

    if np.any(rejected):
        labels_bool = labels.astype(bool)
        rejected_binary_target = binary_decisions[rejected] == "target"
        row["rejected_error_rate"] = float(np.mean(rejected_binary_target != labels_bool[rejected]))
        row["accepted_error_rate"] = math.nan if not np.any(accepted) else 1.0 - float(row["covered_acc"])
    else:
        row["rejected_error_rate"] = math.nan
        row["accepted_error_rate"] = math.nan if not np.any(accepted) else 1.0 - float(row["covered_acc"])
    return row


def write_prediction_rows(
    writer: csv.DictWriter,
    frame: pd.DataFrame,
    embedding: str,
    calib: str,
    feature_mode: str,
    llrs: np.ndarray,
    posterior_target: np.ndarray,
    confidence: np.ndarray,
    risk: np.ndarray | None,
    decisions: np.ndarray,
) -> None:
    group_ids = frame["group_id"].to_numpy(dtype=np.uint64, copy=False)
    labels = frame["label"].to_numpy(dtype=np.uint8, copy=False)
    if risk is None:
        risk_values = np.full(llrs.shape, np.nan, dtype=np.float64)
    else:
        risk_values = risk
    for i in range(frame.shape[0]):
        writer.writerow(
            {
                "group_id": int(group_ids[i]),
                "embedding": embedding,
                "calib": calib,
                "feature_mode": feature_mode,
                "label": int(labels[i]),
                "llr": float(llrs[i]),
                "posterior_target": float(posterior_target[i]),
                "confidence": float(confidence[i]),
                "risk": float(risk_values[i]) if math.isfinite(float(risk_values[i])) else "",
                "decision": str(decisions[i]),
                "accepted": int(decisions[i] != "reject"),
            }
        )


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


def save_backend(
    path: Path,
    *,
    backend: SklearnBinaryBackend,
    embedding: str,
    calib: str,
    feature_mode: str,
    risk_stats: dict[str, dict[str, float]] | None,
    risk_threshold: float | None,
    coverage_target: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "embedding": embedding,
            "calib": calib,
            "feature_mode": feature_mode,
            "model": backend.model,
            "scaler": backend.scaler,
            "feature_columns": backend.feature_columns,
            "model_setting": {"C": MODEL_C, "max_iter": MODEL_MAX_ITER, "solver": "lbfgs"},
            "risk_weights": RISK_WEIGHTS if risk_stats is not None else None,
            "risk_stats": risk_stats,
            "risk_threshold": risk_threshold,
            "coverage_target": coverage_target,
        },
        path,
    )


def load_embeddings_from_transforms() -> list[str]:
    if not TRANSFORMS_JSON.exists():
        return EMBEDDINGS
    with TRANSFORMS_JSON.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    transforms = payload.get("transforms", {})
    discovered = [embedding for embedding in EMBEDDINGS if embedding in transforms]
    discovered.extend(sorted(set(transforms) - set(discovered)))
    return discovered


def main() -> None:
    if not FEATURES_PARQUET.exists():
        raise FileNotFoundError(FEATURES_PARQUET)
    frame = pd.read_parquet(FEATURES_PARQUET)
    train = frame[frame["split"] == 0].reset_index(drop=True)
    test = frame[frame["split"] == 1].reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("C9 features must contain split=0 calibration and split=1 test rows.")

    embeddings = load_embeddings_from_transforms()
    rows: list[dict[str, float | int | str]] = []
    prediction_fields = [
        "group_id",
        "embedding",
        "calib",
        "feature_mode",
        "label",
        "llr",
        "posterior_target",
        "confidence",
        "risk",
        "decision",
        "accepted",
    ]

    PREDICTION_CSV.parent.mkdir(parents=True, exist_ok=True)
    with PREDICTION_CSV.open("w", newline="", encoding="utf-8") as pred_handle:
        pred_writer = csv.DictWriter(pred_handle, fieldnames=prediction_fields)
        pred_writer.writeheader()

        for embedding in embeddings:
            print(f"training C9 backend embedding={embedding}", flush=True)
            backend = fit_backend(train, embedding)

            train_llrs = backend.decision_function(train)
            _, train_confidence = confidence_from_llr(train_llrs)
            train_components = risk_components(train, embedding, train_confidence)
            train_risk_stats = fit_risk_stats(train_components)
            train_risk = reject_risk(train_components, train_risk_stats)
            risk_threshold = float(np.quantile(train_risk, C9_3_COVERAGE_TARGET))

            test_llrs = backend.decision_function(test)
            posterior_target, confidence = confidence_from_llr(test_llrs)
            test_components = risk_components(test, embedding, confidence)
            test_risk = reject_risk(test_components, train_risk_stats)

            for calib, feature_mode, risk, threshold, coverage_target in (
                ("C9-2", "c9_2_multi_binary", None, None, 1.0),
                ("C9-3", "c9_3_multi_conf_disagreement_reject", test_risk, risk_threshold, C9_3_COVERAGE_TARGET),
            ):
                decisions = decisions_from_scores(posterior_target, risk, threshold)
                weight_path = WEIGHTS_DIR / calib.lower().replace("-", "_") / f"{embedding}.joblib"
                save_backend(
                    weight_path,
                    backend=backend,
                    embedding=embedding,
                    calib=calib,
                    feature_mode=feature_mode,
                    risk_stats=train_risk_stats if calib == "C9-3" else None,
                    risk_threshold=threshold,
                    coverage_target=coverage_target,
                )
                rows.append(
                    summary_row(
                        test,
                        embedding,
                        calib,
                        feature_mode,
                        test_llrs,
                        posterior_target,
                        confidence,
                        risk,
                        decisions,
                        threshold,
                        coverage_target,
                        weight_path,
                    )
                )
                write_prediction_rows(
                    pred_writer,
                    test,
                    embedding,
                    calib,
                    feature_mode,
                    test_llrs,
                    posterior_target,
                    confidence,
                    risk,
                    decisions,
                )

    write_rows(OUTPUT_CSV, rows)
    print(f"wrote {OUTPUT_CSV}", flush=True)
    print(f"wrote {PREDICTION_CSV}", flush=True)
    print(f"wrote {WEIGHTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
