from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from tqdm import tqdm

from common import (
    BASELINE_DIR,
    DATASET_NAME,
    EMBEDDINGS,
    add_metric_ci,
    balanced_sample_weights,
    load_fixed_splits,
    metric_block,
    write_rows,
)


CHUNKSIZE = 250_000
MAX_TRAIN_ROWS_PER_CLASS = None
MAX_EVAL_ROWS_PER_CLASS = None
PAV_EPS = 1.0e-6


@dataclass(frozen=True)
class LogisticCalibrator:
    slope: float
    intercept: float
    score_mean: float
    score_std: float
    optimizer_success: bool
    optimizer_message: str
    loss: float

    def predict_llr(self, scores: np.ndarray) -> np.ndarray:
        return self.slope * scores + self.intercept


@dataclass(frozen=True)
class PavCalibrator:
    x_thresholds: np.ndarray
    y_thresholds: np.ndarray
    eps: float

    def predict_llr(self, scores: np.ndarray) -> np.ndarray:
        probabilities = np.interp(
            scores,
            self.x_thresholds,
            self.y_thresholds,
            left=self.y_thresholds[0],
            right=self.y_thresholds[-1],
        )
        probabilities = np.clip(probabilities, self.eps, 1.0 - self.eps)
        return np.log(probabilities / (1.0 - probabilities))


def score_column(embedding: str) -> str:
    return f"score_{embedding}"


def train_logistic(scores: np.ndarray, labels: np.ndarray) -> LogisticCalibrator:
    weights = balanced_sample_weights(labels)
    score_mean = float(np.mean(scores))
    score_std = float(np.std(scores))
    if not math.isfinite(score_std) or score_std <= 0.0:
        score_std = 1.0
    normalized_scores = (scores - score_mean) / score_std
    labels_float = labels.astype(np.float64)

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        llrs = params[0] * normalized_scores + params[1]
        loss_terms = np.logaddexp(0.0, llrs) - labels_float * llrs
        loss = float(np.sum(weights * loss_terms))
        posterior = expit(llrs)
        diff = weights * (posterior - labels_float)
        gradient = np.asarray(
            [
                np.sum(diff * normalized_scores),
                np.sum(diff),
            ],
            dtype=np.float64,
        )
        return loss, gradient

    result = minimize(
        fun=lambda params: objective(params)[0],
        x0=np.asarray([1.0, 0.0], dtype=np.float64),
        jac=lambda params: objective(params)[1],
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1.0e-12, "gtol": 1.0e-8},
    )
    normalized_slope = float(result.x[0])
    normalized_intercept = float(result.x[1])
    slope = normalized_slope / score_std
    intercept = normalized_intercept - normalized_slope * score_mean / score_std
    return LogisticCalibrator(
        slope=slope,
        intercept=intercept,
        score_mean=score_mean,
        score_std=score_std,
        optimizer_success=bool(result.success),
        optimizer_message=str(result.message),
        loss=float(result.fun),
    )


def train_pav(scores: np.ndarray, labels: np.ndarray, eps: float = PAV_EPS) -> PavCalibrator:
    weights = balanced_sample_weights(labels)
    isotonic = IsotonicRegression(y_min=eps, y_max=1.0 - eps, out_of_bounds="clip")
    isotonic.fit(scores, labels.astype(np.float64), sample_weight=weights)
    x_thresholds = np.asarray(isotonic.X_thresholds_, dtype=np.float64)
    y_thresholds = np.asarray(isotonic.y_thresholds_, dtype=np.float64)
    y_thresholds = np.clip(y_thresholds, eps, 1.0 - eps)
    return PavCalibrator(x_thresholds=x_thresholds, y_thresholds=y_thresholds, eps=eps)


def row_prefix(embedding: str, calib: str, feature_mode: str, model: str) -> dict[str, float | int | str]:
    return {
        "dataset": DATASET_NAME,
        "split": "test",
        "embedding": embedding,
        "calib": calib,
        "feature_mode": feature_mode,
        "param_name": feature_mode,
        "C": math.nan,
        "model": model,
        "hidden_dim": math.nan,
        "dropout": math.nan,
        "learning_rate": math.nan,
        "epochs": math.nan,
        "batch_size": math.nan,
        "coverage": 1.0,
    }


def evaluate_calibration(
    frame,
    embedding: str,
    calib: str,
    feature_mode: str,
    model: str,
    decision_scores: np.ndarray,
    llrs: np.ndarray,
) -> dict[str, float | int | str]:
    labels = frame["label"].to_numpy(dtype=np.int8, copy=False)
    row = row_prefix(embedding, calib, feature_mode, model)
    row.update(metric_block(decision_scores, llrs, labels))
    add_metric_ci(row, decision_scores, llrs, labels, desc=f"{embedding} {calib} {feature_mode}")

    same = frame["target"].to_numpy(dtype=bool, copy=False)
    for prefix, subset in (("same_language", same), ("cross_language", ~same)):
        subset_metrics = metric_block(decision_scores[subset], llrs[subset], labels[subset])
        for key, value in subset_metrics.items():
            row[f"{prefix}_{key}"] = value
    return row


def run_c0(output_csv=BASELINE_DIR / "tidyvoice_c0_results.csv") -> None:
    _, test = load_fixed_splits(
        EMBEDDINGS,
        CHUNKSIZE,
        MAX_TRAIN_ROWS_PER_CLASS,
        MAX_EVAL_ROWS_PER_CLASS,
    )
    rows: list[dict[str, float | int | str]] = []
    for embedding in tqdm(EMBEDDINGS, desc="C0 raw score", unit="embedding"):
        scores = test[score_column(embedding)].to_numpy(dtype=np.float64, copy=False)
        rows.append(
            evaluate_calibration(
                test,
                embedding,
                "C0",
                "c0_raw_score",
                "raw_score",
                decision_scores=scores,
                llrs=scores,
            )
        )
    write_rows(output_csv, rows)
    print(f"wrote {output_csv}", flush=True)


def run_c1(output_csv=BASELINE_DIR / "tidyvoice_c1_results.csv") -> None:
    train, test = load_fixed_splits(
        EMBEDDINGS,
        CHUNKSIZE,
        MAX_TRAIN_ROWS_PER_CLASS,
        MAX_EVAL_ROWS_PER_CLASS,
    )
    train_labels = train["label"].to_numpy(dtype=np.int8, copy=False)
    rows: list[dict[str, float | int | str]] = []
    for embedding in tqdm(EMBEDDINGS, desc="C1 logistic", unit="embedding"):
        train_scores = train[score_column(embedding)].to_numpy(dtype=np.float64, copy=False)
        test_scores = test[score_column(embedding)].to_numpy(dtype=np.float64, copy=False)
        calibrator = train_logistic(train_scores, train_labels)
        test_llrs = calibrator.predict_llr(test_scores)
        rows.append(
            evaluate_calibration(
                test,
                embedding,
                "C1",
                "c1_logistic",
                "logistic_calibration",
                decision_scores=test_llrs,
                llrs=test_llrs,
            )
        )
    write_rows(output_csv, rows)
    print(f"wrote {output_csv}", flush=True)


def run_c2(output_csv=BASELINE_DIR / "tidyvoice_c2_results.csv") -> None:
    train, test = load_fixed_splits(
        EMBEDDINGS,
        CHUNKSIZE,
        MAX_TRAIN_ROWS_PER_CLASS,
        MAX_EVAL_ROWS_PER_CLASS,
    )
    train_labels = train["label"].to_numpy(dtype=np.int8, copy=False)
    rows: list[dict[str, float | int | str]] = []
    for embedding in tqdm(EMBEDDINGS, desc="C2 PAV", unit="embedding"):
        train_scores = train[score_column(embedding)].to_numpy(dtype=np.float64, copy=False)
        test_scores = test[score_column(embedding)].to_numpy(dtype=np.float64, copy=False)
        calibrator = train_pav(train_scores, train_labels)
        test_llrs = calibrator.predict_llr(test_scores)
        rows.append(
            evaluate_calibration(
                test,
                embedding,
                "C2",
                "c2_pav",
                "pav_isotonic_regression",
                decision_scores=test_llrs,
                llrs=test_llrs,
            )
        )
    write_rows(output_csv, rows)
    print(f"wrote {output_csv}", flush=True)
