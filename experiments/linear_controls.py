from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import (
    BASELINE_DIR,
    C5_COLS,
    DATASET_NAME,
    EMBEDDINGS,
    add_metric_ci,
    affine_llr_calibration,
    balanced_sample_weights,
    cllr,
    load_fixed_splits,
    metric_block,
)
from rich_mlp import RichFeatureMlpCalibrator


OUTPUT_CSV = BASELINE_DIR / "tidyvoice_linear_controls.csv"
CHUNKSIZE = 250_000
C_GRID = (0.1, 1.0, 10.0, 30.0)
MODES = (
    "linear_score_only_raw",
    "linear_score_metadata_raw",
    "linear_matched_score_only",
    "linear_amec_expanded",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the same-feature linear controls reported with AMECxSV."
        )
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def score_matrix(frame) -> np.ndarray:
    columns = [f"score_{embedding}" for embedding in EMBEDDINGS]
    return frame[columns].to_numpy(dtype=np.float64, copy=False)


def metadata_matrix(frame) -> np.ndarray:
    language = frame["target"].to_numpy(dtype=np.float64, copy=False)[:, None]
    reliability = frame[C5_COLS].to_numpy(dtype=np.float64, copy=False)
    return np.concatenate([language, reliability], axis=1)


def rich_expand(features: np.ndarray) -> np.ndarray:
    calibrator = RichFeatureMlpCalibrator(
        input_dim=features.shape[1],
        hidden_dim=24,
        dropout=0.0,
        learning_rate=5.0e-4,
        c_value=30.0,
        epochs=1,
        batch_size=1,
        device="cpu",
        num_scores=len(EMBEDDINGS),
    )
    return calibrator.augment(features)


def feature_matrix(frame, mode: str) -> np.ndarray:
    scores = score_matrix(frame)
    if mode == "linear_score_only_raw":
        return scores

    if mode == "linear_score_metadata_raw":
        return np.concatenate([scores, metadata_matrix(frame)], axis=1)

    if mode == "linear_matched_score_only":
        placeholders = np.zeros(
            (scores.shape[0], 1 + len(C5_COLS)),
            dtype=np.float64,
        )
        return rich_expand(np.concatenate([scores, placeholders], axis=1))

    if mode == "linear_amec_expanded":
        return rich_expand(
            np.concatenate([scores, metadata_matrix(frame)], axis=1)
        )

    raise ValueError(f"Unknown linear-control mode: {mode}")


def fit_mode(
    mode: str,
    train,
    validation,
    test,
    *,
    bootstrap_samples: int,
) -> dict[str, object]:
    x_train = feature_matrix(train, mode)
    x_validation = feature_matrix(validation, mode)
    x_test = feature_matrix(test, mode)
    y_train = train["label"].to_numpy(dtype=np.int8, copy=False)
    y_validation = validation["label"].to_numpy(dtype=np.int8, copy=False)
    y_test = test["label"].to_numpy(dtype=np.int8, copy=False)
    sample_weights = balanced_sample_weights(y_train)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_validation_scaled = scaler.transform(x_validation)
    x_test_scaled = scaler.transform(x_test)

    candidates: list[tuple[float, float, LogisticRegression, float, float]] = []
    for c_value in C_GRID:
        model = LogisticRegression(
            C=c_value,
            max_iter=1000,
            solver="lbfgs",
            random_state=0,
        )
        model.fit(
            x_train_scaled,
            y_train,
            sample_weight=sample_weights,
        )
        validation_raw = model.decision_function(x_validation_scaled)
        scale, bias = affine_llr_calibration(
            validation_raw,
            y_validation,
        )
        validation_llrs = scale * validation_raw + bias
        candidates.append(
            (
                cllr(validation_llrs, y_validation),
                c_value,
                model,
                scale,
                bias,
            )
        )

    _, c_value, model, scale, bias = min(
        candidates,
        key=lambda item: item[0],
    )
    test_raw = model.decision_function(x_test_scaled)
    test_llrs = scale * test_raw + bias
    row: dict[str, object] = {
        "dataset": DATASET_NAME,
        "split": "test",
        "system": mode,
        "head": "linear_logistic",
        "feature_count": x_train.shape[1],
        "selection_split": "validation",
        "C": c_value,
        "training_seed": 0,
        "llr_scale": scale,
        "llr_bias": bias,
    }
    row.update(metric_block(test_llrs, test_llrs, y_test))
    if bootstrap_samples > 0:
        import common

        original_samples = common.BOOTSTRAP_SAMPLES
        common.BOOTSTRAP_SAMPLES = bootstrap_samples
        try:
            add_metric_ci(
                row,
                test_llrs,
                test_llrs,
                y_test,
                desc=f"linear control {mode}",
            )
        finally:
            common.BOOTSTRAP_SAMPLES = original_samples
    return row


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
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
    args = parse_args()
    max_rows = 2_000 if args.smoke_test else None
    bootstrap_samples = (
        min(args.bootstrap_samples, 3)
        if args.smoke_test
        else args.bootstrap_samples
    )
    train, validation, test = load_fixed_splits(
        EMBEDDINGS,
        CHUNKSIZE,
        max_rows,
        max_rows,
        return_validation=True,
    )
    if validation is None:
        raise ValueError("Linear controls require a validation split")
    rows = [
        fit_mode(
            mode,
            train,
            validation,
            test,
            bootstrap_samples=bootstrap_samples,
        )
        for mode in MODES
    ]
    write_rows(args.output, rows)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
