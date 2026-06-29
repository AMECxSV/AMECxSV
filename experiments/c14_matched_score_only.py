from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

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


OUTPUT_CSV = BASELINE_DIR / "tidyvoice_matched_score_only_results.csv"
CHUNKSIZE = 250_000
MODEL_SETTING = {
    key: value
    for key, value in BEST_MLP_SETTING.items()
    if key != "c_value"
}
C_VALUE = float(BEST_MLP_SETTING["c_value"])
RAW_FEATURE_COUNT = len(EMBEDDINGS) + 1 + len(C5_COLS)
EXPECTED_EXPANDED_FEATURE_COUNT = 112
MODES = ("amec_full", "matched_score_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train AMEC-FC and a strict architecture-matched score-only "
            "control with constant metadata placeholders."
        )
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use at most 2,000 rows per class and train for two epochs.",
    )
    return parser.parse_args()


def score_matrix(frame) -> np.ndarray:
    columns = [f"score_{embedding}" for embedding in EMBEDDINGS]
    return frame[columns].to_numpy(dtype=np.float64, copy=False)


def metadata_matrix(frame) -> np.ndarray:
    language = frame["target"].to_numpy(dtype=np.float64, copy=False)[:, None]
    reliability = frame[C5_COLS].to_numpy(dtype=np.float64, copy=False)
    return np.concatenate([language, reliability], axis=1)


def feature_matrix(frame, mode: str) -> np.ndarray:
    scores = score_matrix(frame)
    if mode == "amec_full":
        metadata = metadata_matrix(frame)
    elif mode == "matched_score_only":
        metadata = np.zeros(
            (scores.shape[0], 1 + len(C5_COLS)),
            dtype=np.float64,
        )
    else:
        raise ValueError(f"Unknown matched-control mode: {mode}")

    features = np.concatenate([scores, metadata], axis=1)
    if features.shape[1] != RAW_FEATURE_COUNT:
        raise RuntimeError(
            f"Expected {RAW_FEATURE_COUNT} raw features, "
            f"got {features.shape[1]}"
        )
    return features


def trainable_parameter_count(model) -> int:
    if model.net is None:
        raise RuntimeError("Cannot count parameters before model fitting")
    return int(
        sum(parameter.numel() for parameter in model.net.parameters())
    )


def fit_mode(
    mode: str,
    train,
    validation,
    test,
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
    test_labels: np.ndarray,
    sample_weights: np.ndarray,
    *,
    device,
    epochs: int,
) -> dict[str, object]:
    setting = dict(MODEL_SETTING)
    setting["epochs"] = epochs
    experiment = Experiment(
        "MATCHED_SCORE_CONTROL",
        mode,
        C_VALUE,
        **setting,
    )
    model = make_rich_model(
        experiment,
        device,
        num_scores=len(EMBEDDINGS),
    )

    train_features = feature_matrix(train, mode)
    validation_features = feature_matrix(validation, mode)
    test_features = feature_matrix(test, mode)
    model.fit(
        train_features,
        train_labels,
        sample_weights,
        desc=f"train {mode}",
        validation_features=validation_features,
        validation_labels=validation_labels,
    )

    if model.backend is None:
        raise RuntimeError("Rich-feature backend was not initialized")
    expanded_feature_count = int(
        model.backend.augment(train_features[:1]).shape[1]
    )
    if expanded_feature_count != EXPECTED_EXPANDED_FEATURE_COUNT:
        raise RuntimeError(
            "Architecture mismatch: expected "
            f"{EXPECTED_EXPANDED_FEATURE_COUNT} expanded features, "
            f"got {expanded_feature_count}"
        )

    test_llrs = model.decision_function(test_features)
    row: dict[str, object] = {
        "dataset": DATASET_NAME,
        "split": "test",
        "experiment": "strict_architecture_matched_score_only",
        "system": mode,
        "metadata_input": (
            "language_plus_reliability"
            if mode == "amec_full"
            else "four_constant_zero_placeholders"
        ),
        "model": "rich_feature_mlp",
        "score_count": len(EMBEDDINGS),
        "metadata_slot_count": 1 + len(C5_COLS),
        "raw_feature_count": train_features.shape[1],
        "expanded_feature_count": expanded_feature_count,
        "trainable_parameter_count": trainable_parameter_count(model),
        "hidden_dim": setting["hidden_dim"],
        "dropout": setting["dropout"],
        "learning_rate": setting["learning_rate"],
        "epochs": setting["epochs"],
        "batch_size": setting["batch_size"],
        "C": C_VALUE,
        "selection_split": "validation",
        "llr_scale": model.llr_scale,
        "llr_bias": model.llr_bias,
    }
    row.update(metric_block(test_llrs, test_llrs, test_labels))
    return row


def validate_matching(rows: list[dict[str, object]]) -> None:
    matched_fields = (
        "model",
        "score_count",
        "metadata_slot_count",
        "raw_feature_count",
        "expanded_feature_count",
        "trainable_parameter_count",
        "hidden_dim",
        "dropout",
        "learning_rate",
        "epochs",
        "batch_size",
        "C",
        "selection_split",
    )
    reference = rows[0]
    mismatches = {
        field: [row[field] for row in rows]
        for field in matched_fields
        if any(row[field] != reference[field] for row in rows[1:])
    }
    if mismatches:
        raise RuntimeError(
            f"Controls are not architecture matched: {mismatches}"
        )


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
    print(f"wrote {path}", flush=True)


def main() -> None:
    args = parse_args()
    max_rows = 2_000 if args.smoke_test else None
    epochs = 2 if args.smoke_test else int(MODEL_SETTING["epochs"])
    device = require_cuda()
    train, validation, test = load_fixed_splits(
        EMBEDDINGS,
        CHUNKSIZE,
        max_rows,
        max_rows,
        return_validation=True,
    )
    if validation is None:
        raise ValueError(
            "The matched score-only control requires a validation split"
        )

    train_labels = train["label"].to_numpy(dtype=np.int8, copy=False)
    validation_labels = validation["label"].to_numpy(
        dtype=np.int8,
        copy=False,
    )
    test_labels = test["label"].to_numpy(dtype=np.int8, copy=False)
    sample_weights = balanced_sample_weights(train_labels)

    rows = [
        fit_mode(
            mode,
            train,
            validation,
            test,
            train_labels,
            validation_labels,
            test_labels,
            sample_weights,
            device=device,
            epochs=epochs,
        )
        for mode in MODES
    ]
    validate_matching(rows)

    full, control = rows
    for metric in ("eer_pct", "Cllr", "actDCF_p001", "actDCF_p01"):
        delta = float(control[metric]) - float(full[metric])
        for row in rows:
            row[f"matched_control_minus_amec_{metric}"] = delta
    write_rows(args.output, rows)


if __name__ == "__main__":
    main()
