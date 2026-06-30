from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    BASELINE_DIR,
    BEST_MLP_SETTING,
    C5_COLS,
    DATASET_NAME,
    EMBEDDINGS,
    Experiment,
    add_metric_ci,
    balanced_sample_weights,
    load_fixed_splits,
    make_rich_model,
    metric_block,
    require_cuda,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_CSV = BASELINE_DIR / "tidyvoice_predicted_language_control.csv"
CHUNKSIZE = 250_000
MODEL_SETTING = {
    key: value
    for key, value in BEST_MLP_SETTING.items()
    if key != "c_value"
}
C_VALUE = float(BEST_MLP_SETTING["c_value"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replace oracle language match with frozen-LID predicted "
            "language match while keeping AMEC score and duration inputs fixed."
        )
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=PROJECT_ROOT / "dataset" / "trial_protocol.parquet",
    )
    parser.add_argument(
        "--utterance-predictions",
        type=Path,
        default=(
            PROJECT_ROOT
            / "dataset"
            / "voxlingua107_language_predictions.csv"
        ),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def read_table(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=columns)
    return pd.read_csv(path, usecols=columns)


def score_columns() -> list[str]:
    return [f"score_{embedding}" for embedding in EMBEDDINGS]


def predicted_language_match(
    frame,
    protocol: pd.DataFrame,
    predictions: pd.DataFrame,
) -> np.ndarray:
    trial_rows = frame[["trial_id"]].merge(
        protocol,
        on="trial_id",
        how="left",
        sort=False,
        validate="one_to_one",
    )
    language_by_utterance = predictions.set_index(
        "utterance_id"
    )["predicted_language"]
    enroll = trial_rows["enroll_utt"].map(language_by_utterance)
    test = trial_rows["test_utt"].map(language_by_utterance)
    if enroll.isna().any() or test.isna().any():
        raise ValueError(
            "Missing frozen-LID predictions for protocol utterances"
        )
    return (enroll.to_numpy() == test.to_numpy()).astype(np.float64)


def feature_matrix(
    frame,
    protocol: pd.DataFrame,
    predictions: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    scores = frame[score_columns()].to_numpy(
        dtype=np.float64,
        copy=False,
    )
    predicted_match = predicted_language_match(
        frame,
        protocol,
        predictions,
    )
    reliability = frame[C5_COLS].to_numpy(
        dtype=np.float64,
        copy=False,
    )
    features = np.concatenate(
        [scores, predicted_match[:, None], reliability],
        axis=1,
    )
    return features, predicted_match


def write_row(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    max_rows = 2_000 if args.smoke_test else None
    epochs = 2 if args.smoke_test else int(MODEL_SETTING["epochs"])
    bootstrap_samples = (
        min(args.bootstrap_samples, 3)
        if args.smoke_test
        else args.bootstrap_samples
    )

    protocol = read_table(
        args.protocol,
        ["trial_id", "enroll_utt", "test_utt"],
    )
    predictions = read_table(
        args.utterance_predictions,
        ["utterance_id", "predicted_language"],
    )
    if predictions["utterance_id"].duplicated().any():
        raise ValueError("Duplicate utterance predictions")

    train, validation, test = load_fixed_splits(
        EMBEDDINGS,
        CHUNKSIZE,
        max_rows,
        max_rows,
        return_validation=True,
    )
    if validation is None:
        raise ValueError(
            "Predicted-language control requires a validation split"
        )

    x_train, _ = feature_matrix(train, protocol, predictions)
    x_validation, _ = feature_matrix(validation, protocol, predictions)
    x_test, predicted_test_match = feature_matrix(
        test,
        protocol,
        predictions,
    )
    y_train = train["label"].to_numpy(dtype=np.int8, copy=False)
    y_validation = validation["label"].to_numpy(
        dtype=np.int8,
        copy=False,
    )
    y_test = test["label"].to_numpy(dtype=np.int8, copy=False)

    setting = dict(MODEL_SETTING)
    setting["epochs"] = epochs
    experiment = Experiment(
        "PREDICTED_LANGUAGE",
        "amec_frozen_lid_language",
        C_VALUE,
        **setting,
    )
    model = make_rich_model(
        experiment,
        require_cuda(),
        num_scores=len(EMBEDDINGS),
    )
    model.fit(
        x_train,
        y_train,
        balanced_sample_weights(y_train),
        desc="train AMEC with frozen-LID language match",
        validation_features=x_validation,
        validation_labels=y_validation,
    )
    test_llrs = model.decision_function(x_test)
    oracle_match = test["target"].to_numpy(
        dtype=np.float64,
        copy=False,
    )
    row: dict[str, object] = {
        "dataset": DATASET_NAME,
        "split": "test",
        "system": "AMEC-FC",
        "language_metadata_source": "frozen_voxlingua107_ecapa",
        "language_match_agreement": float(
            np.mean(predicted_test_match == oracle_match)
        ),
        "model": "rich_feature_mlp",
        "raw_feature_count": x_train.shape[1],
        "hidden_dim": setting["hidden_dim"],
        "dropout": setting["dropout"],
        "learning_rate": setting["learning_rate"],
        "epochs": setting["epochs"],
        "batch_size": setting["batch_size"],
        "C": C_VALUE,
        "selection_split": "validation",
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
                desc="AMEC frozen-LID predicted language",
            )
        finally:
            common.BOOTSTRAP_SAMPLES = original_samples
    write_row(args.output, row)
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
