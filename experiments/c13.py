from __future__ import annotations

import argparse
import csv
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from selective import confidence_from_llr, confidence_threshold
from c8 import (
    feature_matrix as c8_feature_matrix,
    fit_score_stats,
    individual_weights,
    score_matrix,
)
from c10 import feature_matrix as c10_feature_matrix
from common import (
    BASELINE_DIR,
    BEST_MLP_SETTING,
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_WORKERS,
    DATASET_NAME,
    EMBEDDINGS,
    PROJECT_ROOT,
    Experiment,
    balanced_sample_weights,
    load_fixed_splits,
    make_rich_model,
    metric_block,
    require_cuda,
    sorted_score_context,
    stable_seed,
    weighted_metric_block,
)


FOLLOWUP_TABLE = Path(
    os.environ.get(
        "AMECXSV_FOLLOWUP_TABLE",
        PROJECT_ROOT / "data" / "tidyvoice_trial_metadata.parquet",
    )
)
LANGUAGE_OUTPUT = BASELINE_DIR / "tidyvoice_c13_language_analysis.csv"
DURATION_OUTPUT = BASELINE_DIR / "tidyvoice_c13_duration_analysis.csv"
BOOTSTRAP_OUTPUT = (
    BASELINE_DIR / "tidyvoice_c13_speaker_cluster_bootstrap.csv"
)

CHUNKSIZE = 250_000
COVERAGE_TARGET = 0.80
C8_FEATURE_MODE = "c8_3_entropy_fusion"
C10_FEATURE_MODE = "c10_six_score_lang_reliability"
CLUSTER_COLUMN = "enroll_speaker"
BOOTSTRAP_METRICS = (
    "eer_pct",
    "Cllr",
    "actDCF_p001",
    "actDCF_p01",
)


@dataclass(frozen=True)
class SystemOutput:
    name: str
    llrs: np.ndarray
    accepted: np.ndarray
    coverage_target: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run C13 language, duration, and speaker-clustered paired "
            "bootstrap analyses for C8 versus C10."
        )
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=BOOTSTRAP_SAMPLES,
        help=f"Speaker-clustered bootstrap replicates (default: {BOOTSTRAP_SAMPLES}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BASELINE_DIR,
        help="Directory for the three C13 CSV files.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use small balanced splits, two epochs, and at most three bootstrap samples.",
    )
    return parser.parse_args()


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


def load_followup_test(test: pd.DataFrame) -> pd.DataFrame:
    if not FOLLOWUP_TABLE.exists():
        raise FileNotFoundError(
            f"Missing C13 follow-up table: {FOLLOWUP_TABLE}"
        )
    columns = [
        "split",
        "trial_id",
        "enroll_speaker",
        "test_speaker",
        "enroll_language",
        "test_language",
        "language_pair",
        "language_condition",
        "enroll_duration_sec",
        "test_duration_sec",
        "min_duration",
        "duration_ratio",
    ]
    followup = pd.read_parquet(
        FOLLOWUP_TABLE,
        columns=columns,
        filters=[("split", "==", "test")],
    )
    if followup["trial_id"].duplicated().any():
        raise ValueError("Duplicate trial_id values in the C13 follow-up test data")

    metadata = test[["trial_id"]].merge(
        followup.drop(columns=["split"]),
        on="trial_id",
        how="left",
        sort=False,
        validate="one_to_one",
    )
    required = [column for column in metadata.columns if column != "trial_id"]
    missing = metadata[required].isna().sum()
    if int(missing.sum()) != 0:
        details = missing[missing > 0].to_dict()
        raise ValueError(f"Missing C13 metadata after trial_id join: {details}")
    if not np.array_equal(
        metadata["trial_id"].to_numpy(),
        test["trial_id"].to_numpy(),
    ):
        raise RuntimeError("C13 metadata join changed the test trial order")
    return metadata


def train_models(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    device,
    epochs: int,
) -> tuple[list[SystemOutput], np.ndarray, np.ndarray]:
    model_setting = {
        key: value
        for key, value in BEST_MLP_SETTING.items()
        if key != "c_value"
    }
    model_setting["epochs"] = epochs
    c_value = float(BEST_MLP_SETTING["c_value"])

    train_labels = train["label"].to_numpy(dtype=np.int8, copy=False)
    validation_labels = validation["label"].to_numpy(
        dtype=np.int8, copy=False
    )
    test_labels = test["label"].to_numpy(dtype=np.int8, copy=False)
    sample_weights = balanced_sample_weights(train_labels)

    train_scores = score_matrix(train)
    validation_scores = score_matrix(validation)
    test_scores = score_matrix(test)
    mean, std = fit_score_stats(train_scores)
    weights = individual_weights(train_scores, train_labels)
    c8_train = c8_feature_matrix(
        C8_FEATURE_MODE, train_scores, mean, std, weights
    )
    c8_validation = c8_feature_matrix(
        C8_FEATURE_MODE, validation_scores, mean, std, weights
    )
    c8_test = c8_feature_matrix(
        C8_FEATURE_MODE, test_scores, mean, std, weights
    )

    c8_experiment = Experiment(
        "C8-2", C8_FEATURE_MODE, c_value, **model_setting
    )
    c8_model = make_rich_model(
        c8_experiment, device, num_scores=len(EMBEDDINGS)
    )
    c8_model.fit(
        c8_train,
        train_labels,
        sample_weights,
        desc="train C13 C8-2",
        validation_features=c8_validation,
        validation_labels=validation_labels,
    )
    c8_validation_llrs = c8_model.decision_function(c8_validation)
    c8_test_llrs = c8_model.decision_function(c8_test)
    del c8_train, c8_validation, c8_test, c8_model

    c10_train = c10_feature_matrix(train)
    c10_validation = c10_feature_matrix(validation)
    c10_test = c10_feature_matrix(test)
    c10_experiment = Experiment(
        "C10-2", C10_FEATURE_MODE, c_value, **model_setting
    )
    c10_model = make_rich_model(
        c10_experiment, device, num_scores=len(EMBEDDINGS)
    )
    c10_model.fit(
        c10_train,
        train_labels,
        sample_weights,
        desc="train C13 C10-2",
        validation_features=c10_validation,
        validation_labels=validation_labels,
    )
    c10_validation_llrs = c10_model.decision_function(c10_validation)
    c10_test_llrs = c10_model.decision_function(c10_test)
    del c10_train, c10_validation, c10_test, c10_model

    systems: list[SystemOutput] = []
    for name, validation_llrs, test_llrs in [
        ("C8-2", c8_validation_llrs, c8_test_llrs),
        ("C10-2", c10_validation_llrs, c10_test_llrs),
    ]:
        systems.append(
            SystemOutput(
                name=name,
                llrs=test_llrs,
                accepted=np.ones(test_llrs.shape[0], dtype=bool),
                coverage_target=1.0,
            )
        )
        _, validation_confidence = confidence_from_llr(validation_llrs)
        _, test_confidence = confidence_from_llr(test_llrs)
        reject_threshold = confidence_threshold(
            validation_confidence, COVERAGE_TARGET
        )
        systems.append(
            SystemOutput(
                name=name.replace("-2", "-3"),
                llrs=test_llrs,
                accepted=test_confidence >= reject_threshold,
                coverage_target=COVERAGE_TARGET,
            )
        )
    return systems, c8_test_llrs, c10_test_llrs


def group_result_row(
    *,
    analysis: str,
    grouping: str,
    group: str,
    mask: np.ndarray,
    system: SystemOutput,
    labels: np.ndarray,
) -> dict[str, object]:
    accepted = mask & system.accepted
    group_n = int(np.sum(mask))
    accepted_n = int(np.sum(accepted))
    row: dict[str, object] = {
        "dataset": DATASET_NAME,
        "split": "test",
        "analysis": analysis,
        "grouping": grouping,
        "group": group,
        "system": system.name,
        "coverage_target": system.coverage_target,
        "group_n": group_n,
        "group_target_n": int(np.sum(labels[mask] == 1)),
        "group_nontarget_n": int(np.sum(labels[mask] == 0)),
        "accepted_n": accepted_n,
        "rejected_n": group_n - accepted_n,
        "coverage": accepted_n / group_n if group_n else math.nan,
    }
    metrics = metric_block(
        system.llrs[accepted],
        system.llrs[accepted],
        labels[accepted],
    )
    for key, value in metrics.items():
        row[f"accepted_{key}"] = value
    return row


def language_rows(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    systems: list[SystemOutput],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    conditions = metadata["language_condition"].to_numpy(dtype=str)
    pairs = metadata["language_pair"].to_numpy(dtype=str)

    for group in ("same_language", "cross_language"):
        mask = conditions == group
        for system in systems:
            rows.append(
                group_result_row(
                    analysis="language",
                    grouping="language_condition",
                    group=group,
                    mask=mask,
                    system=system,
                    labels=labels,
                )
            )

    for group in sorted(np.unique(pairs)):
        mask = pairs == group
        for system in systems:
            rows.append(
                group_result_row(
                    analysis="language",
                    grouping="language_pair",
                    group=group,
                    mask=mask,
                    system=system,
                    labels=labels,
                )
            )
    return rows


def duration_rows(
    metadata: pd.DataFrame,
    labels: np.ndarray,
    systems: list[SystemOutput],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    enroll_duration = metadata["enroll_duration_sec"].to_numpy(
        dtype=np.float64, copy=False
    )
    test_duration = metadata["test_duration_sec"].to_numpy(
        dtype=np.float64, copy=False
    )
    min_duration = np.minimum(enroll_duration, test_duration)
    duration_ratio = min_duration / np.maximum(
        enroll_duration, test_duration
    )

    groups = [
        ("min_duration_sec", "short_<3", min_duration < 3.0),
        (
            "min_duration_sec",
            "medium_3_to_<10",
            (min_duration >= 3.0) & (min_duration < 10.0),
        ),
        ("min_duration_sec", "long_>=10", min_duration >= 10.0),
        ("duration_ratio", "severe_<0.5", duration_ratio < 0.5),
        (
            "duration_ratio",
            "moderate_0.5_to_<0.8",
            (duration_ratio >= 0.5) & (duration_ratio < 0.8),
        ),
        ("duration_ratio", "matched_>=0.8", duration_ratio >= 0.8),
    ]
    for grouping, group, mask in groups:
        for system in systems:
            row = group_result_row(
                analysis="duration",
                grouping=grouping,
                group=group,
                mask=mask,
                system=system,
                labels=labels,
            )
            row["min_duration_definition"] = (
                "min(enroll_duration_sec,test_duration_sec)"
            )
            row["duration_ratio_definition"] = (
                "min(enroll_duration_sec,test_duration_sec)/"
                "max(enroll_duration_sec,test_duration_sec)"
            )
            rows.append(row)
    return rows


def paired_cluster_bootstrap_rows(
    labels: np.ndarray,
    c8_llrs: np.ndarray,
    c10_llrs: np.ndarray,
    clusters: np.ndarray,
    *,
    samples: int,
) -> list[dict[str, object]]:
    if samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    cluster_names, cluster_codes = np.unique(
        clusters.astype(str), return_inverse=True
    )
    cluster_n = int(cluster_names.shape[0])
    if cluster_n < 2:
        raise ValueError("At least two enrollment-speaker clusters are required")

    llrs_by_system = {"C8-2": c8_llrs, "C10-2": c10_llrs}
    contexts = {
        name: sorted_score_context(llrs, llrs, labels)
        for name, llrs in llrs_by_system.items()
    }
    point = {
        name: metric_block(llrs, llrs, labels)
        for name, llrs in llrs_by_system.items()
    }
    rng = np.random.default_rng(stable_seed("C13 speaker-clustered paired"))
    seeds = rng.integers(
        0,
        np.iinfo(np.uint32).max,
        size=samples,
        dtype=np.uint32,
    )

    def one_sample(seed: np.uint32) -> dict[str, dict[str, float]]:
        local_rng = np.random.default_rng(int(seed))
        sampled = local_rng.integers(
            0, cluster_n, size=cluster_n, dtype=np.int32
        )
        cluster_counts = np.bincount(sampled, minlength=cluster_n)
        trial_counts = cluster_counts[cluster_codes]
        return {
            name: {
                metric: float(value)
                for metric, value in weighted_metric_block(
                    context, trial_counts
                ).items()
                if metric in BOOTSTRAP_METRICS
            }
            for name, context in contexts.items()
        }

    results = []
    with ThreadPoolExecutor(max_workers=BOOTSTRAP_WORKERS) as pool:
        iterator = pool.map(one_sample, seeds)
        for result in tqdm(
            iterator,
            total=samples,
            desc="C13 speaker-clustered paired bootstrap",
            unit="sample",
        ):
            results.append(result)

    rows: list[dict[str, object]] = []
    for metric in BOOTSTRAP_METRICS:
        differences = np.asarray(
            [
                result["C10-2"][metric] - result["C8-2"][metric]
                for result in results
            ],
            dtype=np.float64,
        )
        differences = differences[np.isfinite(differences)]
        if differences.size == 0:
            low = high = half_width = probability_better = p_two_sided = (
                math.nan
            )
        else:
            low, high = np.percentile(differences, [2.5, 97.5])
            half_width = (high - low) / 2.0
            probability_better = float(np.mean(differences < 0.0))
            p_two_sided = float(
                min(
                    1.0,
                    2.0
                    * min(
                        np.mean(differences <= 0.0),
                        np.mean(differences >= 0.0),
                    ),
                )
            )
        rows.append(
            {
                "dataset": DATASET_NAME,
                "split": "test",
                "analysis": "speaker_clustered_paired_bootstrap",
                "cluster_column": CLUSTER_COLUMN,
                "cluster_n": cluster_n,
                "bootstrap_samples": samples,
                "metric": metric,
                "better_direction": "lower",
                "c8_system": "C8-2",
                "c8_value": point["C8-2"][metric],
                "c10_system": "C10-2",
                "c10_value": point["C10-2"][metric],
                "difference_definition": "C10-2 minus C8-2",
                "difference": (
                    float(point["C10-2"][metric])
                    - float(point["C8-2"][metric])
                ),
                "difference_ci95_low": low,
                "difference_ci95_high": high,
                "difference_ci95_half_width": half_width,
                "probability_c10_better": probability_better,
                "paired_p_two_sided": p_two_sided,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")

    max_rows = 2_000 if args.smoke_test else None
    epochs = 2 if args.smoke_test else int(BEST_MLP_SETTING["epochs"])
    bootstrap_samples = (
        min(args.bootstrap_samples, 3)
        if args.smoke_test
        else args.bootstrap_samples
    )

    device = require_cuda()
    train, validation, test = load_fixed_splits(
        EMBEDDINGS,
        CHUNKSIZE,
        max_rows,
        max_rows,
        return_validation=True,
    )
    if validation is None:
        raise ValueError("C13 requires the fixed validation split")
    metadata = load_followup_test(test)
    labels = test["label"].to_numpy(dtype=np.int8, copy=False)
    systems, c8_llrs, c10_llrs = train_models(
        train,
        validation,
        test,
        device=device,
        epochs=epochs,
    )

    language_path = args.output_dir / LANGUAGE_OUTPUT.name
    duration_path = args.output_dir / DURATION_OUTPUT.name
    bootstrap_path = args.output_dir / BOOTSTRAP_OUTPUT.name

    write_rows(language_path, language_rows(metadata, labels, systems))
    write_rows(duration_path, duration_rows(metadata, labels, systems))
    bootstrap_rows = paired_cluster_bootstrap_rows(
        labels,
        c8_llrs,
        c10_llrs,
        metadata[CLUSTER_COLUMN].to_numpy(dtype=str),
        samples=bootstrap_samples,
    )
    write_rows(bootstrap_path, bootstrap_rows)


if __name__ == "__main__":
    main()
