#!/usr/bin/env python3
"""Train the AMEC MLP head on external TidyVoice backbone scores."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch

from external_common import (
    DEFAULT_CONFIG,
    git_commit,
    load_config,
    make_run_id,
    metric_block,
    metric_rows,
    resolve_path,
    utc_timestamp,
    write_json,
    write_metric_report,
    write_table,
)


SOURCE_SYSTEM = "amec_head_external"
DEFAULT_HIDDEN_DIM = 128
DEFAULT_DROPOUT = 0.15
DEFAULT_LEARNING_RATE = 5.0e-4
DEFAULT_EPOCHS = 120
DEFAULT_BATCH_SIZE = 262_144
DEFAULT_C_VALUE = 10.0
DEFAULT_COVERAGE = 0.80
EPS = 1.0e-6

BASE_COLUMNS = [
    "dataset",
    "split",
    "trial_id",
    "label",
    "target",
    "enroll_utt",
    "test_utt",
    "enroll_speaker",
    "test_speaker",
    "enroll_language",
    "test_language",
    "language_pair",
    "language_condition",
    "enroll_duration_sec",
    "test_duration_sec",
    "source_system",
    "checkpoint_id",
]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--calibration-score",
        action="append",
        required=True,
        help="Calibration score table. Use name=path to set the score feature name. Can be repeated.",
    )
    parser.add_argument(
        "--validation-score",
        action="append",
        help=(
            "Optional validation score table. Use name=path to set the score feature name. "
            "Can be repeated and must match --calibration-score names."
        ),
    )
    parser.add_argument(
        "--test-score",
        action="append",
        required=True,
        help="Test score table. Use name=path to set the score feature name. Can be repeated.",
    )
    parser.add_argument("--score-column", default="score")
    parser.add_argument(
        "--feature-set",
        choices=("score_only", "score_plus_metadata"),
        default="score_plus_metadata",
        help=(
            "Use only the supplied external score stream(s), or append "
            "language-match and duration-reliability metadata."
        ),
    )
    parser.add_argument("--tag", help="Output tag. Defaults to joined score names.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--c-value", type=float, default=DEFAULT_C_VALUE)
    parser.add_argument("--coverage-target", type=float, default=DEFAULT_COVERAGE)
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--output-scores", type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--model-output", type=Path)
    return parser.parse_args(argv)


def safe_name(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = value.strip("._-")
    return value or "score"


def parse_named_path(value: str) -> tuple[str | None, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        return safe_name(name), Path(path)
    return None, Path(value)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return device


def table_columns(path: Path) -> list[str]:
    path = resolve_path(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        import pyarrow.parquet as pq

        return pq.ParquetFile(path).schema.names
    if suffix in {".csv", ".txt", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep, nrows=0).columns.tolist()
    raise ValueError(f"Unsupported score table: {path}")


def read_selected_table(path: Path, columns: list[str]) -> pd.DataFrame:
    path = resolve_path(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=columns)
    sep = "\t" if suffix == ".tsv" else ","
    return pd.read_csv(path, sep=sep, usecols=columns)


def infer_source_name(path: Path, score_column: str) -> str:
    columns = table_columns(path)
    usecols = [column for column in ["source_system", score_column] if column in columns]
    if usecols:
        row = read_selected_table(path, usecols).head(1)
        if "source_system" in row and not row.empty:
            value = str(row["source_system"].iloc[0])
            if value and value != "nan":
                return safe_name(value)
    return safe_name(resolve_path(path).stem.replace("_scores", ""))


def load_score_table(path: Path, name: str | None, score_column: str, *, include_metadata: bool) -> pd.DataFrame:
    path = resolve_path(path)
    available = table_columns(path)
    if "trial_id" not in available or score_column not in available:
        raise ValueError(f"{path} must contain trial_id and {score_column}")

    score_name = name or infer_source_name(path, score_column)
    columns = ["trial_id", score_column]
    if include_metadata:
        columns.extend(column for column in BASE_COLUMNS if column in available and column not in columns)
    frame = read_selected_table(path, columns)
    frame = frame.rename(columns={score_column: f"score_{score_name}"})
    frame[f"score_{score_name}"] = pd.to_numeric(frame[f"score_{score_name}"], errors="coerce")
    return frame, score_name


def load_joined_scores(specs: list[str], score_column: str) -> tuple[pd.DataFrame, list[str]]:
    joined: pd.DataFrame | None = None
    names: list[str] = []
    for index, spec in enumerate(specs):
        name, path = parse_named_path(spec)
        frame, score_name = load_score_table(path, name, score_column, include_metadata=index == 0)
        if score_name in names:
            raise ValueError(f"Duplicate score name: {score_name}")
        names.append(score_name)
        if joined is None:
            joined = frame
        else:
            joined = joined.merge(frame[["trial_id", f"score_{score_name}"]], on="trial_id", how="inner", validate="one_to_one")
    if joined is None:
        raise ValueError("No score tables were provided")
    return joined.reset_index(drop=True), names


def validate_score_sets(calibration_names: list[str], test_names: list[str]) -> None:
    if calibration_names != test_names:
        raise ValueError(f"Calibration/test score names differ: {calibration_names} != {test_names}")


def label_array(frame: pd.DataFrame) -> np.ndarray:
    if "label" not in frame.columns:
        raise ValueError("Score table must contain label in the first input table")
    return pd.to_numeric(frame["label"], errors="raise").astype(np.int8).to_numpy(copy=False)


def duration_features(frame: pd.DataFrame) -> pd.DataFrame:
    if not {"enroll_duration_sec", "test_duration_sec"}.issubset(frame.columns):
        raise ValueError("Score table must contain enroll_duration_sec and test_duration_sec")
    enroll = pd.to_numeric(frame["enroll_duration_sec"], errors="coerce").to_numpy(dtype=np.float64)
    test = pd.to_numeric(frame["test_duration_sec"], errors="coerce").to_numpy(dtype=np.float64)
    enroll = np.maximum(enroll, EPS)
    test = np.maximum(test, EPS)
    min_duration = np.minimum(enroll, test)
    max_duration = np.maximum(enroll, test)
    return pd.DataFrame(
        {
            "min_duration": min_duration,
            "duration_ratio": min_duration / max_duration,
            "short_duration_risk": 1.0 / np.sqrt(min_duration),
        },
        index=frame.index,
    )


def same_language(frame: pd.DataFrame) -> np.ndarray:
    if "language_condition" in frame.columns:
        return (frame["language_condition"].astype(str) == "same_language").astype(np.float64).to_numpy()
    if {"enroll_language", "test_language"}.issubset(frame.columns):
        return (frame["enroll_language"].astype(str) == frame["test_language"].astype(str)).astype(np.float64).to_numpy()
    raise ValueError("Score table must contain language_condition or enroll/test_language")


def feature_frame(
    frame: pd.DataFrame,
    score_names: list[str],
    feature_set: str,
) -> pd.DataFrame:
    parts: dict[str, np.ndarray] = {}
    for name in score_names:
        column = f"score_{name}"
        parts[column] = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    if feature_set == "score_plus_metadata":
        duration = duration_features(frame)
        for column in duration.columns:
            parts[column] = duration[column].to_numpy(dtype=np.float64)
        parts["same_language"] = same_language(frame)
    elif feature_set != "score_only":
        raise ValueError(f"Unknown feature set: {feature_set}")
    features = pd.DataFrame(parts, index=frame.index)
    finite = np.isfinite(features.to_numpy(dtype=np.float64, copy=False))
    if not np.all(finite):
        medians = features.replace([np.inf, -np.inf], np.nan).median(numeric_only=True).fillna(0.0)
        features = features.replace([np.inf, -np.inf], np.nan).fillna(medians)
    return features


def balanced_sample_weights(labels: np.ndarray) -> np.ndarray:
    labels_bool = labels.astype(bool)
    target_n = int(np.sum(labels_bool))
    nontarget_n = int(labels_bool.shape[0] - target_n)
    if target_n == 0 or nontarget_n == 0:
        raise ValueError("Both target and nontarget calibration trials are required")
    weights = np.empty(labels_bool.shape[0], dtype=np.float32)
    weights[labels_bool] = 0.5 / target_n
    weights[~labels_bool] = 0.5 / nontarget_n
    return weights


class AmecMlp(torch.nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(feature_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def train_mlp(
    x_train: np.ndarray,
    labels: np.ndarray,
    *,
    device: torch.device,
    hidden_dim: int,
    dropout: float,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    c_value: float,
) -> tuple[AmecMlp, torch.Tensor, torch.Tensor]:
    torch.manual_seed(0)
    x_tensor = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y_tensor = torch.as_tensor(labels.astype(np.float32), dtype=torch.float32, device=device)
    w_tensor = torch.as_tensor(balanced_sample_weights(labels), dtype=torch.float32, device=device)

    mean = x_tensor.mean(dim=0, keepdim=True)
    std = x_tensor.std(dim=0, keepdim=True, unbiased=False).clamp_min(1.0e-6)
    x_scaled = (x_tensor - mean) / std

    model = AmecMlp(x_scaled.shape[1], hidden_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1.0 / max(c_value, 1.0e-12),
    )

    n_rows = int(x_scaled.shape[0])
    actual_batch_size = min(batch_size, n_rows)
    for epoch in range(epochs):
        order = torch.randperm(n_rows, device=device)
        model.train()
        losses = []
        for start in range(0, n_rows, actual_batch_size):
            idx = order[start : start + actual_batch_size]
            logits = model(x_scaled[idx])
            loss_terms = torch.nn.functional.binary_cross_entropy_with_logits(
                logits,
                y_tensor[idx],
                reduction="none",
            )
            weights = w_tensor[idx]
            loss = torch.sum(weights * loss_terms) / torch.sum(weights).clamp_min(1.0e-12)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if (epoch + 1) == 1 or (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            print(f"epoch={epoch + 1}/{epochs} loss={np.mean(losses):.6f}", flush=True)
    model.eval()
    return model, mean.detach(), std.detach()


def predict_mlp(
    model: AmecMlp,
    mean: torch.Tensor,
    std: torch.Tensor,
    x: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            batch = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            batch = (batch - mean.to(device)) / std.to(device)
            outputs.append(model(batch).detach().cpu().numpy().astype(np.float64, copy=False))
    return np.concatenate(outputs)


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def confidence_from_llr(llrs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    posterior = sigmoid(llrs)
    return posterior, np.maximum(posterior, 1.0 - posterior)


def decisions_from_confidence(posterior: np.ndarray, confidence: np.ndarray, threshold: float | None) -> np.ndarray:
    decisions = np.where(posterior >= 0.5, "target", "nontarget").astype(object)
    if threshold is not None:
        decisions[confidence < threshold] = "reject"
    return decisions


def decision_metrics(labels: np.ndarray, decisions: np.ndarray) -> dict[str, float | int]:
    labels_bool = labels.astype(bool)
    target_decision = decisions == "target"
    nontarget_decision = decisions == "nontarget"
    reject_decision = decisions == "reject"
    accepted = ~reject_decision

    total_n = int(labels.shape[0])
    target_n = int(np.sum(labels_bool))
    nontarget_n = total_n - target_n
    accepted_n = int(np.sum(accepted))
    correct_accepted = int(np.sum((target_decision & labels_bool) | (nontarget_decision & ~labels_bool)))
    return {
        "n": total_n,
        "target_n": target_n,
        "nontarget_n": nontarget_n,
        "accepted_n": accepted_n,
        "rejected_n": int(np.sum(reject_decision)),
        "coverage": accepted_n / total_n if total_n else math.nan,
        "effective_acc": correct_accepted / total_n if total_n else math.nan,
        "covered_acc": correct_accepted / accepted_n if accepted_n else math.nan,
        "FAR": float(np.sum(target_decision & ~labels_bool) / nontarget_n) if nontarget_n else math.nan,
        "FRR": (
            float(np.sum((nontarget_decision | reject_decision) & labels_bool) / target_n)
            if target_n
            else math.nan
        ),
    }


def reject_summary_rows(
    frame: pd.DataFrame,
    labels: np.ndarray,
    llrs: np.ndarray,
    decisions: np.ndarray,
    *,
    tag: str,
    input_feature_set: str,
    run_id: str,
    coverage_target: float,
    confidence_threshold: float,
) -> list[dict[str, Any]]:
    accepted = decisions != "reject"
    base = {
        "source_system": SOURCE_SYSTEM,
        "run_id": run_id,
        "timestamp_utc": utc_timestamp(),
        "git_commit": git_commit(),
        "dataset": first_value(frame, "dataset"),
        "split": first_value(frame, "split"),
        "score_kind": f"amec_mlp_{input_feature_set}_conf_reject",
        "feature_set": tag,
        "input_feature_set": input_feature_set,
        "coverage_target": coverage_target,
        "confidence_threshold": confidence_threshold,
    }
    rows: list[dict[str, Any]] = []

    def one(group_name: str, group_value: str, idx: np.ndarray) -> None:
        row = {**base, "group_name": group_name, "group_value": group_value}
        row.update(decision_metrics(labels[idx], decisions[idx]))
        if np.any(accepted[idx]):
            local = idx[accepted[idx]]
            accepted_metrics = metric_block(llrs[local], labels[local], llrs[local])
            for key, value in accepted_metrics.items():
                row[f"accepted_{key}"] = value
        rows.append(row)

    all_idx = np.arange(labels.shape[0])
    one("all", "all", all_idx)
    if "language_condition" in frame.columns:
        for value, subset in frame.groupby("language_condition", sort=True):
            one("language_condition", str(value), subset.index.to_numpy(dtype=np.int64))
    return rows


def first_value(frame: pd.DataFrame, column: str) -> str:
    if column not in frame.columns or frame.empty:
        return ""
    return str(frame[column].iloc[0])


def write_prediction_scores(
    test: pd.DataFrame,
    score_names: list[str],
    llrs: np.ndarray,
    posterior: np.ndarray,
    confidence: np.ndarray,
    decisions: np.ndarray,
    *,
    run_id: str,
    tag: str,
    path: Path,
) -> None:
    columns = [
        column
        for column in [
            "dataset",
            "split",
            "trial_id",
            "label",
            "target",
            "language_condition",
            "enroll_duration_sec",
            "test_duration_sec",
        ]
        if column in test.columns
    ]
    output = test[columns].copy()
    for name in score_names:
        output[f"raw_score_{name}"] = test[f"score_{name}"].to_numpy(copy=False)
    output["calibrated_score"] = llrs
    output["posterior_target"] = posterior
    output["confidence"] = confidence
    output["decision"] = decisions
    output["accepted"] = decisions != "reject"
    output["source_system"] = SOURCE_SYSTEM
    output["feature_set"] = tag
    output["run_id"] = run_id
    output["timestamp_utc"] = utc_timestamp()
    output["git_commit"] = git_commit()
    write_table(output, path)


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def default_path(kind: str, tag: str) -> Path:
    if kind == "scores":
        return Path(f"external/outputs/amec_head_external_{tag}_scores.parquet")
    if kind == "metrics":
        return Path(f"external/outputs/amec_head_external_{tag}_metrics.csv")
    if kind == "report":
        return Path(f"external/outputs/amec_head_external_{tag}.md")
    if kind == "model":
        return Path(f"external/outputs/amec_head_external_{tag}.pt")
    raise ValueError(kind)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    load_config(args.config)
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    device = choose_device(args.device)

    calibration, calibration_names = load_joined_scores(args.calibration_score, args.score_column)
    validation: pd.DataFrame | None = None
    if args.validation_score:
        validation, validation_names = load_joined_scores(args.validation_score, args.score_column)
        validate_score_sets(calibration_names, validation_names)
    test, test_names = load_joined_scores(args.test_score, args.score_column)
    validate_score_sets(calibration_names, test_names)
    default_tag = "_plus_".join(calibration_names)
    tag = safe_name(
        args.tag or f"{default_tag}_{args.feature_set}"
    )

    x_train_frame = feature_frame(
        calibration,
        calibration_names,
        args.feature_set,
    )
    x_validation_frame: pd.DataFrame | None = None
    if validation is not None:
        x_validation_frame = feature_frame(
            validation,
            calibration_names,
            args.feature_set,
        )
    x_test_frame = feature_frame(
        test,
        calibration_names,
        args.feature_set,
    )
    if x_train_frame.columns.tolist() != x_test_frame.columns.tolist():
        raise ValueError("Calibration/test feature columns differ")
    if x_validation_frame is not None and x_train_frame.columns.tolist() != x_validation_frame.columns.tolist():
        raise ValueError("Calibration/validation feature columns differ")
    y_train = label_array(calibration)
    y_validation = None if validation is None else label_array(validation)
    y_test = label_array(test)
    print(
        f"training tag={tag} device={device} train_rows={len(calibration)} "
        f"validation_rows={0 if validation is None else len(validation)} "
        f"test_rows={len(test)} feature_count={x_train_frame.shape[1]}",
        flush=True,
    )

    model, mean, std = train_mlp(
        x_train_frame.to_numpy(dtype=np.float64, copy=False),
        y_train,
        device=device,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        c_value=args.c_value,
    )
    train_llrs = predict_mlp(
        model,
        mean,
        std,
        x_train_frame.to_numpy(dtype=np.float64, copy=False),
        device=device,
        batch_size=args.batch_size,
    )
    validation_llrs = None
    if x_validation_frame is not None:
        validation_llrs = predict_mlp(
            model,
            mean,
            std,
            x_validation_frame.to_numpy(dtype=np.float64, copy=False),
            device=device,
            batch_size=args.batch_size,
        )
    test_llrs = predict_mlp(
        model,
        mean,
        std,
        x_test_frame.to_numpy(dtype=np.float64, copy=False),
        device=device,
        batch_size=args.batch_size,
    )

    _, train_confidence = confidence_from_llr(train_llrs)
    validation_posterior = None
    validation_confidence = None
    validation_decisions = None
    if validation_llrs is not None:
        validation_posterior, validation_confidence = confidence_from_llr(validation_llrs)
    posterior, confidence = confidence_from_llr(test_llrs)
    threshold_confidence_source = validation_confidence if validation_confidence is not None else train_confidence
    threshold_source = "validation" if validation_confidence is not None else "calibration"
    reject_threshold = float(np.quantile(threshold_confidence_source, 1.0 - args.coverage_target))
    if validation_posterior is not None and validation_confidence is not None:
        validation_decisions = decisions_from_confidence(validation_posterior, validation_confidence, reject_threshold)
    decisions = decisions_from_confidence(posterior, confidence, reject_threshold)

    run_id = make_run_id(f"{SOURCE_SYSTEM}_{tag}")
    metrics_rows: list[dict[str, Any]] = []
    metric_frames: list[tuple[pd.DataFrame, str, np.ndarray | None, np.ndarray | None]] = []
    if validation is not None and validation_llrs is not None:
        metric_frames.append((validation, "validation", validation_llrs, validation_decisions))
    metric_frames.append((test, "test", test_llrs, decisions))

    for frame, _, _, _ in metric_frames:
        for name in calibration_names:
            raw_frame = frame.copy()
            raw_frame["raw_score"] = raw_frame[f"score_{name}"]
            raw_rows = metric_rows(
                raw_frame,
                score_column="raw_score",
                llr_column=None,
                source_system=name,
                run_id=run_id,
                score_kind="raw_external_score",
                include_breakdowns=True,
            )
            metrics_rows.extend(raw_rows)

    for frame, split_name, llrs, frame_decisions in metric_frames:
        if llrs is None:
            continue
        calibrated = frame.copy()
        calibrated["calibrated_score"] = llrs
        full_rows = metric_rows(
            calibrated,
            score_column="calibrated_score",
            llr_column="calibrated_score",
            source_system=SOURCE_SYSTEM,
            run_id=run_id,
            score_kind=f"amec_mlp_{args.feature_set}_full_coverage",
            include_breakdowns=True,
        )
        for row in full_rows:
            row["feature_set"] = tag
            row["input_feature_set"] = args.feature_set
        metrics_rows.extend(full_rows)
        if frame_decisions is not None:
            frame_labels = y_validation if split_name == "validation" and y_validation is not None else y_test
            reject_rows = reject_summary_rows(
                frame,
                frame_labels,
                llrs,
                frame_decisions,
                tag=tag,
                input_feature_set=args.feature_set,
                run_id=run_id,
                coverage_target=args.coverage_target,
                confidence_threshold=reject_threshold,
            )
            for row in reject_rows:
                row["threshold_source"] = threshold_source
            metrics_rows.extend(reject_rows)

    scores_path = args.output_scores or default_path("scores", tag)
    metrics_path = args.metrics_output or default_path("metrics", tag)
    report_path = args.report_output or default_path("report", tag)
    model_path = resolve_path(args.model_output or default_path("model", tag))

    write_prediction_scores(
        test,
        calibration_names,
        test_llrs,
        posterior,
        confidence,
        decisions,
        run_id=run_id,
        tag=tag,
        path=scores_path,
    )
    metrics = pd.DataFrame(metrics_rows)
    write_table(metrics, metrics_path)
    write_metric_report(metrics, report_path, f"AMEC Head on External Scores: {tag}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "source_system": SOURCE_SYSTEM,
            "feature_set": tag,
            "input_feature_set": args.feature_set,
            "score_names": calibration_names,
            "feature_columns": x_train_frame.columns.tolist(),
            "run_id": run_id,
            "timestamp_utc": utc_timestamp(),
            "git_commit": git_commit(),
            "model_setting": {
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "c_value": args.c_value,
                "coverage_target": args.coverage_target,
                "device": str(device),
                "threshold_source": threshold_source,
            },
            "mean": mean.detach().cpu(),
            "std": std.detach().cpu(),
            "state_dict": model.state_dict(),
            "train_rows": int(len(calibration)),
            "validation_rows": 0 if validation is None else int(len(validation)),
            "test_rows": int(len(test)),
            "confidence_threshold": reject_threshold,
        },
        model_path,
    )
    write_json(
        {
            "source_system": SOURCE_SYSTEM,
            "feature_set": tag,
            "input_feature_set": args.feature_set,
            "score_names": calibration_names,
            "feature_columns": x_train_frame.columns.tolist(),
            "run_id": run_id,
            "timestamp_utc": utc_timestamp(),
            "git_commit": git_commit(),
            "threshold_source": threshold_source,
            "scores_output": str(resolve_path(scores_path)),
            "metrics_output": str(resolve_path(metrics_path)),
            "report_output": str(resolve_path(report_path)),
            "model_output": str(model_path),
        },
        model_path.with_suffix(".json"),
    )
    print(f"wrote {resolve_path(metrics_path)}")
    print(f"wrote {resolve_path(report_path)}")
    print(f"wrote {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
