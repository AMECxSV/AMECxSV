#!/usr/bin/env python3
"""Run MLP fusion ablations on the TidyVoice multi-similarity wide table."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from similarity_scores import SCORE_SPECS, trainable_parameter_count


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_MODEL_DIR = PROJECT_ROOT / "experiments"
if str(BASELINE_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(BASELINE_MODEL_DIR))

from c8 import threshold_for_max_accuracy, threshold_metrics  # noqa: E402
from common import C5_COLS, EMBEDDINGS, QMF_COLS, balanced_sample_weights, metric_block  # noqa: E402


DEFAULT_INPUT = PROJECT_ROOT / "similarity" / "outputs" / "tidyvoice_similarity_scores_full.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "similarity" / "outputs" / "tidyvoice_similarity_mlp_fusion_results.csv"
MODEL_SETTING = {
    "hidden_dim": 128,
    "dropout": 0.15,
    "learning_rate": 5.0e-4,
    "epochs": 120,
    "batch_size": 262_144,
}
C_VALUE = 10.0
SPLIT_ALIASES = {
    "calibration": {"calibration", "train", "0"},
    "validation": {"validation", "dev", "2"},
    "test": {"test", "1"},
}
METADATA_COLUMNS = {
    "none": [],
    "c10": ["target", *C5_COLS],
    "full": ["target", *QMF_COLS, *C5_COLS],
}
DEFAULT_FUSION_METRICS = [
    "cosine",
    "centered_cosine",
    "whitened_cosine",
    "wccn_cosine",
    "lda_cosine",
    "neg_mahalanobis",
    "asnorm_cosine",
]


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-table", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-split", default="calibration")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--embeddings", help="Comma-separated embedding names. Defaults to embeddings present in the table.")
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_FUSION_METRICS),
        help="Comma-separated metric names for each embedding fusion block.",
    )
    parser.add_argument("--metadata-mode", choices=tuple(METADATA_COLUMNS), default="full")
    parser.add_argument("--fusion-dims", default="1,2,4")
    parser.add_argument("--shared-fusion", action="store_true")
    parser.add_argument("--include-cosine-reference", action="store_true")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--max-rows-per-class", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--hidden-dim", type=int, default=MODEL_SETTING["hidden_dim"])
    parser.add_argument("--fusion-hidden-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=MODEL_SETTING["dropout"])
    parser.add_argument("--learning-rate", type=float, default=MODEL_SETTING["learning_rate"])
    parser.add_argument("--epochs", type=int, default=MODEL_SETTING["epochs"])
    parser.add_argument("--batch-size", type=int, default=MODEL_SETTING["batch_size"])
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def parse_csv_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one comma-separated value.")
    return items


def parse_int_list(value: str) -> list[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one integer.")
    if any(item <= 0 for item in items):
        raise ValueError("Integer list values must be positive.")
    return items


def table_columns(path: Path) -> list[str]:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return list(pq.read_schema(path).names)
    return list(pd.read_csv(path, nrows=0).columns)


def split_matches(series: pd.Series, split: str) -> pd.Series:
    aliases = SPLIT_ALIASES.get(split, {split})
    return series.astype(str).str.strip().isin(aliases)


def table_split_values(path: Path, chunksize: int) -> set[str]:
    values: set[str] = set()
    if path.suffix.lower() in {".parquet", ".pq"}:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=chunksize, columns=["split"]):
            frame = batch.to_pandas()
            values.update(frame["split"].astype(str).unique())
        return values
    for chunk in pd.read_csv(path, usecols=["split"], chunksize=chunksize):
        values.update(chunk["split"].astype(str).unique())
    return values


def metric_column(embedding: str, metric: str, columns: set[str]) -> str | None:
    wide = f"score__{embedding}__{metric}"
    if wide in columns:
        return wide
    compat = f"score_{embedding}"
    if metric == "cosine" and compat in columns:
        return compat
    return None


def available_embeddings(columns: list[str]) -> list[str]:
    column_set = set(columns)
    present = []
    for embedding in EMBEDDINGS:
        if f"score_{embedding}" in column_set or any(column.startswith(f"score__{embedding}__") for column in columns):
            present.append(embedding)
    extras = sorted(
        {
            column.split("__", 2)[1]
            for column in columns
            if column.startswith("score__") and len(column.split("__", 2)) == 3
        }
        - set(present)
    )
    return present + extras


def available_metrics(columns: list[str], embeddings: list[str]) -> list[str]:
    column_set = set(columns)
    metrics = set()
    for embedding in embeddings:
        if f"score_{embedding}" in column_set:
            metrics.add("cosine")
        prefix = f"score__{embedding}__"
        for column in columns:
            if column.startswith(prefix):
                metrics.add(column[len(prefix) :])
    ordered = [metric for metric in SCORE_SPECS if metric in metrics]
    ordered.extend(sorted(metrics - set(ordered)))
    return ordered


def resolve_embeddings(requested: list[str] | None, columns: list[str]) -> list[str]:
    present = available_embeddings(columns)
    if requested is None:
        if not present:
            raise ValueError("No score columns found for any embedding.")
        return present
    missing = sorted(set(requested) - set(present))
    if missing:
        raise ValueError(f"Requested embeddings are missing from the table: {missing}")
    return requested


def resolve_metrics(requested: list[str] | None, columns: list[str], embeddings: list[str]) -> list[str]:
    present = available_metrics(columns, embeddings)
    if requested is None:
        if not present:
            raise ValueError("No similarity metric columns found.")
        return present
    unknown = sorted(set(requested) - set(SCORE_SPECS))
    if unknown:
        raise ValueError(f"Unknown metric(s): {unknown}")
    missing = []
    column_set = set(columns)
    for embedding in embeddings:
        for metric in requested:
            if metric_column(embedding, metric, column_set) is None:
                missing.append(f"{embedding}:{metric}")
    if missing:
        raise ValueError(f"Requested embedding/metric columns are missing: {missing}")
    return requested


def metadata_columns(mode: str, columns: list[str]) -> list[str]:
    requested = METADATA_COLUMNS[mode]
    missing = [column for column in requested if column not in columns]
    if missing:
        raise ValueError(f"Metadata mode {mode!r} requires missing columns: {missing}")
    return requested


def score_columns(embeddings: list[str], metrics: list[str], columns: list[str]) -> list[str]:
    column_set = set(columns)
    resolved = []
    for embedding in embeddings:
        for metric in metrics:
            column = metric_column(embedding, metric, column_set)
            if column is None:
                raise ValueError(f"Missing score column for embedding={embedding}, metric={metric}")
            resolved.append(column)
    return resolved


def apply_rank_limit(frame: pd.DataFrame, split: str, seed: int, max_rows_per_class: int) -> pd.DataFrame:
    frame = frame.copy()
    ranks = pd.util.hash_pandas_object(frame["trial_id"].astype(str) + f"|{split}|{seed}", index=False)
    frame["_sample_rank"] = ranks.to_numpy(dtype=np.uint64, copy=False)
    parts = []
    for label in (0, 1):
        part = frame[frame["label"] == label].sort_values("_sample_rank", kind="mergesort").head(max_rows_per_class)
        if not part.empty:
            parts.append(part)
    if not parts:
        return frame.iloc[0:0].drop(columns=["_sample_rank"])
    return pd.concat(parts, ignore_index=True).drop(columns=["_sample_rank"])


def merge_limited(existing: dict[int, pd.DataFrame], chunk: pd.DataFrame, split: str, seed: int, max_rows_per_class: int) -> None:
    ranked = apply_rank_limit(chunk, split, seed, max_rows_per_class)
    for label in (0, 1):
        part = ranked[ranked["label"] == label]
        if part.empty:
            continue
        current = existing.get(label)
        combined = part if current is None else pd.concat([current, part], ignore_index=True)
        existing[label] = apply_rank_limit(combined, split, seed, max_rows_per_class)


def iter_table_chunks(path: Path, columns: list[str], chunksize: int) -> Iterable[pd.DataFrame]:
    if path.suffix.lower() in {".parquet", ".pq"}:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=chunksize, columns=columns):
            yield batch.to_pandas()
        return
    yield from pd.read_csv(path, usecols=columns, chunksize=chunksize)


def compact_loaded_frame(frame: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [column for column in ("split", "trial_id", "_sample_rank") if column in frame.columns]
    if drop_cols:
        frame = frame.drop(columns=drop_cols)
    if "label" in frame.columns:
        frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(np.int8, copy=False)
    feature_cols = [column for column in frame.columns if column != "label"]
    for column in feature_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(np.float32, copy=False)
    return frame


def load_split_rows(
    path: Path,
    *,
    split: str,
    columns: list[str],
    max_rows_per_class: int | None,
    seed: int,
    chunksize: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    limited: dict[int, pd.DataFrame] = {}
    for chunk in iter_table_chunks(path, columns, chunksize):
        chunk = chunk[split_matches(chunk["split"], split)]
        if chunk.empty:
            continue
        if max_rows_per_class is None:
            frames.append(compact_loaded_frame(chunk))
        else:
            merge_limited(limited, chunk, split, seed, max_rows_per_class)
    if max_rows_per_class is not None:
        frames = [compact_loaded_frame(limited[label]) for label in (0, 1) if label in limited]
    if not frames:
        raise ValueError(f"No rows found for split={split}")
    frame = pd.concat(frames, ignore_index=True)
    labels = set(pd.to_numeric(frame["label"], errors="coerce").dropna().astype(int).unique())
    if not {0, 1}.issubset(labels):
        raise ValueError(f"Split {split!r} must contain both label classes; found {sorted(labels)}")
    return frame.reset_index(drop=True)


def select_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable.")
        return torch.device("cuda")
    if name == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_head(input_dim: int, hidden_dim: int, dropout: float) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.LayerNorm(hidden_dim),
        torch.nn.SiLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.LayerNorm(hidden_dim),
        torch.nn.SiLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(hidden_dim, 1),
    )


def build_fusion(input_dim: int, hidden_dim: int, output_dim: int) -> torch.nn.Sequential:
    return torch.nn.Sequential(
        torch.nn.LayerNorm(input_dim),
        torch.nn.Linear(input_dim, hidden_dim),
        torch.nn.SiLU(),
        torch.nn.Linear(hidden_dim, output_dim),
    )


class FusionMlpNet(torch.nn.Module):
    def __init__(
        self,
        *,
        embedding_count: int,
        metric_count: int,
        other_dim: int,
        fusion_dim: int,
        fusion_hidden_dim: int,
        hidden_dim: int,
        dropout: float,
        shared_fusion: bool,
    ) -> None:
        super().__init__()
        self.embedding_count = embedding_count
        self.fusion_dim = fusion_dim
        self.shared_fusion = shared_fusion
        if shared_fusion:
            self.shared = build_fusion(metric_count, fusion_hidden_dim, fusion_dim)
            self.fusions = None
        else:
            self.shared = None
            self.fusions = torch.nn.ModuleList(
                [build_fusion(metric_count, fusion_hidden_dim, fusion_dim) for _ in range(embedding_count)]
            )
        self.head = build_head(embedding_count * fusion_dim + other_dim, hidden_dim, dropout)

    def forward(self, scores: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
        if self.shared_fusion:
            fused = self.shared(scores.reshape(-1, scores.shape[-1])).reshape(scores.shape[0], self.embedding_count, self.fusion_dim)
        else:
            if self.fusions is None:
                raise RuntimeError("Independent fusion layers are not initialized.")
            fused = torch.stack([layer(scores[:, idx, :]) for idx, layer in enumerate(self.fusions)], dim=1)
        fused_flat = fused.reshape(scores.shape[0], self.embedding_count * self.fusion_dim)
        if other.shape[1] > 0:
            fused_flat = torch.cat([fused_flat, other], dim=1)
        return self.head(fused_flat).squeeze(1)


class FusionMlpCalibrationModel:
    def __init__(
        self,
        *,
        embedding_count: int,
        metric_count: int,
        other_dim: int,
        fusion_dim: int,
        fusion_hidden_dim: int,
        hidden_dim: int,
        dropout: float,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        c_value: float,
        seed: int,
        shared_fusion: bool,
        device: torch.device,
    ) -> None:
        self.embedding_count = embedding_count
        self.metric_count = metric_count
        self.other_dim = other_dim
        self.fusion_dim = fusion_dim
        self.fusion_hidden_dim = fusion_hidden_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.c_value = c_value
        self.seed = seed
        self.shared_fusion = shared_fusion
        self.device = device
        self.score_mean: torch.Tensor | None = None
        self.score_std: torch.Tensor | None = None
        self.other_mean: torch.Tensor | None = None
        self.other_std: torch.Tensor | None = None
        self.net: FusionMlpNet | None = None

    def fit(self, scores: np.ndarray, other: np.ndarray, labels: np.ndarray, sample_weights: np.ndarray) -> None:
        torch.manual_seed(self.seed)
        score_tensor = torch.as_tensor(scores, dtype=torch.float32, device=self.device)
        other_tensor = torch.as_tensor(other, dtype=torch.float32, device=self.device)
        y_tensor = torch.as_tensor(labels.astype(np.float32), dtype=torch.float32, device=self.device)
        w_tensor = torch.as_tensor(sample_weights.astype(np.float32), dtype=torch.float32, device=self.device)

        self.score_mean = score_tensor.mean(dim=0, keepdim=True)
        self.score_std = score_tensor.std(dim=0, keepdim=True, unbiased=False).clamp_min(1.0e-6)
        score_scaled = (score_tensor - self.score_mean) / self.score_std
        if other_tensor.shape[1] > 0:
            self.other_mean = other_tensor.mean(dim=0, keepdim=True)
            self.other_std = other_tensor.std(dim=0, keepdim=True, unbiased=False).clamp_min(1.0e-6)
            other_scaled = (other_tensor - self.other_mean) / self.other_std
        else:
            self.other_mean = torch.empty((1, 0), dtype=torch.float32, device=self.device)
            self.other_std = torch.empty((1, 0), dtype=torch.float32, device=self.device)
            other_scaled = other_tensor

        self.net = FusionMlpNet(
            embedding_count=self.embedding_count,
            metric_count=self.metric_count,
            other_dim=self.other_dim,
            fusion_dim=self.fusion_dim,
            fusion_hidden_dim=self.fusion_hidden_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            shared_fusion=self.shared_fusion,
        ).to(self.device)
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.learning_rate,
            weight_decay=1.0 / max(self.c_value, 1.0e-12),
        )

        n_rows = score_scaled.shape[0]
        batch_size = min(self.batch_size, n_rows)
        for _ in range(self.epochs):
            order = torch.randperm(n_rows, device=self.device)
            self.net.train()
            for start in range(0, n_rows, batch_size):
                idx = order[start : start + batch_size]
                logits = self.net(score_scaled[idx], other_scaled[idx])
                loss_terms = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_tensor[idx], reduction="none")
                weights = w_tensor[idx]
                loss = torch.sum(weights * loss_terms) / torch.sum(weights).clamp_min(1.0e-12)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        self.net.eval()

    def decision_function(self, scores: np.ndarray, other: np.ndarray, batch_size: int = 1_000_000) -> np.ndarray:
        if self.score_mean is None or self.score_std is None or self.other_mean is None or self.other_std is None or self.net is None:
            raise RuntimeError("Model is not fitted")
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, scores.shape[0], batch_size):
                score_batch = torch.as_tensor(scores[start : start + batch_size], dtype=torch.float32, device=self.device)
                other_batch = torch.as_tensor(other[start : start + batch_size], dtype=torch.float32, device=self.device)
                score_batch = (score_batch - self.score_mean) / self.score_std
                if other_batch.shape[1] > 0:
                    other_batch = (other_batch - self.other_mean) / self.other_std
                logits = self.net(score_batch, other_batch)
                outputs.append(logits.detach().cpu().numpy().astype(np.float64, copy=False))
        return np.concatenate(outputs)

    @property
    def parameter_count(self) -> int:
        if self.net is None:
            raise RuntimeError("Model is not fitted")
        return int(sum(parameter.numel() for parameter in self.net.parameters() if parameter.requires_grad))


def concat_matrix(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    if not columns:
        return np.empty((frame.shape[0], 0), dtype=np.float32)
    return frame[columns].to_numpy(dtype=np.float32, copy=False)


def score_blocks(frame: pd.DataFrame, embeddings: list[str], metrics: list[str], columns: list[str]) -> np.ndarray:
    all_columns = score_columns(embeddings, metrics, columns)
    flat = frame[all_columns].to_numpy(dtype=np.float32, copy=False)
    return flat.reshape(frame.shape[0], len(embeddings), len(metrics))


def finite_concat_mask(values: np.ndarray) -> np.ndarray:
    if values.shape[1] == 0:
        return np.ones(values.shape[0], dtype=bool)
    return np.isfinite(values).all(axis=1)


def finite_fusion_mask(scores: np.ndarray, other: np.ndarray) -> np.ndarray:
    score_mask = np.isfinite(scores).all(axis=(1, 2))
    other_mask = finite_concat_mask(other)
    return score_mask & other_mask


def write_rows(path: Path, rows: list[dict[str, object]], *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"Output already exists: {path}. Use --overwrite to replace it.")
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


def base_result_row(
    *,
    mode: str,
    seed: int,
    model_name: str,
    train_split: str,
    eval_split: str,
    embeddings: list[str],
    metrics: list[str],
    metadata_mode: str,
    metadata_cols: list[str],
    input_feature_count: int,
    model_input_dim: int,
    parameter_count: int,
    dropped_train: int,
    dropped_eval: int,
    llrs: np.ndarray,
    labels: np.ndarray,
    train_llrs: np.ndarray,
    train_labels: np.ndarray,
) -> dict[str, object]:
    threshold = threshold_for_max_accuracy(train_llrs, train_labels)
    row: dict[str, object] = {
        "system": "TIDYVOICE-SIM-MLP",
        "split": eval_split,
        "embedding": "multi_embedding_fusion",
        "feature_mode": mode,
        "model": model_name,
        "seed": seed,
        "fit_split": train_split,
        "evaluation_split": eval_split,
        "embedding_count": len(embeddings),
        "embeddings": "|".join(embeddings),
        "score_metric_count": len(metrics),
        "score_metrics": "|".join(metrics),
        "metadata_mode": metadata_mode,
        "metadata_feature_count": len(metadata_cols),
        "metadata_columns": "|".join(metadata_cols),
        "input_feature_count": input_feature_count,
        "model_input_dim": model_input_dim,
        "parameter_count": parameter_count,
        "hidden_dim": MODEL_SETTING["hidden_dim"],
        "dropout": MODEL_SETTING["dropout"],
        "learning_rate": MODEL_SETTING["learning_rate"],
        "epochs": MODEL_SETTING["epochs"],
        "batch_size": MODEL_SETTING["batch_size"],
        "C": C_VALUE,
        "coverage": 1.0,
        "dropped_nonfinite_train_rows": dropped_train,
        "dropped_nonfinite_eval_rows": dropped_eval,
    }
    row.update(metric_block(llrs, llrs, labels))
    row.update({f"threshold_{key}": value for key, value in threshold_metrics(llrs, labels, threshold).items()})
    return row


def run_concat_mode(
    *,
    train: pd.DataFrame,
    eval_frame: pd.DataFrame,
    feature_cols: list[str],
    mode: str,
    seed: int,
    device: torch.device,
    train_split: str,
    eval_split: str,
    embeddings: list[str],
    metrics: list[str],
    metadata_mode: str,
    metadata_cols: list[str],
) -> dict[str, object]:
    from common import Experiment, make_model

    train_x = concat_matrix(train, feature_cols)
    eval_x = concat_matrix(eval_frame, feature_cols)
    train_y = train["label"].to_numpy(dtype=np.int8, copy=False)
    eval_y = eval_frame["label"].to_numpy(dtype=np.int8, copy=False)
    train_mask = finite_concat_mask(train_x)
    eval_mask = finite_concat_mask(eval_x)
    train_x_used = train_x[train_mask]
    eval_x_used = eval_x[eval_mask]
    train_y_used = train_y[train_mask]
    eval_y_used = eval_y[eval_mask]
    experiment_setting = {
        key: MODEL_SETTING[key]
        for key in ("hidden_dim", "dropout", "learning_rate", "epochs", "batch_size")
    }
    experiment = Experiment("SIM-MLP", mode, C_VALUE, seed=seed, **experiment_setting)
    model = make_model(experiment, device)
    model.fit(train_x_used, train_y_used, balanced_sample_weights(train_y_used), desc=f"train {mode} seed={seed}")
    train_llrs = model.decision_function(train_x_used)
    llrs = model.decision_function(eval_x_used)
    parameter_count = trainable_parameter_count(train_x_used.shape[1], MODEL_SETTING["hidden_dim"])
    return base_result_row(
        mode=mode,
        seed=seed,
        model_name="mlp_concat",
        train_split=train_split,
        eval_split=eval_split,
        embeddings=embeddings,
        metrics=metrics,
        metadata_mode=metadata_mode,
        metadata_cols=metadata_cols,
        input_feature_count=len(feature_cols),
        model_input_dim=train_x_used.shape[1],
        parameter_count=parameter_count,
        dropped_train=int((~train_mask).sum()),
        dropped_eval=int((~eval_mask).sum()),
        llrs=llrs,
        labels=eval_y_used,
        train_llrs=train_llrs,
        train_labels=train_y_used,
    )


def run_fusion_mode(
    *,
    train: pd.DataFrame,
    eval_frame: pd.DataFrame,
    embeddings: list[str],
    metrics: list[str],
    table_cols: list[str],
    metadata_cols: list[str],
    fusion_dim: int,
    seed: int,
    device: torch.device,
    train_split: str,
    eval_split: str,
    metadata_mode: str,
    shared_fusion: bool,
) -> dict[str, object]:
    train_scores = score_blocks(train, embeddings, metrics, table_cols)
    eval_scores = score_blocks(eval_frame, embeddings, metrics, table_cols)
    train_other = concat_matrix(train, metadata_cols)
    eval_other = concat_matrix(eval_frame, metadata_cols)
    train_y = train["label"].to_numpy(dtype=np.int8, copy=False)
    eval_y = eval_frame["label"].to_numpy(dtype=np.int8, copy=False)
    train_mask = finite_fusion_mask(train_scores, train_other)
    eval_mask = finite_fusion_mask(eval_scores, eval_other)
    train_scores_used = train_scores[train_mask]
    eval_scores_used = eval_scores[eval_mask]
    train_other_used = train_other[train_mask]
    eval_other_used = eval_other[eval_mask]
    train_y_used = train_y[train_mask]
    eval_y_used = eval_y[eval_mask]
    model = FusionMlpCalibrationModel(
        embedding_count=len(embeddings),
        metric_count=len(metrics),
        other_dim=len(metadata_cols),
        fusion_dim=fusion_dim,
        fusion_hidden_dim=int(MODEL_SETTING["fusion_hidden_dim"]),
        hidden_dim=int(MODEL_SETTING["hidden_dim"]),
        dropout=float(MODEL_SETTING["dropout"]),
        learning_rate=float(MODEL_SETTING["learning_rate"]),
        epochs=int(MODEL_SETTING["epochs"]),
        batch_size=int(MODEL_SETTING["batch_size"]),
        c_value=C_VALUE,
        seed=seed,
        shared_fusion=shared_fusion,
        device=device,
    )
    model.fit(train_scores_used, train_other_used, train_y_used, balanced_sample_weights(train_y_used))
    train_llrs = model.decision_function(train_scores_used, train_other_used)
    llrs = model.decision_function(eval_scores_used, eval_other_used)
    mode = f"similarity_fusion_{fusion_dim}d"
    return base_result_row(
        mode=mode,
        seed=seed,
        model_name="mlp_fusion_shared" if shared_fusion else "mlp_fusion_independent",
        train_split=train_split,
        eval_split=eval_split,
        embeddings=embeddings,
        metrics=metrics,
        metadata_mode=metadata_mode,
        metadata_cols=metadata_cols,
        input_feature_count=len(embeddings) * len(metrics) + len(metadata_cols),
        model_input_dim=len(embeddings) * fusion_dim + len(metadata_cols),
        parameter_count=model.parameter_count,
        dropped_train=int((~train_mask).sum()),
        dropped_eval=int((~eval_mask).sum()),
        llrs=llrs,
        labels=eval_y_used,
        train_llrs=train_llrs,
        train_labels=train_y_used,
    ) | {
        "fusion_dim": fusion_dim,
        "fusion_hidden_dim": MODEL_SETTING["fusion_hidden_dim"],
        "shared_fusion": shared_fusion,
    }


def apply_runtime_settings(args: argparse.Namespace) -> None:
    MODEL_SETTING["hidden_dim"] = args.hidden_dim
    MODEL_SETTING["fusion_hidden_dim"] = args.fusion_hidden_dim
    MODEL_SETTING["dropout"] = args.dropout
    MODEL_SETTING["learning_rate"] = args.learning_rate
    MODEL_SETTING["epochs"] = args.epochs
    MODEL_SETTING["batch_size"] = args.batch_size


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    apply_runtime_settings(args)
    input_table = args.input_table.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not input_table.exists():
        raise SystemExit(f"Input table does not exist: {input_table}")
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output}. Use --overwrite to replace it.")

    columns = table_columns(input_table)
    split_values = table_split_values(input_table, args.chunksize)
    train_aliases = SPLIT_ALIASES.get(args.train_split, {args.train_split})
    eval_aliases = SPLIT_ALIASES.get(args.eval_split, {args.eval_split})
    if split_values.isdisjoint(train_aliases):
        raise SystemExit(f"Train split {args.train_split!r} is not present in {input_table}. Available splits: {sorted(split_values)}")
    if split_values.isdisjoint(eval_aliases):
        raise SystemExit(
            f"Evaluation split {args.eval_split!r} is not present in {input_table}. "
            f"Available splits: {sorted(split_values)}. Pass --eval-split test explicitly for final-test evaluation."
        )

    embeddings = resolve_embeddings(parse_csv_list(args.embeddings), columns)
    metrics = resolve_metrics(parse_csv_list(args.metrics), columns, embeddings)
    metadata_cols = metadata_columns(args.metadata_mode, columns)
    all_score_cols = score_columns(embeddings, metrics, columns)
    cosine_cols = score_columns(embeddings, ["cosine"], columns)
    concat_cols = all_score_cols + metadata_cols
    required_cols = ["split", "trial_id", "label", *concat_cols]
    if args.include_cosine_reference:
        required_cols.extend(cosine_cols)
    required_cols = list(dict.fromkeys(required_cols))

    seed_values = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if not seed_values:
        raise SystemExit("At least one seed is required.")
    fusion_dims = parse_int_list(args.fusion_dims)
    device = select_device(args.device)

    train = load_split_rows(
        input_table,
        split=args.train_split,
        columns=required_cols,
        max_rows_per_class=args.max_rows_per_class,
        seed=seed_values[0],
        chunksize=args.chunksize,
    )
    print(f"loaded {args.train_split} rows={train.shape[0]}", flush=True)
    eval_frame = load_split_rows(
        input_table,
        split=args.eval_split,
        columns=required_cols,
        max_rows_per_class=args.max_rows_per_class,
        seed=seed_values[0],
        chunksize=args.chunksize,
    )
    print(f"loaded {args.eval_split} rows={eval_frame.shape[0]}", flush=True)
    print(
        f"loaded train={train.shape[0]} eval={eval_frame.shape[0]} "
        f"embeddings={len(embeddings)} metrics={len(metrics)} metadata={args.metadata_mode} device={device}",
        flush=True,
    )

    rows: list[dict[str, object]] = []
    for seed in seed_values:
        if args.include_cosine_reference:
            rows.append(
                run_concat_mode(
                    train=train,
                    eval_frame=eval_frame,
                    feature_cols=cosine_cols + metadata_cols,
                    mode="similarity_cosine_reference",
                    seed=seed,
                    device=device,
                    train_split=args.train_split,
                    eval_split=args.eval_split,
                    embeddings=embeddings,
                    metrics=["cosine"],
                    metadata_mode=args.metadata_mode,
                    metadata_cols=metadata_cols,
                )
            )
        rows.append(
            run_concat_mode(
                train=train,
                eval_frame=eval_frame,
                feature_cols=concat_cols,
                mode="similarity_concat_all",
                seed=seed,
                device=device,
                train_split=args.train_split,
                eval_split=args.eval_split,
                embeddings=embeddings,
                metrics=metrics,
                metadata_mode=args.metadata_mode,
                metadata_cols=metadata_cols,
            )
        )
        for fusion_dim in fusion_dims:
            rows.append(
                run_fusion_mode(
                    train=train,
                    eval_frame=eval_frame,
                    embeddings=embeddings,
                    metrics=metrics,
                    table_cols=columns,
                    metadata_cols=metadata_cols,
                    fusion_dim=fusion_dim,
                    seed=seed,
                    device=device,
                    train_split=args.train_split,
                    eval_split=args.eval_split,
                    metadata_mode=args.metadata_mode,
                    shared_fusion=args.shared_fusion,
                )
            )

    write_rows(output, rows, overwrite=args.overwrite)
    repro = output.with_suffix(".reproduction.sh")
    max_rows_arg = "" if args.max_rows_per_class is None else f" --max-rows-per-class {args.max_rows_per_class}"
    shared_arg = " --shared-fusion" if args.shared_fusion else ""
    cosine_ref_arg = " --include-cosine-reference" if args.include_cosine_reference else ""
    repro.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"python scripts/run_tidyvoice_similarity_mlp_fusion.py --input-table {input_table} --output {output} "
        f"--train-split {args.train_split} --eval-split {args.eval_split} --metadata-mode {args.metadata_mode} "
        f"--metrics {','.join(metrics)} --embeddings {','.join(embeddings)} --fusion-dims {','.join(map(str, fusion_dims))} "
        f"--seeds {args.seeds} --device {args.device} --hidden-dim {args.hidden_dim} --fusion-hidden-dim {args.fusion_hidden_dim} "
        f"--dropout {args.dropout} --learning-rate {args.learning_rate} --epochs {args.epochs} --batch-size {args.batch_size}"
        f"{max_rows_arg}{shared_arg}{cosine_ref_arg} --overwrite\n",
        encoding="utf-8",
    )
    print(f"wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
