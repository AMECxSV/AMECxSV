from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

try:
    import torch
except ImportError:
    torch = None

from selective import (
    confidence_from_llr,
    confidence_threshold,
    decision_metrics,
    decisions_from_confidence,
)
from c8 import threshold_for_max_accuracy, threshold_metrics
from rich_mlp import validation_cllr
from common import (
    BASELINE_DIR,
    C5_COLS,
    EMBEDDINGS,
    PROJECT_ROOT,
    affine_llr_calibration,
    balanced_sample_weights,
    metric_block,
)


DEFAULT_INPUT = Path(
    os.environ.get(
        "AMECXSV_SIMILARITY_TABLE",
        PROJECT_ROOT / "dataset" / "tidyvoice_similarity_scores.parquet",
    )
)
OUTPUT_CSV = BASELINE_DIR / "tidyvoice_c12_mlp_results.csv"
OUTPUT_SCORES = BASELINE_DIR / "tidyvoice_c12_test_scores.parquet"
HISTORY_CSV = BASELINE_DIR / "tidyvoice_c12_training_history.csv"
DATASET_NAME = "tidyvoice_backend_similarity_scores_full"
NUM_ENCODERS = 6
SCORE_VARIANTS = (
    "cosine",
    "centered_cosine",
    "whitened_cosine",
    "wccn_cosine",
    "lda_cosine",
    "neg_mahalanobis",
    "asnorm_cosine",
)
SCORE_COLS = [f"score__{embedding}__{variant}" for embedding in EMBEDDINGS for variant in SCORE_VARIANTS]
PRIMARY_SCORE_COLS = [f"score_{embedding}" for embedding in EMBEDDINGS]
PRIMARY_SCORE_IDXS = [idx * len(SCORE_VARIANTS) for idx in range(len(EMBEDDINGS))]
META_COLS = [*C5_COLS, "target"]
FEATURE_COLS = [*SCORE_COLS, *META_COLS]
BASE_COLS = ["split", "trial_id", "label"]
SPLIT_HASH_MOD = 10_000


@dataclass
class SplitArrays:
    features: np.ndarray
    labels: np.ndarray
    primary_scores: np.ndarray
    trial_ids: np.ndarray | None = None


@dataclass
class C12Dataset:
    train: SplitArrays
    validation: SplitArrays
    calibration: SplitArrays
    test: SplitArrays


class ResidualBlock(torch.nn.Module if torch is not None else object):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Linear(width, width),
            torch.nn.LayerNorm(width),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(width, width),
            torch.nn.Dropout(dropout),
        )
        self.norm = torch.nn.LayerNorm(width)
        self.activation = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(x + self.block(x)))


class C12GatedFusionNet(torch.nn.Module if torch is not None else object):
    def __init__(
        self,
        *,
        score_variants: int,
        num_encoders: int,
        meta_dim: int,
        latent_dim: int,
        head_dim: int,
        residual_blocks: int,
        dropout: float,
        max_fusion_residual_scale: float,
    ) -> None:
        super().__init__()
        self.score_variants = score_variants
        self.num_encoders = num_encoders
        self.meta_dim = meta_dim

        self.score_encoder = torch.nn.Sequential(
            torch.nn.Linear(score_variants, latent_dim),
            torch.nn.LayerNorm(latent_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(latent_dim, latent_dim),
            torch.nn.LayerNorm(latent_dim),
            torch.nn.ReLU(),
        )
        self.meta_encoder = torch.nn.Sequential(
            torch.nn.Linear(meta_dim, latent_dim),
            torch.nn.LayerNorm(latent_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(latent_dim, latent_dim),
            torch.nn.LayerNorm(latent_dim),
            torch.nn.ReLU(),
        )
        self.gate = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, latent_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(latent_dim, num_encoders),
        )

        stats_dim = 6
        head_input_dim = latent_dim + latent_dim + num_encoders * score_variants + meta_dim + stats_dim
        self.input_projection = torch.nn.Sequential(
            torch.nn.Linear(head_input_dim, head_dim),
            torch.nn.LayerNorm(head_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )
        self.blocks = torch.nn.Sequential(*[ResidualBlock(head_dim, dropout) for _ in range(residual_blocks)])
        self.output = torch.nn.Linear(head_dim, 1)
        self.alpha = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32))
        primary_stats_dim = 6
        primary_backbone_dim = num_encoders + meta_dim + primary_stats_dim
        self.primary_backbone = torch.nn.Sequential(
            torch.nn.Linear(primary_backbone_dim, head_dim),
            torch.nn.LayerNorm(head_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            ResidualBlock(head_dim, dropout),
            torch.nn.Linear(head_dim, 1),
        )
        self.primary_score_skip = torch.nn.Linear(num_encoders, 1)
        with torch.no_grad():
            self.primary_score_skip.weight.fill_(1.0 / num_encoders)
            self.primary_score_skip.bias.zero_()
        self.max_fusion_residual_scale = max_fusion_residual_scale
        self.fusion_residual_logit_scale = torch.nn.Parameter(torch.tensor(-1.8718022, dtype=torch.float32))

    def score_stats(self, scores: torch.Tensor) -> torch.Tensor:
        sorted_scores = torch.sort(scores, dim=1, descending=True).values
        top2_gap = sorted_scores[:, 0:1] - sorted_scores[:, 1:2]
        return torch.cat(
            [
                scores.mean(dim=1, keepdim=True),
                scores.std(dim=1, unbiased=False, keepdim=True),
                scores.max(dim=1, keepdim=True).values,
                scores.min(dim=1, keepdim=True).values,
                scores.median(dim=1, keepdim=True).values,
                top2_gap,
            ],
            dim=1,
        )

    def forward(self, features: torch.Tensor, primary_scores: torch.Tensor) -> torch.Tensor:
        score_dim = self.num_encoders * self.score_variants
        scores = features[:, :score_dim]
        meta = features[:, score_dim : score_dim + self.meta_dim]

        score_tokens = scores.reshape(-1, self.num_encoders, self.score_variants)
        encoded_scores = self.score_encoder(score_tokens.reshape(-1, self.score_variants))
        encoded_scores = encoded_scores.reshape(scores.shape[0], self.num_encoders, -1)

        meta_feature = self.meta_encoder(meta)
        gate_logits = self.gate(meta_feature)
        gate_weights = torch.softmax(gate_logits, dim=1)
        fused = torch.sum(encoded_scores * gate_weights.unsqueeze(-1), dim=1)

        residual_features = torch.cat(
            [
                fused,
                meta_feature,
                scores,
                meta,
                self.score_stats(scores),
            ],
            dim=1,
        )
        hidden = self.input_projection(residual_features)
        hidden = self.blocks(hidden)
        fusion_residual = torch.tanh(self.output(hidden).squeeze(1))
        primary_features = torch.cat(
            [
                primary_scores,
                meta,
                self.score_stats(primary_scores),
            ],
            dim=1,
        )
        primary_logit = self.primary_score_skip(primary_scores).squeeze(1)
        primary_logit = primary_logit + self.primary_backbone(primary_features).squeeze(1)
        raw_mean_score = primary_scores.mean(dim=1)
        residual_scale = self.max_fusion_residual_scale * torch.sigmoid(self.fusion_residual_logit_scale)
        return primary_logit + self.alpha * raw_mean_score + residual_scale * fusion_residual


class C12GatedFusionCalibrator:
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        residual_blocks: int,
        dropout: float,
        learning_rate: float,
        c_value: float,
        epochs: int,
        batch_size: int,
        device: torch.device,
        early_stopping_patience: int,
        scheduler_patience: int,
        max_fusion_residual_scale: float,
        min_delta: float = 1.0e-5,
    ) -> None:
        if torch is None:
            raise RuntimeError("torch is required for C12GatedFusionCalibrator")
        expected_dim = len(SCORE_COLS) + len(META_COLS)
        if input_dim != expected_dim:
            raise ValueError(f"C12 expects {expected_dim} features, got {input_dim}")
        self.name = (
            f"c12_primary_backbone_gated_residual_{input_dim}d_lat{latent_dim}_h{hidden_dim}_"
            f"blocks{residual_blocks}_drop{dropout:g}_maxres{max_fusion_residual_scale:g}"
        )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.residual_blocks = residual_blocks
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.c_value = c_value
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.early_stopping_patience = early_stopping_patience
        self.scheduler_patience = scheduler_patience
        self.max_fusion_residual_scale = max_fusion_residual_scale
        self.min_delta = min_delta
        self.seed = 0
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        self.primary_mean: torch.Tensor | None = None
        self.primary_std: torch.Tensor | None = None
        self.net: C12GatedFusionNet | None = None
        self.best_epoch: int | None = None
        self.best_val_cllr: float | None = None
        self.trained_epochs = 0
        self.stopped_early = False
        self.epoch_history: list[dict[str, float | int | bool]] = []

    def build_net(self) -> C12GatedFusionNet:
        return C12GatedFusionNet(
            score_variants=len(SCORE_VARIANTS),
            num_encoders=NUM_ENCODERS,
            meta_dim=len(META_COLS),
            latent_dim=self.latent_dim,
            head_dim=self.hidden_dim,
            residual_blocks=self.residual_blocks,
            dropout=self.dropout,
            max_fusion_residual_scale=self.max_fusion_residual_scale,
        )

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sample_weights: np.ndarray,
        *,
        primary_scores: np.ndarray,
        validation_features: np.ndarray,
        validation_primary_scores: np.ndarray,
        validation_labels: np.ndarray,
    ) -> None:
        torch.manual_seed(self.seed)
        x_tensor = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        primary_tensor = torch.as_tensor(primary_scores, dtype=torch.float32, device=self.device)
        y_tensor = torch.as_tensor(labels.astype(np.float32), dtype=torch.float32, device=self.device)
        w_tensor = torch.as_tensor(sample_weights.astype(np.float32), dtype=torch.float32, device=self.device)
        validation_tensor = torch.as_tensor(validation_features, dtype=torch.float32, device=self.device)
        validation_primary_tensor = torch.as_tensor(
            validation_primary_scores,
            dtype=torch.float32,
            device=self.device,
        )

        self.mean = x_tensor.mean(dim=0, keepdim=True)
        self.std = x_tensor.std(dim=0, keepdim=True, unbiased=False).clamp_min(1.0e-6)
        self.primary_mean = primary_tensor.mean(dim=0, keepdim=True)
        self.primary_std = primary_tensor.std(dim=0, keepdim=True, unbiased=False).clamp_min(1.0e-6)
        primary_scaled = (primary_tensor - self.primary_mean) / self.primary_std

        self.net = self.build_net().to(self.device)
        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.learning_rate,
            weight_decay=1.0 / max(self.c_value, 1.0e-12),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=self.scheduler_patience,
            threshold=self.min_delta,
            min_lr=1.0e-6,
        )

        n_rows = x_tensor.shape[0]
        batch_size = min(self.batch_size, n_rows)
        best_state = None
        best_cllr = math.inf
        best_epoch = -1
        epochs_without_improvement = 0
        self.epoch_history = []
        self.stopped_early = False
        self.trained_epochs = 0

        for epoch in tqdm(range(self.epochs), desc=f"train {self.name}", unit="epoch"):
            order = torch.randperm(n_rows, device=self.device)
            self.net.train()
            for start in range(0, n_rows, batch_size):
                idx = order[start : start + batch_size]
                batch = (x_tensor[idx] - self.mean) / self.std
                logits = self.net(batch, primary_scaled[idx])
                loss_terms = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits,
                    y_tensor[idx],
                    reduction="none",
                )
                weights = w_tensor[idx]
                loss = torch.sum(weights * loss_terms) / torch.sum(weights).clamp_min(1.0e-12)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
                optimizer.step()

            validation_llrs = self._predict_tensor(validation_tensor, validation_primary_tensor)
            current_cllr = validation_cllr(validation_llrs, validation_labels)
            previous_lr = float(optimizer.param_groups[0]["lr"])
            scheduler.step(current_cllr)
            current_lr = float(optimizer.param_groups[0]["lr"])
            improved = current_cllr < best_cllr - self.min_delta
            if current_lr < previous_lr:
                print(
                    f"{self.name} lr_reduce epoch={epoch + 1} val_Cllr={current_cllr:.6f} lr={current_lr:g}",
                    flush=True,
                )

            if improved:
                best_cllr = current_cllr
                best_epoch = epoch + 1
                best_state = {key: value.detach().cpu().clone() for key, value in self.net.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            self.trained_epochs = epoch + 1
            self.epoch_history.append(
                {
                    "epoch": epoch + 1,
                    "val_Cllr": current_cllr,
                    "best_val_Cllr": best_cllr,
                    "best_epoch": best_epoch,
                    "improved": improved,
                    "epochs_without_improvement": epochs_without_improvement,
                    "learning_rate": current_lr,
                }
            )
            if epoch == 0 or improved or (epoch + 1) % 25 == 0:
                print(
                    f"{self.name} epoch={epoch + 1} val_Cllr={current_cllr:.6f} "
                    f"best_epoch={best_epoch} best_val_Cllr={best_cllr:.6f} "
                    f"no_improve={epochs_without_improvement} lr={current_lr:g}",
                    flush=True,
                )
            if epochs_without_improvement >= self.early_stopping_patience:
                self.stopped_early = True
                print(
                    f"{self.name} early_stop epoch={epoch + 1} best_epoch={best_epoch} "
                    f"best_val_Cllr={best_cllr:.6f} lr={current_lr:g}",
                    flush=True,
                )
                break

        self.net.eval()
        if best_state is not None:
            self.net.load_state_dict(best_state)
            self.net.eval()
        self.best_epoch = best_epoch
        self.best_val_cllr = best_cllr
        print(
            f"{self.name} restored best_epoch={best_epoch} best_val_Cllr={best_cllr:.6f} "
            f"trained_epochs={self.trained_epochs} stopped_early={self.stopped_early}",
            flush=True,
        )

    def _predict_tensor(
        self,
        features: torch.Tensor,
        primary_scores: torch.Tensor,
        batch_size: int = 1_000_000,
    ) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("Model is not fitted")
        if self.mean is None or self.std is None or self.primary_mean is None or self.primary_std is None:
            raise RuntimeError("Model is not fitted")
        outputs: list[np.ndarray] = []
        self.net.eval()
        with torch.no_grad():
            for start in range(0, features.shape[0], batch_size):
                batch = (features[start : start + batch_size] - self.mean) / self.std
                primary_batch = (primary_scores[start : start + batch_size] - self.primary_mean) / self.primary_std
                logits = self.net(batch, primary_batch)
                outputs.append(logits.detach().cpu().numpy().astype(np.float64, copy=False))
        return np.concatenate(outputs)

    def decision_function(
        self,
        features: np.ndarray,
        primary_scores: np.ndarray,
        batch_size: int = 1_000_000,
    ) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Model is not fitted")
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, features.shape[0], batch_size):
                batch = torch.as_tensor(features[start : start + batch_size], dtype=torch.float32, device=self.device)
                primary_batch = torch.as_tensor(
                    primary_scores[start : start + batch_size],
                    dtype=torch.float32,
                    device=self.device,
                )
                outputs.append(self._predict_tensor(batch, primary_batch, batch_size=batch_size))
        return np.concatenate(outputs)


def canonical_split(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.lower()


def validation_mask_from_trial_ids(trial_ids: pd.Series, validation_fraction: float) -> np.ndarray:
    cutoff = int(round(validation_fraction * SPLIT_HASH_MOD))
    hashes = pd.util.hash_pandas_object(trial_ids.astype("string"), index=False).to_numpy(dtype=np.uint64)
    return (hashes % SPLIT_HASH_MOD) < cutoff


def split_masks(frame: pd.DataFrame, validation_fraction: float, calibration_fraction: float) -> dict[str, np.ndarray]:
    if validation_fraction + calibration_fraction >= 1.0:
        raise ValueError("validation_fraction + calibration_fraction must be less than 1")
    validation_cutoff = int(round(validation_fraction * SPLIT_HASH_MOD))
    calibration_cutoff = int(round((validation_fraction + calibration_fraction) * SPLIT_HASH_MOD))
    splits = canonical_split(frame["split"])
    calibration_source = splits.eq("calibration") | splits.eq("train")
    test = splits.eq("test").to_numpy()
    hashes = pd.util.hash_pandas_object(frame["trial_id"].astype("string"), index=False).to_numpy(dtype=np.uint64)
    buckets = hashes % SPLIT_HASH_MOD
    validation = calibration_source.to_numpy() & (buckets < validation_cutoff)
    calibration = calibration_source.to_numpy() & (buckets >= validation_cutoff) & (buckets < calibration_cutoff)
    train = calibration_source.to_numpy() & (buckets >= calibration_cutoff)
    return {
        "train": train,
        "validation": validation,
        "calibration": calibration,
        "test": test,
    }


def count_splits(
    path: Path,
    validation_fraction: float,
    calibration_fraction: float,
    batch_size: int,
) -> dict[str, int]:
    counts = {"train": 0, "validation": 0, "calibration": 0, "test": 0}
    parquet = pq.ParquetFile(path)
    for batch in tqdm(
        parquet.iter_batches(columns=["split", "trial_id"], batch_size=batch_size),
        desc="count c12 splits",
        unit="batch",
    ):
        frame = batch.to_pandas()
        masks = split_masks(frame, validation_fraction, calibration_fraction)
        for split, mask in masks.items():
            counts[split] += int(np.sum(mask))
    return counts


def empty_split(n_rows: int, keep_trial_ids: bool) -> SplitArrays:
    trial_ids = np.empty(n_rows, dtype=object) if keep_trial_ids else None
    return SplitArrays(
        features=np.empty((n_rows, len(FEATURE_COLS)), dtype=np.float32),
        labels=np.empty(n_rows, dtype=np.int8),
        primary_scores=np.empty((n_rows, len(PRIMARY_SCORE_COLS)), dtype=np.float32),
        trial_ids=trial_ids,
    )


def fill_split(target: SplitArrays, offset: int, frame: pd.DataFrame) -> int:
    n_rows = int(frame.shape[0])
    if n_rows == 0:
        return offset
    target.features[offset : offset + n_rows] = frame[FEATURE_COLS].to_numpy(dtype=np.float32, copy=False)
    target.labels[offset : offset + n_rows] = frame["label"].to_numpy(dtype=np.int8, copy=False)
    if all(column in frame.columns for column in PRIMARY_SCORE_COLS):
        primary_scores = frame[PRIMARY_SCORE_COLS].to_numpy(dtype=np.float32, copy=False)
    else:
        primary_scores = frame.iloc[:, [frame.columns.get_loc(SCORE_COLS[idx]) for idx in PRIMARY_SCORE_IDXS]].to_numpy(
            dtype=np.float32,
            copy=False,
        )
    target.primary_scores[offset : offset + n_rows] = primary_scores
    if target.trial_ids is not None:
        target.trial_ids[offset : offset + n_rows] = frame["trial_id"].to_numpy(dtype=object, copy=False)
    return offset + n_rows


def load_c12_dataset(
    path: Path,
    *,
    validation_fraction: float,
    calibration_fraction: float,
    batch_size: int,
    keep_test_trial_ids: bool,
) -> C12Dataset:
    if not path.exists():
        raise FileNotFoundError(path)
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1")

    parquet = pq.ParquetFile(path)
    available_columns = set(parquet.schema_arrow.names)
    missing = sorted(set([*BASE_COLS, *FEATURE_COLS]) - available_columns)
    if missing:
        raise ValueError(f"{path} is missing required C12 columns: {missing}")

    counts = count_splits(path, validation_fraction, calibration_fraction, batch_size)
    if counts["train"] == 0 or counts["validation"] == 0 or counts["calibration"] == 0 or counts["test"] == 0:
        raise ValueError(f"Expected non-empty train/validation/calibration/test splits, got {counts}")

    train = empty_split(counts["train"], keep_trial_ids=False)
    validation = empty_split(counts["validation"], keep_trial_ids=False)
    calibration = empty_split(counts["calibration"], keep_trial_ids=False)
    test = empty_split(counts["test"], keep_trial_ids=keep_test_trial_ids)
    offsets = {"train": 0, "validation": 0, "calibration": 0, "test": 0}

    columns = [*BASE_COLS, *FEATURE_COLS, *[column for column in PRIMARY_SCORE_COLS if column in available_columns]]
    for batch in tqdm(
        parquet.iter_batches(columns=columns, batch_size=batch_size),
        desc="load c12 parquet",
        unit="batch",
    ):
        frame = batch.to_pandas()
        masks = split_masks(frame, validation_fraction, calibration_fraction)

        offsets["train"] = fill_split(train, offsets["train"], frame.loc[masks["train"]])
        offsets["validation"] = fill_split(validation, offsets["validation"], frame.loc[masks["validation"]])
        offsets["calibration"] = fill_split(calibration, offsets["calibration"], frame.loc[masks["calibration"]])
        offsets["test"] = fill_split(test, offsets["test"], frame.loc[masks["test"]])

    print(
        f"loaded c12 data train={train.labels.size} validation={validation.labels.size} "
        f"calibration={calibration.labels.size} test={test.labels.size}",
        flush=True,
    )
    return C12Dataset(train=train, validation=validation, calibration=calibration, test=test)


def resolve_device(name: str) -> torch.device:
    if torch is None:
        raise RuntimeError("torch is required for C12")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        print(f"cuda_device={torch.cuda.get_device_name(device)}", flush=True)
    else:
        print("device=cpu", flush=True)
    return device


def row_prefix(model: C12GatedFusionCalibrator, calib: str) -> dict[str, Any]:
    fusion_residual_scale = math.nan
    max_fusion_residual_scale = math.nan
    if model.net is not None:
        with torch.no_grad():
            fusion_residual_scale = float(
                (model.net.max_fusion_residual_scale * torch.sigmoid(model.net.fusion_residual_logit_scale)).item()
            )
            max_fusion_residual_scale = float(model.net.max_fusion_residual_scale)
    return {
        "dataset": DATASET_NAME,
        "model": "cuda_gated_fusion_mlp" if model.device.type == "cuda" else "cpu_gated_fusion_mlp",
        "model_name": model.name,
        "calib": calib,
        "hidden_dim": model.hidden_dim,
        "latent_dim": model.latent_dim,
        "residual_blocks": model.residual_blocks,
        "dropout": model.dropout,
        "learning_rate": model.learning_rate,
        "epochs": model.epochs,
        "trained_epochs": model.trained_epochs,
        "stopped_early": model.stopped_early,
        "best_epoch": model.best_epoch,
        "best_val_Cllr": model.best_val_cllr,
        "early_stopping_patience": model.early_stopping_patience,
        "min_delta": model.min_delta,
        "batch_size": model.batch_size,
        "C": model.c_value,
        "feature_count": len(FEATURE_COLS),
        "score_feature_count": len(SCORE_COLS),
        "primary_score_skip_count": len(PRIMARY_SCORE_COLS),
        "meta_feature_count": len(META_COLS),
        "fusion_residual_scale": fusion_residual_scale,
        "max_fusion_residual_scale": max_fusion_residual_scale,
    }


def build_binary_row(
    model: C12GatedFusionCalibrator,
    *,
    validation_llrs: np.ndarray,
    validation_labels: np.ndarray,
    test_llrs: np.ndarray,
    test_labels: np.ndarray,
    llr_scale: float,
    llr_bias: float,
) -> dict[str, Any]:
    row = row_prefix(model, "C12-2")
    row["selection_split"] = "validation"
    row["llr_calibration_split"] = "calibration"
    row["llr_scale"] = llr_scale
    row["llr_bias"] = llr_bias
    threshold = threshold_for_max_accuracy(validation_llrs, validation_labels)
    row.update(metric_block(test_llrs, test_llrs, test_labels))
    selected_metrics = threshold_metrics(test_llrs, test_labels, threshold)
    row["final_accuracy"] = selected_metrics["accuracy"]
    row.update({f"threshold_{key}": value for key, value in selected_metrics.items()})
    return row


def build_reject_row(
    model: C12GatedFusionCalibrator,
    *,
    validation_llrs: np.ndarray,
    test_llrs: np.ndarray,
    test_labels: np.ndarray,
    coverage: float,
    llr_scale: float,
    llr_bias: float,
) -> dict[str, Any]:
    row = row_prefix(model, "C12-3")
    row["selection_split"] = "validation"
    row["llr_calibration_split"] = "calibration"
    row["llr_scale"] = llr_scale
    row["llr_bias"] = llr_bias

    _, validation_confidence = confidence_from_llr(validation_llrs)
    posterior_target, confidence = confidence_from_llr(test_llrs)
    threshold = confidence_threshold(validation_confidence, coverage)
    decisions = decisions_from_confidence(posterior_target, confidence, threshold)
    accepted = decisions != "reject"

    row["coverage_target"] = coverage
    row["confidence_threshold"] = threshold
    row["confidence_mean"] = float(np.mean(confidence))
    row["confidence_accepted_mean"] = float(np.mean(confidence[accepted])) if np.any(accepted) else np.nan
    row["confidence_rejected_mean"] = float(np.mean(confidence[~accepted])) if np.any(~accepted) else np.nan
    row["posterior_target_mean"] = float(np.mean(posterior_target))
    row.update(decision_metrics(test_labels, decisions))
    row["final_accuracy"] = row["accuracy"]

    accepted_metrics = metric_block(test_llrs[accepted], test_llrs[accepted], test_labels[accepted])
    for key, value in accepted_metrics.items():
        row[f"accepted_{key}"] = value
    return row


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
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


def write_history(path: Path, model: C12GatedFusionCalibrator) -> None:
    if not model.epoch_history:
        return
    rows = []
    for row in model.epoch_history:
        rows.append(
            {
                "model_name": model.name,
                "epochs": model.epochs,
                "early_stopping_patience": model.early_stopping_patience,
                "min_delta": model.min_delta,
                "max_fusion_residual_scale": model.max_fusion_residual_scale,
                **row,
            }
        )
    write_rows(path, rows)


def write_test_scores(
    path: Path,
    *,
    trial_ids: np.ndarray | None,
    labels: np.ndarray,
    llrs: np.ndarray,
) -> None:
    if trial_ids is None:
        raise ValueError("trial_ids were not retained; load with keep_test_trial_ids=True")
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "trial_id": trial_ids,
            "label": labels,
            "c12_llr": llrs,
        }
    )
    frame.to_parquet(path, index=False)


def train(args: argparse.Namespace) -> list[dict[str, Any]]:
    device = resolve_device(args.device)
    data = load_c12_dataset(
        args.input,
        validation_fraction=args.validation_fraction,
        calibration_fraction=args.calibration_fraction,
        batch_size=args.read_batch_size,
        keep_test_trial_ids=args.save_scores,
    )

    model = C12GatedFusionCalibrator(
        input_dim=len(FEATURE_COLS),
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        residual_blocks=args.residual_blocks,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        c_value=args.c_value,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        early_stopping_patience=args.early_stopping_patience,
        scheduler_patience=args.scheduler_patience,
        max_fusion_residual_scale=args.max_fusion_residual_scale,
        min_delta=args.min_delta,
    )
    model.fit(
        data.train.features,
        data.train.labels,
        balanced_sample_weights(data.train.labels),
        primary_scores=data.train.primary_scores,
        validation_features=data.validation.features,
        validation_primary_scores=data.validation.primary_scores,
        validation_labels=data.validation.labels,
    )

    validation_llrs = model.decision_function(
        data.validation.features,
        data.validation.primary_scores,
        batch_size=args.predict_batch_size,
    )
    calibration_llrs = model.decision_function(
        data.calibration.features,
        data.calibration.primary_scores,
        batch_size=args.predict_batch_size,
    )
    test_llrs = model.decision_function(
        data.test.features,
        data.test.primary_scores,
        batch_size=args.predict_batch_size,
    )
    llr_scale, llr_bias = affine_llr_calibration(calibration_llrs, data.calibration.labels)
    validation_llrs = llr_scale * validation_llrs + llr_bias
    calibration_llrs = llr_scale * calibration_llrs + llr_bias
    test_llrs = llr_scale * test_llrs + llr_bias

    binary = build_binary_row(
        model,
        validation_llrs=validation_llrs,
        validation_labels=data.validation.labels,
        test_llrs=test_llrs,
        test_labels=data.test.labels,
        llr_scale=llr_scale,
        llr_bias=llr_bias,
    )
    reject = build_reject_row(
        model,
        validation_llrs=validation_llrs,
        test_llrs=test_llrs,
        test_labels=data.test.labels,
        coverage=args.coverage,
        llr_scale=llr_scale,
        llr_bias=llr_bias,
    )
    rows = [binary, reject]

    print(
        f"{model.name} C12-2 | ValCllr {model.best_val_cllr:.6f} | "
        f"EER {binary['eer_pct']:.3f} | Cllr {binary['Cllr']:.3f} | "
        f"actDCF .01 {binary['actDCF_p01']:.3f} | Acc {binary['final_accuracy']:.4f} | "
        f"FAR {binary['threshold_FAR']:.4f} | FRR {binary['threshold_FRR']:.4f}",
        flush=True,
    )
    print(
        f"{model.name} C12-3 | Cov {reject['coverage']:.3f} | EER {reject['accepted_eer_pct']:.3f} | "
        f"Cllr {reject['accepted_Cllr']:.3f} | actDCF .01 {reject['accepted_actDCF_p01']:.3f} | "
        f"CovAcc {reject['covered_acc']:.4f} | FAR {reject['FAR']:.4f} | FRR {reject['FRR']:.4f}",
        flush=True,
    )

    write_rows(args.output, rows)
    print(f"wrote {args.output}", flush=True)
    write_history(args.history_output, model)
    print(f"wrote {args.history_output}", flush=True)

    if args.save_scores:
        write_test_scores(args.output_scores, trial_ids=data.test.trial_ids, labels=data.test.labels, llrs=test_llrs)
        print(f"wrote {args.output_scores}", flush=True)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train C12 gated score fusion on TidyVoice backend scores.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--output-scores", type=Path, default=OUTPUT_SCORES)
    parser.add_argument("--history-output", type=Path, default=HISTORY_CSV)
    parser.add_argument("--save-scores", action="store_true", help="Also write per-test-trial C12 LLRs.")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="cuda")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--calibration-fraction", type=float, default=0.10)
    parser.add_argument("--coverage", type=float, default=0.80)
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=131_072)
    parser.add_argument("--read-batch-size", type=int, default=250_000)
    parser.add_argument("--predict-batch-size", type=int, default=1_000_000)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--residual-blocks", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--learning-rate", type=float, default=5.0e-4)
    parser.add_argument("--c-value", type=float, default=10.0)
    parser.add_argument("--early-stopping-patience", type=int, default=40)
    parser.add_argument("--scheduler-patience", type=int, default=8)
    parser.add_argument("--max-fusion-residual-scale", type=float, default=0.85)
    parser.add_argument("--min-delta", type=float, default=1.0e-5)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()

