from __future__ import annotations

import csv
import json
import math
import os
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

try:
    import torch
except ImportError:
    torch = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_TABLE = Path(
    os.environ.get(
        "AMECXSV_INPUT_TABLE",
        PROJECT_ROOT / "data" / "tidyvoice_trials.parquet",
    )
)
DATASET_NAME = os.environ.get("AMECXSV_DATASET_NAME", "tidyvoice")
BASELINE_DIR = PROJECT_ROOT / "outputs"

EMBEDDINGS = [
    "speechbrain_ecapa_tdnn_voxceleb",
    "wespeaker_resnet34_cnceleb",
    "funasr_campplus_cn_3k",
    "funasr_eres2netv2_cn_200k",
    "hf_wavlm_base_sv_voxceleb1",
    "hf_wavlm_base_plus_sv_voxceleb1",
]

QMF_COLS = ["qmf1", "qmf2", "qmf3", "qmf4_sum", "qmf4_diff"]
C5_COLS = ["min_duration", "duration_ratio", "short_duration_risk"]
BASE_COLS = ["split", "trial_id", "label", "target", *QMF_COLS, *C5_COLS]
SPLIT_ALIASES = {
    "train": ("train", "calibration", 0),
    "calibration": ("train", "calibration", 0),
    "validation": ("validation", "valid", "val", 2),
    "valid": ("validation", "valid", "val", 2),
    "val": ("validation", "valid", "val", 2),
    "test": ("test", 1),
}
PRIORS = {"p001": 0.001, "p01": 0.01}
BOOTSTRAP_SAMPLES = 1000
BOOTSTRAP_SEED = 20260608
BOOTSTRAP_WORKERS = min(8, os.cpu_count() or 1)
METRIC_CI_KEYS = (
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
DECISION_CI_KEYS = (
    "accuracy",
    "effective_acc",
    "covered_acc",
    "coverage",
    "FAR",
    "FRR",
)
BEST_MLP_SETTING = {
    "hidden_dim": 24,
    "dropout": 0.12,
    "learning_rate": 5.0e-4,
    "epochs": 300,
    "batch_size": 131_072,
    "c_value": 30.0,
}


@dataclass(frozen=True)
class Experiment:
    calib: str
    feature_mode: str
    c_value: float
    hidden_dim: int = 64
    dropout: float = 0.10
    learning_rate: float = 1.0e-3
    epochs: int = 80
    batch_size: int = 262_144
    reject_rate: float = 0.0
    risk_duration_weight: float = 0.0
    risk_language_weight: float = 0.0
    seed: int = 0

    @property
    def param_name(self) -> str:
        model_part = (
            f"mlp_h={self.hidden_dim}|drop={self.dropout:g}|lr={self.learning_rate:g}"
            f"|epochs={self.epochs}|C={self.c_value:g}"
        )
        if self.calib == "C5":
            return (
                f"{self.feature_mode}|{model_part}|reject={self.reject_rate:g}"
                f"|dur_w={self.risk_duration_weight:g}|lang_w={self.risk_language_weight:g}"
            )
        return f"{self.feature_mode}|{model_part}"


@dataclass(frozen=True)
class RiskStats:
    margin_mean: float
    margin_std: float
    duration_mean: float
    duration_std: float


class CudaMlpCalibrationModel:
    def __init__(self, experiment: Experiment, device: torch.device) -> None:
        self.experiment = experiment
        self.device = device
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        self.net = None

    def fit(self, x: np.ndarray, labels: np.ndarray, sample_weights: np.ndarray, desc: str) -> None:
        torch.manual_seed(int(self.experiment.seed))
        x_tensor = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        y_tensor = torch.as_tensor(labels.astype(np.float32), dtype=torch.float32, device=self.device)
        w_tensor = torch.as_tensor(sample_weights.astype(np.float32), dtype=torch.float32, device=self.device)

        self.mean = x_tensor.mean(dim=0, keepdim=True)
        self.std = x_tensor.std(dim=0, keepdim=True, unbiased=False).clamp_min(1.0e-6)
        x_scaled = (x_tensor - self.mean) / self.std

        hidden = self.experiment.hidden_dim
        self.net = torch.nn.Sequential(
            torch.nn.Linear(x_scaled.shape[1], hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.SiLU(),
            torch.nn.Dropout(self.experiment.dropout),
            torch.nn.Linear(hidden, hidden),
            torch.nn.LayerNorm(hidden),
            torch.nn.SiLU(),
            torch.nn.Dropout(self.experiment.dropout),
            torch.nn.Linear(hidden, 1),
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.net.parameters(),
            lr=self.experiment.learning_rate,
            weight_decay=1.0 / max(self.experiment.c_value, 1.0e-12),
        )

        n_rows = x_scaled.shape[0]
        batch_size = min(self.experiment.batch_size, n_rows)
        for _ in range(self.experiment.epochs):
            order = torch.randperm(n_rows, device=self.device)
            self.net.train()
            for start in range(0, n_rows, batch_size):
                idx = order[start : start + batch_size]
                logits = self.net(x_scaled[idx]).squeeze(1)
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
        self.net.eval()

    def decision_function(self, x: np.ndarray, batch_size: int = 1_000_000) -> np.ndarray:
        if self.mean is None or self.std is None or self.net is None:
            raise RuntimeError("Model is not fitted")
        outputs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, x.shape[0], batch_size):
                batch = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=self.device)
                batch = (batch - self.mean) / self.std
                logits = self.net(batch).squeeze(1)
                outputs.append(logits.detach().cpu().numpy().astype(np.float64, copy=False))
        return np.concatenate(outputs)


class RichMlpCalibrationModel:
    """C10 rich MLP with validation early stopping and affine LLR calibration."""

    def __init__(self, experiment: Experiment, device: torch.device, num_scores: int) -> None:
        self.experiment = experiment
        self.device = device
        self.num_scores = num_scores
        self.backend = None
        self.llr_scale = 1.0
        self.llr_bias = 0.0

    @property
    def mean(self):
        return None if self.backend is None else self.backend.mean

    @property
    def std(self):
        return None if self.backend is None else self.backend.std

    @property
    def net(self):
        return None if self.backend is None else self.backend.net

    @property
    def name(self) -> str:
        return "rich_feature_mlp" if self.backend is None else self.backend.name

    def _initialize_input_dim(self, input_dim: int) -> None:
        if self.backend is not None:
            if self.backend.input_dim != input_dim:
                raise ValueError(f"Expected {self.backend.input_dim} input features, got {input_dim}")
            return
        from rich_mlp import rich_feature_mlp

        self.backend = rich_feature_mlp(
            input_dim=input_dim,
            num_scores=self.num_scores,
            hidden_dim=self.experiment.hidden_dim,
            dropout=self.experiment.dropout,
            learning_rate=self.experiment.learning_rate,
            c_value=self.experiment.c_value,
            epochs=self.experiment.epochs,
            batch_size=self.experiment.batch_size,
            device=self.device,
        )

    def fit(
        self,
        x: np.ndarray,
        labels: np.ndarray,
        sample_weights: np.ndarray,
        desc: str = "",
        *,
        validation_features: np.ndarray | None = None,
        validation_labels: np.ndarray | None = None,
    ) -> None:
        del desc
        self._initialize_input_dim(x.shape[1])
        self.backend.fit(
            x,
            labels,
            sample_weights,
            validation_features=validation_features,
            validation_labels=validation_labels,
        )
        if validation_features is not None and validation_labels is not None:
            validation_llrs = self.backend.decision_function(validation_features)
            self.llr_scale, self.llr_bias = affine_llr_calibration(validation_llrs, validation_labels)

    def decision_function(self, x: np.ndarray, batch_size: int = 1_000_000) -> np.ndarray:
        if self.backend is None:
            raise RuntimeError("Model is not fitted")
        raw = self.backend.decision_function(x, batch_size=batch_size)
        return self.llr_scale * raw + self.llr_bias


def score_cols(embeddings: list[str]) -> list[str]:
    return [f"score_{embedding}" for embedding in embeddings]


def load_fixed_splits(
    embeddings: list[str],
    chunksize: int,
    max_train_rows_per_class: int | None,
    max_eval_rows_per_class: int | None,
    *,
    return_validation: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame] | tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    if not INPUT_TABLE.exists():
        raise FileNotFoundError(INPUT_TABLE)
    usecols = BASE_COLS + score_cols(embeddings)
    if INPUT_TABLE.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(INPUT_TABLE, columns=usecols)
        train = limit_rows_per_class(frame[split_mask(frame["split"], "train")], max_train_rows_per_class)
        test = limit_rows_per_class(frame[split_mask(frame["split"], "test")], max_eval_rows_per_class)
        if train.empty or test.empty:
            raise ValueError(f"No train/test rows found in {INPUT_TABLE}")
        if return_validation:
            validation_frame = frame[split_mask(frame["split"], "validation")]
            validation = None if validation_frame.empty else limit_rows_per_class(validation_frame, max_eval_rows_per_class)
            return train, validation, test
        return train, test
    train = load_split("train", usecols, chunksize, max_train_rows_per_class)
    test = load_split("test", usecols, chunksize, max_eval_rows_per_class)
    if return_validation:
        validation = load_split("validation", usecols, chunksize, max_eval_rows_per_class, required=False)
        if validation.empty:
            return train, None, test
        return train, validation, test
    return train, test


def load_fixed_train_validation_test_splits(
    embeddings: list[str],
    chunksize: int,
    max_train_rows_per_class: int | None,
    max_validation_rows_per_class: int | None,
    max_test_rows_per_class: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not INPUT_TABLE.exists():
        raise FileNotFoundError(INPUT_TABLE)
    usecols = BASE_COLS + score_cols(embeddings)
    if INPUT_TABLE.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(INPUT_TABLE, columns=usecols)
        train = limit_rows_per_class(
            frame[split_mask(frame["split"], "calibration")],
            max_train_rows_per_class,
        )
        validation = limit_rows_per_class(
            frame[split_mask(frame["split"], "validation")],
            max_validation_rows_per_class,
        )
        test = limit_rows_per_class(
            frame[split_mask(frame["split"], "test")],
            max_test_rows_per_class,
        )
        if train.empty or validation.empty or test.empty:
            raise ValueError(f"No rows found in {INPUT_TABLE}")
        return train, validation, test
    train = load_split("calibration", usecols, chunksize, max_train_rows_per_class)
    validation = load_split("validation", usecols, chunksize, max_validation_rows_per_class)
    test = load_split("test", usecols, chunksize, max_test_rows_per_class)
    return train, validation, test


def split_mask(series: pd.Series, split: str) -> pd.Series:
    aliases = SPLIT_ALIASES.get(split)
    if aliases is None:
        aliases = (split,)
    text_aliases = [str(alias).lower() for alias in aliases if isinstance(alias, str)]
    numeric_aliases = [alias for alias in aliases if not isinstance(alias, str)]
    mask = series.isin(numeric_aliases)
    if text_aliases:
        mask = mask | series.astype("string").str.strip().str.lower().isin(text_aliases)
    return mask.fillna(False)


def assert_speaker_disjoint(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> dict[str, int]:
    speaker_columns = {"enroll_speaker", "test_speaker"}
    if not speaker_columns.issubset(train.columns):
        return {}

    def speakers(frame: pd.DataFrame) -> set[str]:
        values = set(frame["enroll_speaker"].astype(str))
        values.update(frame["test_speaker"].astype(str))
        values.discard("")
        return values

    train_speakers = speakers(train)
    validation_speakers = speakers(validation)
    test_speakers = speakers(test)
    overlaps = {
        "calibration_validation": len(train_speakers & validation_speakers),
        "calibration_test": len(train_speakers & test_speakers),
        "validation_test": len(validation_speakers & test_speakers),
    }
    if any(overlaps.values()):
        raise ValueError(f"Speaker overlap detected: {overlaps}")
    return overlaps


def limit_rows_per_class(frame: pd.DataFrame, max_rows_per_class: int | None) -> pd.DataFrame:
    frame = frame.reset_index(drop=True)
    if max_rows_per_class is None:
        return frame
    parts = [frame[frame["label"] == label].head(max_rows_per_class) for label in (0, 1)]
    if any(part.empty for part in parts):
        raise ValueError("Selected split does not contain both classes")
    return pd.concat(parts, ignore_index=True)


def load_split(
    split: str,
    usecols: list[str],
    chunksize: int,
    max_rows_per_class: int | None,
    *,
    required: bool = True,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    counts = {0: 0, 1: 0}
    reader = pd.read_csv(INPUT_TABLE, usecols=usecols, chunksize=chunksize)
    for chunk in tqdm(reader, desc=f"load {split}", unit="chunk"):
        chunk = chunk[split_mask(chunk["split"], split)]
        if chunk.empty:
            continue
        if max_rows_per_class is None:
            frames.append(chunk)
            continue
        selected_parts = []
        for label in (0, 1):
            need = max_rows_per_class - counts[label]
            if need <= 0:
                continue
            part = chunk[chunk["label"] == label].head(need)
            if not part.empty:
                counts[label] += int(part.shape[0])
                selected_parts.append(part)
        if selected_parts:
            frames.append(pd.concat(selected_parts, ignore_index=True))
        if counts[0] >= max_rows_per_class and counts[1] >= max_rows_per_class:
            break
    if not frames:
        if not required:
            return pd.DataFrame(columns=usecols)
        raise ValueError(f"No rows found for split={split}")
    frame = pd.concat(frames, ignore_index=True)
    if required and max_rows_per_class is not None and (counts[0] == 0 or counts[1] == 0):
        raise ValueError(f"split={split} does not contain both classes")
    return frame


def balanced_sample_weights(labels: np.ndarray) -> np.ndarray:
    labels_bool = labels.astype(bool)
    target_n = int(labels_bool.sum())
    nontarget_n = int(labels_bool.shape[0] - target_n)
    if target_n == 0 or nontarget_n == 0:
        raise ValueError("Both target and nontarget trials are required")
    weights = np.empty(labels_bool.shape[0], dtype=np.float64)
    weights[labels_bool] = 0.5 / target_n
    weights[~labels_bool] = 0.5 / nontarget_n
    return weights


def affine_llr_calibration(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    y = labels.astype(np.float64, copy=False)
    weights = balanced_sample_weights(labels).astype(
        np.float64, copy=False
    )

    def objective(
        params: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        log_scale, bias = params
        scale = float(np.exp(np.clip(log_scale, -8.0, 8.0)))
        logits = scale * scores + bias
        loss_terms = np.logaddexp(0.0, logits) - y * logits
        loss = float(np.sum(weights * loss_terms) / np.sum(weights))
        posterior = 1.0 / (
            1.0 + np.exp(-np.clip(logits, -50.0, 50.0))
        )
        grad_logits = weights * (posterior - y) / np.sum(weights)
        grad_scale = float(np.sum(grad_logits * scores) * scale)
        grad_bias = float(np.sum(grad_logits))
        return loss, np.asarray(
            [grad_scale, grad_bias], dtype=np.float64
        )

    result = minimize(
        objective,
        x0=np.asarray([0.0, 0.0], dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        bounds=[(-8.0, 8.0), (None, None)],
        options={"maxiter": 200},
    )
    log_scale, bias = (
        result.x if result.success else (0.0, 0.0)
    )
    return (
        float(np.exp(np.clip(log_scale, -8.0, 8.0))),
        float(bias),
    )


def require_cuda() -> torch.device:
    if torch is None:
        raise RuntimeError("CUDA is required, but torch is not installed in the cvpr environment.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Install CUDA-enabled torch in the cvpr environment.")
    device = torch.device("cuda")
    print(f"cuda_device={torch.cuda.get_device_name(device)}", flush=True)
    return device


def make_model(experiment: Experiment, device: torch.device) -> CudaMlpCalibrationModel:
    return CudaMlpCalibrationModel(experiment=experiment, device=device)


def make_rich_model(experiment: Experiment, device: torch.device, *, num_scores: int) -> RichMlpCalibrationModel:
    return RichMlpCalibrationModel(experiment=experiment, device=device, num_scores=num_scores)


def model_key(experiment: Experiment) -> tuple[str, float, int, float, float, int, int]:
    return (
        experiment.feature_mode,
        experiment.c_value,
        experiment.hidden_dim,
        experiment.dropout,
        experiment.learning_rate,
        experiment.epochs,
        experiment.batch_size,
    )


def feature_matrix(frame: pd.DataFrame, embedding: str, mode: str) -> np.ndarray:
    score = frame[f"score_{embedding}"].to_numpy(dtype=np.float64, copy=False)
    lang_same = frame["target"].to_numpy(dtype=np.float64, copy=False)
    qmf = frame[QMF_COLS].to_numpy(dtype=np.float64, copy=False)
    c5 = frame[C5_COLS].to_numpy(dtype=np.float64, copy=False)

    columns: list[np.ndarray] = [score[:, None]]
    if mode == "c3_qmf":
        columns.append(qmf)
    elif mode == "c4_lang":
        columns.append(lang_same[:, None])
    elif mode == "c5_reject":
        columns.append(c5)
    elif mode == "c6_fusion":
        columns.extend([qmf, c5, lang_same[:, None]])
    elif mode == "c6_no_quality":
        columns.extend([c5, lang_same[:, None]])
    elif mode in {"c7_2_binary", "c7_3_conf_reject"}:
        columns.extend([c5, lang_same[:, None]])
    elif mode == "c6_no_language":
        columns.extend([qmf, c5])
    elif mode == "c6_no_reliability":
        columns.extend([qmf, lang_same[:, None]])
    elif mode == "c6_score_only":
        pass
    else:
        raise ValueError(f"Unknown feature mode: {mode}")
    return np.concatenate(columns, axis=1)


def rates_for_unique_thresholds(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels_bool = labels.astype(bool)
    target_n = int(labels_bool.sum())
    nontarget_n = int(labels_bool.shape[0] - target_n)
    if target_n == 0 or nontarget_n == 0:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty

    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_scores = scores[order]
    sorted_labels = labels_bool[order]
    target_cumsum = np.cumsum(sorted_labels, dtype=np.int64)
    nontarget_cumsum = np.cumsum(~sorted_labels, dtype=np.int64)
    unique_ends = np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:])
    unique_ends = np.concatenate([unique_ends, np.asarray([sorted_scores.shape[0] - 1], dtype=np.int64)])

    target_accepts = target_cumsum[unique_ends]
    nontarget_accepts = nontarget_cumsum[unique_ends]
    pmiss = (target_n - target_accepts).astype(np.float64) / target_n
    pfa = nontarget_accepts.astype(np.float64) / nontarget_n
    thresholds = sorted_scores[unique_ends]

    return (
        np.concatenate([np.asarray([np.inf]), thresholds]),
        np.concatenate([np.asarray([1.0]), pmiss]),
        np.concatenate([np.asarray([0.0]), pfa]),
    )


def eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    thresholds, pmiss, pfa = rates_for_unique_thresholds(scores, labels)
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


def normalized_dcf(pmiss: np.ndarray, pfa: np.ndarray, prior: float) -> np.ndarray:
    return (prior * pmiss + (1.0 - prior) * pfa) / min(prior, 1.0 - prior)


def min_dcf(scores: np.ndarray, labels: np.ndarray, prior: float) -> float:
    _, pmiss, pfa = rates_for_unique_thresholds(scores, labels)
    if pmiss.size == 0:
        return math.nan
    return float(np.min(normalized_dcf(pmiss, pfa, prior)))


def act_dcf(llrs: np.ndarray, labels: np.ndarray, prior: float) -> float:
    labels_bool = labels.astype(bool)
    target_n = int(labels_bool.sum())
    nontarget_n = int(labels_bool.shape[0] - target_n)
    if target_n == 0 or nontarget_n == 0:
        return math.nan
    threshold = math.log((1.0 - prior) / prior)
    accept = llrs >= threshold
    pmiss = np.sum(labels_bool & ~accept) / target_n
    pfa = np.sum(~labels_bool & accept) / nontarget_n
    return float(normalized_dcf(np.asarray([pmiss]), np.asarray([pfa]), prior)[0])


def cllr(llrs: np.ndarray, labels: np.ndarray) -> float:
    labels_bool = labels.astype(bool)
    target_llrs = llrs[labels_bool]
    nontarget_llrs = llrs[~labels_bool]
    if target_llrs.size == 0 or nontarget_llrs.size == 0:
        return math.nan
    target_loss = np.mean(np.logaddexp(0.0, -target_llrs))
    nontarget_loss = np.mean(np.logaddexp(0.0, nontarget_llrs))
    return float(0.5 * (target_loss + nontarget_loss) / math.log(2.0))


def classification_metrics(llrs: np.ndarray, labels: np.ndarray, threshold: float = 0.0) -> dict[str, float]:
    labels_bool = labels.astype(bool)
    pred = llrs >= threshold
    tp = int(np.sum(pred & labels_bool))
    fp = int(np.sum(pred & ~labels_bool))
    tn = int(np.sum(~pred & ~labels_bool))
    fn = int(np.sum(~pred & labels_bool))
    total = labels_bool.shape[0]
    precision = math.nan if tp + fp == 0 else tp / (tp + fp)
    recall = math.nan if tp + fn == 0 else tp / (tp + fn)
    f1 = math.nan if not math.isfinite(precision) or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "accuracy": math.nan if total == 0 else (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def metric_block(scores: np.ndarray, llrs: np.ndarray, labels: np.ndarray) -> dict[str, float | int]:
    labels_bool = labels.astype(bool)
    target_n = int(labels_bool.sum())
    nontarget_n = int(labels_bool.shape[0] - target_n)
    row: dict[str, float | int] = {
        "n": int(labels.shape[0]),
        "target_n": target_n,
        "nontarget_n": nontarget_n,
    }
    if labels.shape[0] == 0 or target_n == 0 or nontarget_n == 0:
        row.update({"eer_pct": math.nan, "eer_threshold": math.nan, "Cllr": math.nan})
        for suffix in PRIORS:
            row[f"minDCF_{suffix}"] = math.nan
            row[f"actDCF_{suffix}"] = math.nan
        row.update(classification_metrics(llrs, labels))
        return row
    eer_value, eer_threshold = eer(scores, labels)
    row.update({"eer_pct": 100.0 * eer_value, "eer_threshold": eer_threshold, "Cllr": cllr(llrs, labels)})
    for suffix, prior in PRIORS.items():
        row[f"minDCF_{suffix}"] = min_dcf(scores, labels, prior)
        row[f"actDCF_{suffix}"] = act_dcf(llrs, labels, prior)
    row.update(classification_metrics(llrs, labels))
    return row


def stable_seed(desc: str) -> int:
    return int((BOOTSTRAP_SEED + zlib.crc32(desc.encode("utf-8"))) % np.iinfo(np.uint32).max)


def percentile_ci95(values: list[float]) -> float:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return math.nan
    low, high = np.percentile(finite, [2.5, 97.5])
    return float((high - low) / 2.0)


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


def weighted_metric_block(context: dict[str, np.ndarray], counts: np.ndarray) -> dict[str, float | int]:
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


def bootstrap_metric_ci95(
    scores: np.ndarray,
    llrs: np.ndarray,
    labels: np.ndarray,
    *,
    desc: str,
    keys: tuple[str, ...] = METRIC_CI_KEYS,
) -> dict[str, float]:
    if BOOTSTRAP_SAMPLES <= 0 or labels.shape[0] == 0:
        return {f"{key}_ci95": math.nan for key in keys}

    samples: dict[str, list[float]] = {key: [] for key in keys}
    n = int(labels.shape[0])
    context = sorted_score_context(scores, llrs, labels)
    rng = np.random.default_rng(stable_seed(desc))
    seeds = rng.integers(0, np.iinfo(np.uint32).max, size=BOOTSTRAP_SAMPLES, dtype=np.uint32)

    def one_sample(seed: np.uint32) -> dict[str, float]:
        local_rng = np.random.default_rng(int(seed))
        idx = local_rng.integers(0, n, size=n, dtype=np.int64)
        counts = np.bincount(idx, minlength=n)
        metrics = weighted_metric_block(context, counts)
        return {key: float(metrics[key]) for key in keys}

    with ThreadPoolExecutor(max_workers=BOOTSTRAP_WORKERS) as pool:
        iterator = pool.map(one_sample, seeds)
        for metrics in tqdm(iterator, total=BOOTSTRAP_SAMPLES, desc=f"bootstrap {desc}", unit="sample", leave=False):
            for key in keys:
                samples[key].append(metrics[key])
    return {f"{key}_ci95": percentile_ci95(values) for key, values in samples.items()}


def add_metric_ci(
    row: dict[str, float | int | str],
    scores: np.ndarray,
    llrs: np.ndarray,
    labels: np.ndarray,
    *,
    desc: str,
    prefix: str = "",
) -> None:
    ci = bootstrap_metric_ci95(scores, llrs, labels, desc=desc)
    for key, value in ci.items():
        row[f"{prefix}{key}"] = value


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
    correct_accepted = int(np.sum(counts[(target_decision & labels_bool) | (nontarget_decision & ~labels_bool)]))
    correct_all = correct_accepted
    if not np.any(reject_decision):
        correct_all = int(np.sum(counts[(target_decision & labels_bool) | (nontarget_decision & ~labels_bool)]))

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
        "rejected_n": int(np.sum(counts[reject_decision])),
        "coverage": coverage,
        "accuracy": accuracy,
        "effective_acc": effective_acc,
        "covered_acc": covered_acc,
        "FAR": far,
        "FRR": frr,
    }


def bootstrap_decision_ci95(
    labels: np.ndarray,
    decisions: np.ndarray,
    *,
    desc: str,
    keys: tuple[str, ...] = DECISION_CI_KEYS,
) -> dict[str, float]:
    if BOOTSTRAP_SAMPLES <= 0 or labels.shape[0] == 0:
        return {f"{key}_ci95": math.nan for key in keys}

    samples: dict[str, list[float]] = {key: [] for key in keys}
    n = int(labels.shape[0])
    rng = np.random.default_rng(stable_seed(desc))
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


def zscore_from_stats(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    if not math.isfinite(std) or std <= 0.0:
        return np.zeros_like(values, dtype=np.float64)
    return (values - mean) / std


def risk_stats(frame: pd.DataFrame, llrs: np.ndarray, decision_prior: float) -> RiskStats:
    threshold = math.log((1.0 - decision_prior) / decision_prior)
    margin_risk = -np.abs(llrs - threshold)
    duration_risk = frame["short_duration_risk"].to_numpy(dtype=np.float64, copy=False)
    ratio_risk = 1.0 - frame["duration_ratio"].to_numpy(dtype=np.float64, copy=False)
    duration_total = duration_risk + ratio_risk
    return RiskStats(
        margin_mean=float(np.mean(margin_risk)),
        margin_std=float(np.std(margin_risk)),
        duration_mean=float(np.mean(duration_total)),
        duration_std=float(np.std(duration_total)),
    )


def reject_risk(
    frame: pd.DataFrame,
    llrs: np.ndarray,
    decision_prior: float,
    duration_weight: float,
    language_weight: float,
    stats: RiskStats,
) -> np.ndarray:
    threshold = math.log((1.0 - decision_prior) / decision_prior)
    margin_risk = -np.abs(llrs - threshold)
    duration_risk = frame["short_duration_risk"].to_numpy(dtype=np.float64, copy=False)
    ratio_risk = 1.0 - frame["duration_ratio"].to_numpy(dtype=np.float64, copy=False)
    language_risk = 1.0 - frame["target"].to_numpy(dtype=np.float64, copy=False)
    risk = zscore_from_stats(margin_risk, stats.margin_mean, stats.margin_std)
    risk = risk + duration_weight * zscore_from_stats(duration_risk + ratio_risk, stats.duration_mean, stats.duration_std)
    risk = risk + language_weight * language_risk
    return risk


def evaluate(
    frame: pd.DataFrame,
    embedding: str,
    experiment: Experiment,
    llrs: np.ndarray,
    risk: np.ndarray | None,
    reject_threshold: float | None = None,
    selection_split: str = "train",
) -> dict[str, float | int | str]:
    labels = frame["label"].to_numpy(dtype=np.int8, copy=False)
    accept_mask = np.ones(labels.shape[0], dtype=bool)
    if experiment.calib == "C5" and experiment.reject_rate > 0.0:
        threshold = float(np.quantile(risk, 1.0 - experiment.reject_rate)) if reject_threshold is None else reject_threshold
        accept_mask = risk <= threshold

    row: dict[str, float | int | str] = {
        "dataset": DATASET_NAME,
        "split": "test",
        "embedding": embedding,
        "calib": experiment.calib,
        "feature_mode": experiment.feature_mode,
        "param_name": experiment.param_name,
        "C": experiment.c_value,
        "model": "rich_feature_mlp",
        "hidden_dim": experiment.hidden_dim,
        "dropout": experiment.dropout,
        "learning_rate": experiment.learning_rate,
        "epochs": experiment.epochs,
        "batch_size": experiment.batch_size,
        "coverage": float(np.mean(accept_mask)),
        "selection_split": selection_split,
    }
    if experiment.calib == "C5":
        row.update(
            {
                "reject_rate_target": experiment.reject_rate,
                "reject_rate_actual": 1.0 - float(np.mean(accept_mask)),
                "reject_threshold": math.nan if reject_threshold is None else reject_threshold,
                "risk_duration_weight": experiment.risk_duration_weight,
                "risk_language_weight": experiment.risk_language_weight,
            }
        )
    row.update(metric_block(llrs[accept_mask], llrs[accept_mask], labels[accept_mask]))

    same = frame["target"].to_numpy(dtype=bool, copy=False)
    for prefix, subset in (("same_language", same), ("cross_language", ~same)):
        subset_mask = accept_mask & subset
        subset_metrics = metric_block(llrs[subset_mask], llrs[subset_mask], labels[subset_mask])
        for key, value in subset_metrics.items():
            row[f"{prefix}_{key}"] = value

    if experiment.calib == "C5" and np.any(~accept_mask):
        rejected = ~accept_mask
        rejected_pred = llrs[rejected] >= 0.0
        rejected_labels = labels[rejected].astype(bool)
        row["rejected_n"] = int(np.sum(rejected))
        row["rejected_error_rate"] = float(np.mean(rejected_pred != rejected_labels))
        row["accepted_error_rate"] = 1.0 - float(row["accuracy"])
    elif experiment.calib == "C5":
        row["rejected_n"] = 0
        row["rejected_error_rate"] = math.nan
        row["accepted_error_rate"] = 1.0 - float(row["accuracy"])
    labels_for_ci = labels[accept_mask]
    llrs_for_ci = llrs[accept_mask]
    add_metric_ci(
        row,
        llrs_for_ci,
        llrs_for_ci,
        labels_for_ci,
        desc=f"{embedding} {experiment.calib} {experiment.feature_mode}",
    )
    return row


def run_experiments(
    output_csv: Path,
    embeddings: list[str],
    experiments: list[Experiment],
    chunksize: int,
    max_train_rows_per_class: int | None,
    max_eval_rows_per_class: int | None,
    decision_prior: float,
) -> None:
    device = require_cuda()
    train, validation, test = load_fixed_splits(
        embeddings,
        chunksize,
        max_train_rows_per_class,
        max_eval_rows_per_class,
        return_validation=True,
    )
    labels = train["label"].to_numpy(dtype=np.int8, copy=False)
    sample_weights = balanced_sample_weights(labels)
    selection_frame = train if validation is None else validation
    selection_split = "train" if validation is None else "validation"
    rows: list[dict[str, float | int | str]] = []

    for embedding in tqdm(embeddings, desc="embedding"):
        trained: dict[tuple[str, float, int, float, float, int, int], RichMlpCalibrationModel] = {}
        test_llrs: dict[tuple[str, float, int, float, float, int, int], np.ndarray] = {}
        c5_risk_stats: dict[tuple[str, float, int, float, float, int, int], RiskStats] = {}
        for experiment in tqdm(experiments, desc=f"{embedding} experiments", leave=False):
            key = model_key(experiment)
            if key not in trained:
                model = make_rich_model(experiment, device, num_scores=1)
                x_train = feature_matrix(train, embedding, experiment.feature_mode)
                validation_features = None
                validation_labels = None
                if validation is not None:
                    validation_features = feature_matrix(validation, embedding, experiment.feature_mode)
                    validation_labels = validation["label"].to_numpy(dtype=np.int8, copy=False)
                model.fit(
                    x_train,
                    labels,
                    sample_weights,
                    desc=f"train {embedding} {experiment.param_name}",
                    validation_features=validation_features,
                    validation_labels=validation_labels,
                )
                trained[key] = model
                if experiment.calib == "C5":
                    c5_risk_stats[key] = risk_stats(train, model.decision_function(x_train), decision_prior)
            if key not in test_llrs:
                x_test = feature_matrix(test, embedding, experiment.feature_mode)
                test_llrs[key] = trained[key].decision_function(x_test)

            llrs = test_llrs[key]
            risk = None
            reject_threshold = None
            if experiment.calib == "C5":
                selection_features = feature_matrix(selection_frame, embedding, experiment.feature_mode)
                selection_llrs = trained[key].decision_function(selection_features)
                selection_risk = reject_risk(
                    selection_frame,
                    selection_llrs,
                    decision_prior,
                    experiment.risk_duration_weight,
                    experiment.risk_language_weight,
                    c5_risk_stats[key],
                )
                reject_threshold = float(np.quantile(selection_risk, 1.0 - experiment.reject_rate))
                risk = reject_risk(
                    test,
                    llrs,
                    decision_prior,
                    experiment.risk_duration_weight,
                    experiment.risk_language_weight,
                    c5_risk_stats[key],
                )
            rows.append(evaluate(test, embedding, experiment, llrs, risk, reject_threshold, selection_split))

        print(
            json.dumps(
                {
                    "embedding": embedding,
                    "train_rows": int(train.shape[0]),
                    "selection_split": selection_split,
                    "selection_rows": int(selection_frame.shape[0]),
                    "test_rows": int(test.shape[0]),
                    "train_target_rows": int(labels.sum()),
                    "train_nontarget_rows": int(labels.shape[0] - labels.sum()),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    write_rows(output_csv, rows)
    print(f"wrote {output_csv}", flush=True)


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
