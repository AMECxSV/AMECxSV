#!/usr/bin/env python3
"""Calibration-only backend models for trial similarity scores."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.covariance import LedoitWolf, OAS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from similarity_scores import EPS, cosine, validate_pairwise_inputs


BACKEND_METRICS = {
    "centered_cosine",
    "whitened_cosine",
    "wccn_cosine",
    "lda_cosine",
    "neg_mahalanobis",
    "plda_llr",
    "asnorm_cosine",
}


class BackendFitError(RuntimeError):
    """Raised when a backend cannot be fit without violating assumptions."""


def stable_speaker_hash(speakers: Iterable[str]) -> str:
    """Hash a speaker set without leaking raw IDs into compact artifacts."""

    payload = "\n".join(sorted({str(speaker) for speaker in speakers})).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def manifest_no_test_leakage(fit_speakers: Iterable[str], test_speakers: Iterable[str]) -> dict[str, Any]:
    """Return speaker-overlap metadata for an artifact manifest."""

    fit_set = {str(speaker) for speaker in fit_speakers}
    test_set = {str(speaker) for speaker in test_speakers}
    overlap = sorted(fit_set & test_set)
    return {
        "fit_speaker_count": len(fit_set),
        "fit_speaker_hash": stable_speaker_hash(fit_set),
        "test_speaker_count": len(test_set),
        "test_speaker_hash": stable_speaker_hash(test_set),
        "test_speaker_intersection_count": len(overlap),
        "test_speaker_intersection_preview": overlap[:10],
    }


def _validate_fit_matrix(embeddings: np.ndarray, *, min_rows: int = 2) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2:
        raise BackendFitError(f"Expected 2D embeddings, got shape={values.shape}")
    if values.shape[0] < min_rows:
        raise BackendFitError(f"Need at least {min_rows} embeddings, got {values.shape[0]}")
    if values.shape[1] == 0:
        raise BackendFitError("Embedding dimension must be nonzero.")
    if not np.isfinite(values).all():
        raise BackendFitError("Embeddings contain NaN or Inf values.")
    return values


def _speaker_array(speaker_labels: Sequence[str], n_rows: int) -> np.ndarray:
    speakers = np.asarray([str(value) for value in speaker_labels], dtype=object)
    if speakers.shape[0] != n_rows:
        raise BackendFitError(f"speaker_labels length={speakers.shape[0]} does not match embeddings rows={n_rows}")
    return speakers


def _covariance(values: np.ndarray, method: str) -> tuple[np.ndarray, dict[str, Any]]:
    if method == "ledoit_wolf":
        estimator = LedoitWolf().fit(values)
        return np.asarray(estimator.covariance_, dtype=np.float64), {
            "covariance_method": method,
            "shrinkage": float(estimator.shrinkage_),
        }
    if method == "oas":
        estimator = OAS().fit(values)
        return np.asarray(estimator.covariance_, dtype=np.float64), {
            "covariance_method": method,
            "shrinkage": float(estimator.shrinkage_),
        }
    if method == "empirical":
        return np.cov(values, rowvar=False, bias=True), {
            "covariance_method": method,
            "shrinkage": 0.0,
        }
    raise BackendFitError(f"Unsupported covariance method: {method}")


def _whitening_matrix(
    covariance: np.ndarray,
    *,
    eigenvalue_floor: float,
    output_dim: int | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if eigenvalue_floor <= 0.0 or not math.isfinite(eigenvalue_floor):
        raise BackendFitError(f"eigenvalue_floor must be positive and finite, got {eigenvalue_floor}")
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.float64)
    dim = eigenvectors.shape[1]
    if output_dim is not None:
        if output_dim <= 0:
            raise BackendFitError("output_dim must be positive when provided.")
        dim = min(int(output_dim), dim)
    selected_values = np.maximum(eigenvalues[:dim], eigenvalue_floor)
    selected_vectors = eigenvectors[:, :dim]
    transform = selected_vectors / np.sqrt(selected_values)[None, :]
    condition = float(np.max(selected_values) / max(np.min(selected_values), eigenvalue_floor))
    return transform, {
        "eigenvalue_floor": float(eigenvalue_floor),
        "output_dim": int(dim),
        "raw_min_eigenvalue": float(np.min(eigenvalues)),
        "raw_max_eigenvalue": float(np.max(eigenvalues)),
        "condition_number_after_floor": condition,
    }


def _speaker_residuals(embeddings: np.ndarray, speakers: np.ndarray, *, min_utterances_per_speaker: int = 2) -> np.ndarray:
    residual_parts: list[np.ndarray] = []
    speaker_count = 0
    for speaker in np.unique(speakers):
        rows = embeddings[speakers == speaker]
        if rows.shape[0] < min_utterances_per_speaker:
            continue
        speaker_count += 1
        residual_parts.append(rows - np.mean(rows, axis=0, keepdims=True))
    if speaker_count < 2 or not residual_parts:
        raise BackendFitError(
            "Need at least two speakers with enough utterances for supervised covariance fitting."
        )
    residuals = np.vstack(residual_parts)
    if residuals.shape[0] < 2:
        raise BackendFitError("Not enough within-speaker residual rows.")
    return residuals


def _length_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, EPS)


@dataclass(frozen=True)
class SimilarityBackendModel:
    """Serializable fitted backend model."""

    metric: str
    extractor: str
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]

    def score(self, enroll: np.ndarray, test: np.ndarray) -> np.ndarray:
        enroll_arr, test_arr = validate_pairwise_inputs(enroll, test)
        if self.metric == "centered_cosine":
            mu = self.arrays["mean"]
            return cosine(enroll_arr - mu, test_arr - mu)
        if self.metric in {"whitened_cosine", "wccn_cosine"}:
            mu = self.arrays.get("mean", np.zeros((enroll_arr.shape[1],), dtype=np.float64))
            transform = self.arrays["transform"]
            return cosine((enroll_arr - mu) @ transform, (test_arr - mu) @ transform)
        if self.metric == "lda_cosine":
            mean = self.arrays["mean"]
            projection = self.arrays["projection"]
            left = _length_normalize((enroll_arr - mean) @ projection)
            right = _length_normalize((test_arr - mean) @ projection)
            return cosine(left, right)
        if self.metric == "neg_mahalanobis":
            diff = enroll_arr - test_arr
            if "precision_diag" in self.arrays:
                return -np.sum((diff**2) * self.arrays["precision_diag"], axis=1)
            precision = self.arrays["precision"]
            return -np.einsum("ij,jk,ik->i", diff, precision, diff)
        if self.metric == "asnorm_cosine":
            return _score_asnorm(enroll_arr, test_arr, self.arrays["cohort"], self.metadata)
        if self.metric == "plda_llr":
            raise RuntimeError("plda_llr artifact is marked not_run_unverified; no verified scorer is available.")
        raise RuntimeError(f"Unsupported backend metric: {self.metric}")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(self.arrays)
        metadata = dict(self.metadata)
        metadata.update({"metric": self.metric, "extractor": self.extractor})
        payload["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True), dtype=object)
        np.savez_compressed(path, **payload)


def load_backend_model(path: Path) -> SimilarityBackendModel:
    """Load a backend model saved by :meth:`SimilarityBackendModel.save`."""

    with np.load(path, allow_pickle=True) as payload:
        if "metadata_json" not in payload:
            raise ValueError(f"{path} does not contain metadata_json")
        metadata = json.loads(str(payload["metadata_json"].item()))
        arrays = {
            name: np.asarray(payload[name], dtype=np.float64)
            for name in payload.files
            if name != "metadata_json"
        }
    metric = str(metadata["metric"])
    extractor = str(metadata.get("extractor", "unknown"))
    return SimilarityBackendModel(metric=metric, extractor=extractor, arrays=arrays, metadata=metadata)


def fit_centered_cosine(
    embeddings: np.ndarray,
    *,
    extractor: str,
    fit_speakers: Iterable[str],
    test_speakers: Iterable[str],
) -> SimilarityBackendModel:
    values = _validate_fit_matrix(embeddings)
    metadata = {
        "status": "ok",
        "utterance_count": int(values.shape[0]),
        "dimension": int(values.shape[1]),
        **manifest_no_test_leakage(fit_speakers, test_speakers),
    }
    if metadata["test_speaker_intersection_count"]:
        raise BackendFitError("Fit speakers intersect test speakers.")
    return SimilarityBackendModel(
        metric="centered_cosine",
        extractor=extractor,
        arrays={"mean": np.mean(values, axis=0)},
        metadata=metadata,
    )


def fit_whitened_cosine(
    embeddings: np.ndarray,
    *,
    extractor: str,
    fit_speakers: Iterable[str],
    test_speakers: Iterable[str],
    covariance_method: str = "ledoit_wolf",
    eigenvalue_floor: float = 1.0e-5,
    output_dim: int | None = None,
) -> SimilarityBackendModel:
    values = _validate_fit_matrix(embeddings)
    mu = np.mean(values, axis=0)
    centered = values - mu
    cov, cov_meta = _covariance(centered, covariance_method)
    transform, transform_meta = _whitening_matrix(
        cov,
        eigenvalue_floor=eigenvalue_floor,
        output_dim=output_dim,
    )
    metadata = {
        "status": "ok",
        "utterance_count": int(values.shape[0]),
        "dimension": int(values.shape[1]),
        **cov_meta,
        **transform_meta,
        **manifest_no_test_leakage(fit_speakers, test_speakers),
    }
    if metadata["test_speaker_intersection_count"]:
        raise BackendFitError("Fit speakers intersect test speakers.")
    return SimilarityBackendModel(
        metric="whitened_cosine",
        extractor=extractor,
        arrays={"mean": mu, "transform": transform},
        metadata=metadata,
    )


def fit_wccn_cosine(
    embeddings: np.ndarray,
    speaker_labels: Sequence[str],
    *,
    extractor: str,
    fit_speakers: Iterable[str],
    test_speakers: Iterable[str],
    covariance_method: str = "ledoit_wolf",
    eigenvalue_floor: float = 1.0e-5,
    min_utterances_per_speaker: int = 2,
) -> SimilarityBackendModel:
    values = _validate_fit_matrix(embeddings)
    speakers = _speaker_array(speaker_labels, values.shape[0])
    residuals = _speaker_residuals(
        values,
        speakers,
        min_utterances_per_speaker=min_utterances_per_speaker,
    )
    cov, cov_meta = _covariance(residuals, covariance_method)
    transform, transform_meta = _whitening_matrix(
        cov,
        eigenvalue_floor=eigenvalue_floor,
        output_dim=None,
    )
    metadata = {
        "status": "ok",
        "utterance_count": int(values.shape[0]),
        "residual_count": int(residuals.shape[0]),
        "dimension": int(values.shape[1]),
        "min_utterances_per_speaker": int(min_utterances_per_speaker),
        **cov_meta,
        **transform_meta,
        **manifest_no_test_leakage(fit_speakers, test_speakers),
    }
    if metadata["test_speaker_intersection_count"]:
        raise BackendFitError("Fit speakers intersect test speakers.")
    return SimilarityBackendModel(
        metric="wccn_cosine",
        extractor=extractor,
        arrays={"mean": np.zeros(values.shape[1], dtype=np.float64), "transform": transform},
        metadata=metadata,
    )


def fit_lda_cosine(
    embeddings: np.ndarray,
    speaker_labels: Sequence[str],
    *,
    extractor: str,
    fit_speakers: Iterable[str],
    test_speakers: Iterable[str],
    output_dim: int = 128,
) -> SimilarityBackendModel:
    values = _validate_fit_matrix(embeddings)
    speakers = _speaker_array(speaker_labels, values.shape[0])
    unique_speakers = np.unique(speakers)
    max_dim = min(values.shape[1], unique_speakers.shape[0] - 1)
    if max_dim <= 0:
        raise BackendFitError("LDA needs at least two speakers.")
    dim = min(int(output_dim), int(max_dim))
    lda = LinearDiscriminantAnalysis(solver="svd")
    lda.fit(values, speakers)
    projection = np.asarray(lda.scalings_[:, :dim], dtype=np.float64)
    mean = np.asarray(getattr(lda, "xbar_", np.mean(values, axis=0)), dtype=np.float64)
    metadata = {
        "status": "ok",
        "utterance_count": int(values.shape[0]),
        "dimension": int(values.shape[1]),
        "speaker_count": int(unique_speakers.shape[0]),
        "output_dim": int(dim),
        **manifest_no_test_leakage(fit_speakers, test_speakers),
    }
    if metadata["test_speaker_intersection_count"]:
        raise BackendFitError("Fit speakers intersect test speakers.")
    return SimilarityBackendModel(
        metric="lda_cosine",
        extractor=extractor,
        arrays={"mean": mean, "projection": projection},
        metadata=metadata,
    )


def fit_neg_mahalanobis(
    embeddings: np.ndarray,
    speaker_labels: Sequence[str] | None,
    *,
    extractor: str,
    fit_speakers: Iterable[str],
    test_speakers: Iterable[str],
    covariance_method: str = "ledoit_wolf",
    eigenvalue_floor: float = 1.0e-5,
    diagonal: bool = False,
) -> SimilarityBackendModel:
    values = _validate_fit_matrix(embeddings)
    if speaker_labels is None:
        cov_values = values - np.mean(values, axis=0, keepdims=True)
        covariance_source = "global"
    else:
        speakers = _speaker_array(speaker_labels, values.shape[0])
        cov_values = _speaker_residuals(values, speakers)
        covariance_source = "within_speaker"
    cov, cov_meta = _covariance(cov_values, covariance_method)
    eigenvalues = np.linalg.eigvalsh(cov)
    condition = float(np.max(eigenvalues) / max(float(np.min(eigenvalues)), eigenvalue_floor))
    metadata = {
        "status": "ok",
        "utterance_count": int(values.shape[0]),
        "dimension": int(values.shape[1]),
        "covariance_source": covariance_source,
        "diagonal": bool(diagonal),
        "eigenvalue_floor": float(eigenvalue_floor),
        "raw_condition_number": condition,
        **cov_meta,
        **manifest_no_test_leakage(fit_speakers, test_speakers),
    }
    if metadata["test_speaker_intersection_count"]:
        raise BackendFitError("Fit speakers intersect test speakers.")
    if diagonal:
        diag = np.maximum(np.diag(cov), eigenvalue_floor)
        arrays = {"precision_diag": 1.0 / diag}
        metadata["fallback"] = "diagonal"
    else:
        eigvals, eigvecs = np.linalg.eigh(cov)
        floored = np.maximum(eigvals, eigenvalue_floor)
        precision = (eigvecs / floored[None, :]) @ eigvecs.T
        arrays = {"precision": precision}
        metadata["fallback"] = "none"
    return SimilarityBackendModel(
        metric="neg_mahalanobis",
        extractor=extractor,
        arrays=arrays,
        metadata=metadata,
    )


def fit_plda_placeholder(
    *,
    extractor: str,
    fit_speakers: Iterable[str],
    test_speakers: Iterable[str],
    blocked_reason: str = "No trusted verified PLDA implementation configured.",
) -> SimilarityBackendModel:
    metadata = {
        "status": "not_run_unverified",
        "blocked_reason": blocked_reason,
        **manifest_no_test_leakage(fit_speakers, test_speakers),
    }
    return SimilarityBackendModel(
        metric="plda_llr",
        extractor=extractor,
        arrays={"placeholder": np.asarray([0.0], dtype=np.float64)},
        metadata=metadata,
    )


def _stable_order_key(seed: str, *parts: object) -> str:
    payload = "\x1f".join([seed, *(str(part) for part in parts)]).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_asnorm_cohort(
    embeddings: np.ndarray,
    speaker_labels: Sequence[str],
    utterance_ids: Sequence[str],
    *,
    max_cohort_size: int = 1000,
    per_speaker: int = 1,
    seed: str = "asnorm_cohort_v1",
) -> tuple[np.ndarray, dict[str, Any]]:
    values = _validate_fit_matrix(embeddings)
    speakers = _speaker_array(speaker_labels, values.shape[0])
    utterances = np.asarray([str(value) for value in utterance_ids], dtype=object)
    if utterances.shape[0] != values.shape[0]:
        raise BackendFitError("utterance_ids length does not match embeddings rows.")
    selected: list[int] = []
    for speaker in sorted(np.unique(speakers)):
        indices = np.flatnonzero(speakers == speaker)
        ranked = sorted(indices.tolist(), key=lambda idx: _stable_order_key(seed, speaker, utterances[idx]))
        selected.extend(ranked[:per_speaker])
    selected = sorted(selected, key=lambda idx: _stable_order_key(seed, speakers[idx], utterances[idx]))
    if max_cohort_size > 0:
        selected = selected[:max_cohort_size]
    if not selected:
        raise BackendFitError("No cohort utterances selected.")
    cohort = values[np.asarray(selected, dtype=np.int64)]
    manifest = {
        "cohort_size": int(cohort.shape[0]),
        "per_speaker": int(per_speaker),
        "max_cohort_size": int(max_cohort_size),
        "cohort_speaker_count": int(len({str(speakers[idx]) for idx in selected})),
        "cohort_speaker_hash": stable_speaker_hash(str(speakers[idx]) for idx in selected),
        "cohort_utterance_hash": hashlib.sha256(
            "\n".join(sorted(str(utterances[idx]) for idx in selected)).encode("utf-8")
        ).hexdigest(),
        "seed": seed,
    }
    return cohort, manifest


def fit_asnorm_cosine(
    embeddings: np.ndarray,
    speaker_labels: Sequence[str],
    utterance_ids: Sequence[str],
    *,
    extractor: str,
    fit_speakers: Iterable[str],
    test_speakers: Iterable[str],
    max_cohort_size: int = 1000,
    per_speaker: int = 1,
    top_k: int = 100,
    sigma_floor: float = 1.0e-4,
    seed: str = "asnorm_cohort_v1",
) -> SimilarityBackendModel:
    values = _validate_fit_matrix(embeddings)
    cohort, cohort_manifest = select_asnorm_cohort(
        values,
        speaker_labels,
        utterance_ids,
        max_cohort_size=max_cohort_size,
        per_speaker=per_speaker,
        seed=seed,
    )
    if top_k <= 0:
        raise BackendFitError("top_k must be positive.")
    metadata = {
        "status": "ok",
        "utterance_count": int(values.shape[0]),
        "dimension": int(values.shape[1]),
        "top_k": int(min(top_k, cohort.shape[0])),
        "sigma_floor": float(sigma_floor),
        **cohort_manifest,
        **manifest_no_test_leakage(fit_speakers, test_speakers),
    }
    if metadata["test_speaker_intersection_count"]:
        raise BackendFitError("Fit speakers intersect test speakers.")
    return SimilarityBackendModel(
        metric="asnorm_cosine",
        extractor=extractor,
        arrays={"cohort": cohort},
        metadata=metadata,
    )


def _topk_stats_against_cohort(
    values: np.ndarray,
    cohort: np.ndarray,
    *,
    top_k: int,
    batch_size: int,
    sigma_floor: float,
) -> tuple[np.ndarray, np.ndarray]:
    left = _length_normalize(values)
    right = _length_normalize(cohort)
    k = min(int(top_k), right.shape[0])
    means: list[np.ndarray] = []
    stds: list[np.ndarray] = []
    for start in range(0, left.shape[0], batch_size):
        scores = left[start : start + batch_size] @ right.T
        if k < scores.shape[1]:
            top = np.partition(scores, scores.shape[1] - k, axis=1)[:, -k:]
        else:
            top = scores
        means.append(np.mean(top, axis=1))
        stds.append(np.maximum(np.std(top, axis=1), sigma_floor))
    return np.concatenate(means), np.concatenate(stds)


def _score_asnorm(enroll: np.ndarray, test: np.ndarray, cohort: np.ndarray, metadata: Mapping[str, Any]) -> np.ndarray:
    raw = cosine(enroll, test)
    batch_size = int(metadata.get("cohort_batch_size", 8192))
    top_k = int(metadata.get("top_k", 100))
    sigma_floor = float(metadata.get("sigma_floor", 1.0e-4))
    enroll_mu, enroll_sigma = _topk_stats_against_cohort(
        enroll,
        cohort,
        top_k=top_k,
        batch_size=batch_size,
        sigma_floor=sigma_floor,
    )
    test_mu, test_sigma = _topk_stats_against_cohort(
        test,
        cohort,
        top_k=top_k,
        batch_size=batch_size,
        sigma_floor=sigma_floor,
    )
    return 0.5 * ((raw - enroll_mu) / enroll_sigma + (raw - test_mu) / test_sigma)


def asnorm_stats_for_model(model: SimilarityBackendModel, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return AS-Norm cohort top-k mean/std for each input row."""

    if model.metric != "asnorm_cosine":
        raise ValueError(f"Expected asnorm_cosine model, got {model.metric}")
    values_arr = np.asarray(values, dtype=np.float64)
    if values_arr.ndim == 1:
        values_arr = values_arr.reshape(1, -1)
    if values_arr.ndim != 2:
        raise ValueError(f"Expected 1D or 2D AS-Norm input, got {values_arr.ndim}D")
    cohort = model.arrays["cohort"]
    if values_arr.shape[1] != cohort.shape[1]:
        raise ValueError(f"AS-Norm dimension mismatch: values={values_arr.shape} cohort={cohort.shape}")
    return _topk_stats_against_cohort(
        values_arr,
        cohort,
        top_k=int(model.metadata.get("top_k", 100)),
        batch_size=int(model.metadata.get("cohort_batch_size", 8192)),
        sigma_floor=float(model.metadata.get("sigma_floor", 1.0e-4)),
    )


def asnorm_scores_from_stats(
    enroll: np.ndarray,
    test: np.ndarray,
    enroll_mu: np.ndarray,
    enroll_sigma: np.ndarray,
    test_mu: np.ndarray,
    test_sigma: np.ndarray,
) -> np.ndarray:
    """Score AS-Norm using precomputed per-utterance cohort statistics."""

    enroll_arr, test_arr = validate_pairwise_inputs(enroll, test)
    raw = cosine(enroll_arr, test_arr)
    enroll_mu_arr = np.asarray(enroll_mu, dtype=np.float64).reshape(-1)
    enroll_sigma_arr = np.asarray(enroll_sigma, dtype=np.float64).reshape(-1)
    test_mu_arr = np.asarray(test_mu, dtype=np.float64).reshape(-1)
    test_sigma_arr = np.asarray(test_sigma, dtype=np.float64).reshape(-1)
    expected = raw.shape
    for name, values in (
        ("enroll_mu", enroll_mu_arr),
        ("enroll_sigma", enroll_sigma_arr),
        ("test_mu", test_mu_arr),
        ("test_sigma", test_sigma_arr),
    ):
        if values.shape != expected:
            raise ValueError(f"{name} shape={values.shape} does not match scores shape={expected}")
    return 0.5 * ((raw - enroll_mu_arr) / enroll_sigma_arr + (raw - test_mu_arr) / test_sigma_arr)


def score_backend_metrics(
    models: Mapping[str, SimilarityBackendModel],
    enroll: np.ndarray,
    test: np.ndarray,
    metrics: Sequence[str],
    *,
    allow_blocked: bool = False,
) -> dict[str, np.ndarray]:
    """Score a set of fitted backend metrics with loaded artifacts."""

    enroll_arr, test_arr = validate_pairwise_inputs(enroll, test)
    output: dict[str, np.ndarray] = {}
    for metric in metrics:
        if metric not in BACKEND_METRICS:
            continue
        if metric not in models:
            raise FileNotFoundError(f"No backend model loaded for metric={metric}")
        model = models[metric]
        if model.metadata.get("status") != "ok":
            if allow_blocked:
                output[metric] = np.full(enroll_arr.shape[0], np.nan, dtype=np.float64)
                continue
            raise RuntimeError(
                f"Backend metric {metric} is not runnable: "
                f"{model.metadata.get('status')} {model.metadata.get('blocked_reason', '')}"
            )
        output[metric] = model.score(enroll_arr, test_arr)
    return output
