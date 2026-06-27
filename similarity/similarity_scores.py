#!/usr/bin/env python3
"""Vectorized trial-level similarity and score-context features.

All similarity scores follow the convention that larger values are more
target-like. Distance metrics are therefore negated.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np


EPS = 1.0e-12


@dataclass(frozen=True)
class SimilarityScoreSpec:
    """Metadata for a score exposed by the similarity-score registry."""

    name: str
    family: str
    requires_fit: bool
    diagnostic_only: bool
    higher_is_target: bool
    symmetric: bool
    description: str


SCORE_SPECS: dict[str, SimilarityScoreSpec] = {
    "cosine": SimilarityScoreSpec(
        name="cosine",
        family="geometric",
        requires_fit=False,
        diagnostic_only=False,
        higher_is_target=True,
        symmetric=True,
        description="dot(e,t)/(||e||||t||); baseline raw cosine score",
    ),
    "dot_raw": SimilarityScoreSpec(
        name="dot_raw",
        family="geometric_raw",
        requires_fit=False,
        diagnostic_only=False,
        higher_is_target=True,
        symmetric=True,
        description="Unnormalized dot product, preserving embedding norm information",
    ),
    "neg_l2_raw": SimilarityScoreSpec(
        name="neg_l2_raw",
        family="geometric_diagnostic",
        requires_fit=False,
        diagnostic_only=True,
        higher_is_target=True,
        symmetric=True,
        description="Negative raw Euclidean distance",
    ),
    "neg_sq_l2_raw": SimilarityScoreSpec(
        name="neg_sq_l2_raw",
        family="geometric_diagnostic",
        requires_fit=False,
        diagnostic_only=True,
        higher_is_target=True,
        symmetric=True,
        description="Negative raw squared Euclidean distance",
    ),
    "neg_l1_raw": SimilarityScoreSpec(
        name="neg_l1_raw",
        family="geometric_raw",
        requires_fit=False,
        diagnostic_only=False,
        higher_is_target=True,
        symmetric=True,
        description="Negative raw L1 distance",
    ),
    "pearson_corr": SimilarityScoreSpec(
        name="pearson_corr",
        family="geometric_raw",
        requires_fit=False,
        diagnostic_only=False,
        higher_is_target=True,
        symmetric=True,
        description="Correlation after subtracting each vector's own coordinate mean",
    ),
    "angular_similarity_diagnostic": SimilarityScoreSpec(
        name="angular_similarity_diagnostic",
        family="monotonic_diagnostic",
        requires_fit=False,
        diagnostic_only=True,
        higher_is_target=True,
        symmetric=True,
        description="1 - arccos(cosine)/pi; monotonic transform of cosine",
    ),
    "rbf_similarity_diagnostic": SimilarityScoreSpec(
        name="rbf_similarity_diagnostic",
        family="monotonic_diagnostic",
        requires_fit=False,
        diagnostic_only=True,
        higher_is_target=True,
        symmetric=True,
        description="exp(-gamma ||e-t||^2); monotonic transform of squared L2 for fixed gamma",
    ),
    "log_norm_enroll": SimilarityScoreSpec(
        name="log_norm_enroll",
        family="norm_context",
        requires_fit=False,
        diagnostic_only=False,
        higher_is_target=True,
        symmetric=False,
        description="log(||enroll|| + eps), score-quality context not speaker similarity",
    ),
    "log_norm_test": SimilarityScoreSpec(
        name="log_norm_test",
        family="norm_context",
        requires_fit=False,
        diagnostic_only=False,
        higher_is_target=True,
        symmetric=False,
        description="log(||test|| + eps), score-quality context not speaker similarity",
    ),
    "log_norm_product": SimilarityScoreSpec(
        name="log_norm_product",
        family="norm_context",
        requires_fit=False,
        diagnostic_only=False,
        higher_is_target=True,
        symmetric=True,
        description="log(||enroll|| + eps) + log(||test|| + eps)",
    ),
    "abs_log_norm_ratio": SimilarityScoreSpec(
        name="abs_log_norm_ratio",
        family="norm_context",
        requires_fit=False,
        diagnostic_only=False,
        higher_is_target=True,
        symmetric=True,
        description="abs(log(||enroll|| + eps) - log(||test|| + eps))",
    ),
    "min_log_norm": SimilarityScoreSpec(
        name="min_log_norm",
        family="norm_context",
        requires_fit=False,
        diagnostic_only=False,
        higher_is_target=True,
        symmetric=True,
        description="min log norm of the pair",
    ),
    "max_log_norm": SimilarityScoreSpec(
        name="max_log_norm",
        family="norm_context",
        requires_fit=False,
        diagnostic_only=False,
        higher_is_target=True,
        symmetric=True,
        description="max log norm of the pair",
    ),
    "centered_cosine": SimilarityScoreSpec(
        name="centered_cosine",
        family="backend_unsupervised",
        requires_fit=True,
        diagnostic_only=False,
        higher_is_target=True,
        symmetric=True,
        description="Cosine after subtracting extractor-specific calibration mean",
    ),
    "whitened_cosine": SimilarityScoreSpec(
        name="whitened_cosine",
        family="backend_unsupervised",
        requires_fit=True,
        diagnostic_only=False,
        higher_is_target=True,
        symmetric=True,
        description="Cosine after extractor-specific shrinkage whitening",
    ),
    "wccn_cosine": SimilarityScoreSpec(
        name="wccn_cosine",
        family="backend_supervised",
        requires_fit=True,
        diagnostic_only=False,
        higher_is_target=True,
        symmetric=True,
        description="Cosine after within-class covariance normalization",
    ),
    "lda_cosine": SimilarityScoreSpec(
        name="lda_cosine",
        family="backend_supervised",
        requires_fit=True,
        diagnostic_only=False,
        higher_is_target=True,
        symmetric=True,
        description="Cosine after supervised LDA projection and length normalization",
    ),
    "neg_mahalanobis": SimilarityScoreSpec(
        name="neg_mahalanobis",
        family="backend_supervised",
        requires_fit=True,
        diagnostic_only=False,
        higher_is_target=True,
        symmetric=True,
        description="Negative Mahalanobis distance using calibration-only covariance",
    ),
    "plda_llr": SimilarityScoreSpec(
        name="plda_llr",
        family="backend_supervised",
        requires_fit=True,
        diagnostic_only=False,
        higher_is_target=True,
        symmetric=True,
        description="Verified PLDA log-likelihood ratio if a trusted implementation is available",
    ),
    "asnorm_cosine": SimilarityScoreSpec(
        name="asnorm_cosine",
        family="cohort_normalized",
        requires_fit=True,
        diagnostic_only=False,
        higher_is_target=True,
        symmetric=True,
        description="Adaptive s-norm normalized cosine with calibration-only cohort",
    ),
}


DEFAULT_METRICS = ("cosine",)
NORM_CONTEXT_METRICS = (
    "log_norm_enroll",
    "log_norm_test",
    "log_norm_product",
    "abs_log_norm_ratio",
    "min_log_norm",
    "max_log_norm",
)


def available_metrics(*, include_diagnostic: bool = True) -> list[str]:
    """Return registered metric names in stable order."""

    return [
        name
        for name, spec in SCORE_SPECS.items()
        if include_diagnostic or not spec.diagnostic_only
    ]


def metric_config_hash(metrics: Sequence[str], options: Mapping[str, object] | None = None) -> str:
    """Create a stable short hash for metric names and relevant options."""

    payload = {
        "metrics": list(metrics),
        "options": dict(options or {}),
        "registry": {name: asdict(SCORE_SPECS[name]) for name in sorted(set(metrics)) if name in SCORE_SPECS},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def parse_metric_list(value: str | Sequence[str] | None) -> list[str]:
    """Parse a comma-separated metric list and validate registry membership."""

    if value is None:
        metrics = list(DEFAULT_METRICS)
    elif isinstance(value, str):
        metrics = [item.strip() for item in value.split(",") if item.strip()]
    else:
        metrics = [str(item).strip() for item in value if str(item).strip()]
    if not metrics:
        raise ValueError("At least one similarity metric is required.")
    unknown = sorted(set(metrics) - set(SCORE_SPECS))
    if unknown:
        raise ValueError(f"Unknown similarity metric(s): {unknown}. Available: {available_metrics()}")
    return metrics


def validate_pairwise_inputs(enroll: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return float64 2D arrays with matching shape."""

    enroll_arr = np.asarray(enroll, dtype=np.float64)
    test_arr = np.asarray(test, dtype=np.float64)
    if enroll_arr.ndim == 1:
        enroll_arr = enroll_arr.reshape(1, -1)
    if test_arr.ndim == 1:
        test_arr = test_arr.reshape(1, -1)
    if enroll_arr.ndim != 2 or test_arr.ndim != 2:
        raise ValueError(
            f"Expected 1D or 2D arrays for enroll/test; got {enroll_arr.ndim}D and {test_arr.ndim}D."
        )
    if enroll_arr.shape != test_arr.shape:
        raise ValueError(f"Shape mismatch: enroll={enroll_arr.shape} test={test_arr.shape}")
    if enroll_arr.shape[1] == 0:
        raise ValueError("Embedding dimension must be nonzero.")
    return enroll_arr, test_arr


def _row_norm(values: np.ndarray) -> np.ndarray:
    return np.linalg.norm(values, axis=1)


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    output = np.full(numerator.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(numerator) & np.isfinite(denominator) & (np.abs(denominator) > EPS)
    output[valid] = numerator[valid] / denominator[valid]
    return output


def cosine(enroll: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Compute cosine similarity. Undefined zero-norm rows return NaN."""

    numerator = np.einsum("ij,ij->i", enroll, test)
    denominator = _row_norm(enroll) * _row_norm(test)
    return _safe_divide(numerator, denominator)


def pearson_corr(enroll: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Compute per-row Pearson correlation after vector-internal centering."""

    enroll_centered = enroll - np.mean(enroll, axis=1, keepdims=True)
    test_centered = test - np.mean(test, axis=1, keepdims=True)
    numerator = np.einsum("ij,ij->i", enroll_centered, test_centered)
    denominator = _row_norm(enroll_centered) * _row_norm(test_centered)
    return _safe_divide(numerator, denominator)


def compute_pairwise_scores(
    enroll: np.ndarray,
    test: np.ndarray,
    metrics: Sequence[str],
    *,
    rbf_gamma: float = 1.0,
) -> dict[str, np.ndarray]:
    """Compute selected pairwise scores for aligned enrollment/test arrays.

    Args:
        enroll: Array shaped ``(n_pairs, dim)`` or a single vector ``(dim,)``.
        test: Array with the same shape as ``enroll``.
        metrics: Registered metric names.
        rbf_gamma: Fixed gamma for ``rbf_similarity_diagnostic``.

    Returns:
        Mapping from metric name to one score per input row.
    """

    metric_names = parse_metric_list(metrics)
    fitted = [name for name in metric_names if SCORE_SPECS[name].requires_fit]
    if fitted:
        raise ValueError(
            "compute_pairwise_scores only handles no-fit scores. "
            f"Use similarity_backend_models for fitted metrics: {fitted}"
        )
    if "rbf_similarity_diagnostic" in metric_names and (not math.isfinite(rbf_gamma) or rbf_gamma <= 0.0):
        raise ValueError(f"rbf_gamma must be positive and finite, got {rbf_gamma!r}")

    enroll_arr, test_arr = validate_pairwise_inputs(enroll, test)
    diff = enroll_arr - test_arr
    squared_l2 = np.einsum("ij,ij->i", diff, diff)
    enroll_norm = _row_norm(enroll_arr)
    test_norm = _row_norm(test_arr)
    log_norm_enroll = np.log(np.maximum(enroll_norm, 0.0) + EPS)
    log_norm_test = np.log(np.maximum(test_norm, 0.0) + EPS)

    cache: dict[str, np.ndarray] = {}
    output: dict[str, np.ndarray] = {}

    for name in metric_names:
        if name == "cosine":
            values = cache.setdefault("cosine", cosine(enroll_arr, test_arr))
        elif name == "dot_raw":
            values = np.einsum("ij,ij->i", enroll_arr, test_arr)
        elif name == "neg_l2_raw":
            values = -np.sqrt(np.maximum(squared_l2, 0.0))
        elif name == "neg_sq_l2_raw":
            values = -squared_l2
        elif name == "neg_l1_raw":
            values = -np.sum(np.abs(diff), axis=1)
        elif name == "pearson_corr":
            values = pearson_corr(enroll_arr, test_arr)
        elif name == "angular_similarity_diagnostic":
            cos_values = cache.setdefault("cosine", cosine(enroll_arr, test_arr))
            values = 1.0 - np.arccos(np.clip(cos_values, -1.0, 1.0)) / math.pi
        elif name == "rbf_similarity_diagnostic":
            values = np.exp(-float(rbf_gamma) * squared_l2)
        elif name == "log_norm_enroll":
            values = log_norm_enroll
        elif name == "log_norm_test":
            values = log_norm_test
        elif name == "log_norm_product":
            values = log_norm_enroll + log_norm_test
        elif name == "abs_log_norm_ratio":
            values = np.abs(log_norm_enroll - log_norm_test)
        elif name == "min_log_norm":
            values = np.minimum(log_norm_enroll, log_norm_test)
        elif name == "max_log_norm":
            values = np.maximum(log_norm_enroll, log_norm_test)
        else:
            raise ValueError(f"Unhandled similarity metric: {name}")
        output[name] = np.asarray(values, dtype=np.float64)
    return output


def finite_audit(scores: Mapping[str, np.ndarray]) -> dict[str, dict[str, int]]:
    """Count finite, NaN, and infinite values for each score array."""

    audit: dict[str, dict[str, int]] = {}
    for name, values in scores.items():
        arr = np.asarray(values)
        audit[name] = {
            "count": int(arr.size),
            "finite": int(np.isfinite(arr).sum()),
            "nan": int(np.isnan(arr).sum()),
            "posinf": int(np.isposinf(arr).sum()),
            "neginf": int(np.isneginf(arr).sum()),
        }
    return audit


def trainable_parameter_count(input_dim: int, hidden_dim: int = 128) -> int:
    """Parameter count for the repo's two-hidden-layer MLP calibrator."""

    if input_dim <= 0 or hidden_dim <= 0:
        raise ValueError("input_dim and hidden_dim must be positive.")
    first = input_dim * hidden_dim + hidden_dim
    first_layer_norm = 2 * hidden_dim
    second = hidden_dim * hidden_dim + hidden_dim
    second_layer_norm = 2 * hidden_dim
    output = hidden_dim * 1 + 1
    return int(first + first_layer_norm + second + second_layer_norm + output)
