from __future__ import annotations

import math

import numpy as np


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def confidence_from_llr(llrs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    posterior_target = sigmoid(llrs)
    confidence = np.maximum(posterior_target, 1.0 - posterior_target)
    return posterior_target, confidence


def confidence_threshold(confidence: np.ndarray, coverage: float) -> float:
    if not 0.0 < coverage <= 1.0:
        raise ValueError("coverage must be in (0, 1]")
    return float(np.quantile(confidence, 1.0 - coverage))


def decisions_from_confidence(
    posterior_target: np.ndarray,
    confidence: np.ndarray,
    threshold: float | None,
) -> np.ndarray:
    decisions = np.where(
        posterior_target >= 0.5, "target", "nontarget"
    ).astype(object)
    if threshold is not None:
        decisions[confidence < threshold] = "reject"
    return decisions


def decision_metrics(
    labels: np.ndarray, decisions: np.ndarray
) -> dict[str, float | int]:
    labels_bool = labels.astype(bool)
    target_decision = decisions == "target"
    nontarget_decision = decisions == "nontarget"
    rejected = decisions == "reject"
    accepted = ~rejected

    total_n = int(labels.shape[0])
    target_n = int(np.sum(labels_bool))
    nontarget_n = total_n - target_n
    accepted_n = int(np.sum(accepted))
    correct_accepted = int(
        np.sum(
            (target_decision & labels_bool)
            | (nontarget_decision & ~labels_bool)
        )
    )

    return {
        "n": total_n,
        "target_n": target_n,
        "nontarget_n": nontarget_n,
        "accepted_n": accepted_n,
        "rejected_n": int(np.sum(rejected)),
        "coverage": accepted_n / total_n if total_n else math.nan,
        "accuracy": correct_accepted / total_n if total_n else math.nan,
        "effective_acc": (
            correct_accepted / total_n if total_n else math.nan
        ),
        "covered_acc": (
            correct_accepted / accepted_n if accepted_n else math.nan
        ),
        "FAR": (
            float(np.sum(target_decision & ~labels_bool) / nontarget_n)
            if nontarget_n
            else math.nan
        ),
        "FRR": (
            float(
                np.sum((nontarget_decision | rejected) & labels_bool)
                / target_n
            )
            if target_n
            else math.nan
        ),
    }
