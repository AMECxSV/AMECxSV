#!/usr/bin/env python3
"""Shared utilities for AMEC external-baseline experiments."""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = PROJECT_ROOT / "external"
DEFAULT_CONFIG = EXTERNAL_ROOT / "config" / "baselines.yaml"
PRIORS = {"p001": 0.001, "p01": 0.01}

TRIAL_COLUMNS = [
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
]

SCORE_OUTPUT_COLUMNS = [
    *TRIAL_COLUMNS,
    "score",
    "source_system",
    "checkpoint_id",
    "run_id",
    "timestamp_utc",
    "git_commit",
]


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def make_run_id(source_system: str) -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{source_system}_{stamp}_{uuid.uuid4().hex[:8]}"


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""
    if value in {"null", "None", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def _load_yaml_fallback(path: Path) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if value.strip():
            parent[key] = _parse_scalar(value)
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
    return root


def load_config(path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        return loaded or {}
    except Exception:
        return _load_yaml_fallback(path)


def config_path(config: dict[str, Any], name: str, default: str | Path | None = None) -> Path | None:
    value = (config.get("paths") or {}).get(name, default)
    if value is None:
        return None
    return resolve_path(value)


def resolve_path(value: str | Path) -> Path:
    text = str(value)
    text = text.replace("${PROJECT_ROOT}", str(PROJECT_ROOT))
    text = text.replace("${EXTERNAL_BASELINES_ROOT}", str(EXTERNAL_ROOT))
    text = os.path.expandvars(os.path.expanduser(text))
    path = Path(text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def ensure_columns(frame: pd.DataFrame, required: Iterable[str], *, context: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{context} is missing columns: {missing}")


def read_table(path: Path | str, columns: list[str] | None = None, nrows: int | None = None) -> pd.DataFrame:
    path = resolve_path(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, columns=columns)
    if suffix in {".csv", ".txt", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        if columns is None:
            return pd.read_csv(path, sep=sep, nrows=nrows)
        header = pd.read_csv(path, sep=sep, nrows=0).columns.tolist()
        usecols = [column for column in columns if column in header]
        return pd.read_csv(path, sep=sep, usecols=usecols, nrows=nrows)
    raise ValueError(f"Unsupported table format: {path}")


def write_table(frame: pd.DataFrame, path: Path | str) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
        return
    if suffix in {".csv", ".txt"}:
        frame.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported output table format: {path}")


def write_json(data: dict[str, Any], path: Path | str) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def label_array(frame: pd.DataFrame) -> np.ndarray:
    if "label" in frame.columns:
        values = frame["label"]
    elif "target" in frame.columns:
        values = frame["target"]
    else:
        raise ValueError("Expected a label or target column")

    if pd.api.types.is_numeric_dtype(values):
        return values.astype(np.int8).to_numpy(copy=False)
    normalized = values.astype(str).str.lower().str.strip()
    return normalized.isin({"1", "true", "target", "same", "same_speaker"}).astype(np.int8).to_numpy()


def split_frame(frame: pd.DataFrame, split: str | None) -> pd.DataFrame:
    if split is None or "split" not in frame.columns:
        return frame.reset_index(drop=True)
    return frame[frame["split"].astype(str) == str(split)].reset_index(drop=True)


def load_trial_table(path: Path | str, split: str | None = None) -> pd.DataFrame:
    frame = read_table(path)
    ensure_columns(frame, ["trial_id", "enroll_utt", "test_utt", "enroll_speaker", "test_speaker"], context=str(path))
    if "label" not in frame.columns and "target" not in frame.columns:
        raise ValueError(f"{path} must contain label or target")
    return split_frame(frame, split)


def pair_key(frame: pd.DataFrame) -> pd.Series:
    ensure_columns(frame, ["enroll_utt", "test_utt"], context="pair key input")
    return frame["enroll_utt"].astype(str) + "||" + frame["test_utt"].astype(str)


def align_scores_to_trials(
    trials: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    score_column: str,
    match_key: str,
    source_system: str,
    checkpoint_id: str,
    run_id: str,
    allow_partial: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    ensure_columns(scores, [score_column], context="score file")
    if match_key == "trial_id":
        ensure_columns(trials, ["trial_id"], context="trial table")
        ensure_columns(scores, ["trial_id"], context="score file")
        trial_key = trials["trial_id"].astype(str)
        score_key = scores["trial_id"].astype(str)
        key_name = "_align_key"
    elif match_key == "utterance_pair":
        trial_key = pair_key(trials)
        score_key = pair_key(scores)
        key_name = "_align_key"
    else:
        raise ValueError("--match-key must be trial_id or utterance_pair")

    trial_work = trials.copy()
    score_work = scores.copy()
    if "score" in trial_work.columns:
        trial_work = trial_work.drop(columns=["score"])
    trial_work[key_name] = trial_key
    score_work[key_name] = score_key

    duplicate_trials = int(trial_work[key_name].duplicated().sum())
    duplicate_scores = int(score_work[key_name].duplicated().sum())
    if duplicate_trials:
        raise ValueError(f"Trial table has {duplicate_trials} duplicate alignment keys")
    if duplicate_scores:
        raise ValueError(f"Score table has {duplicate_scores} duplicate alignment keys")

    score_small = score_work[[key_name, score_column]].rename(columns={score_column: "score"})
    merged = trial_work.merge(score_small, on=key_name, how="left", validate="one_to_one")
    missing_mask = merged["score"].isna()

    trial_keys = set(trial_work[key_name].astype(str))
    score_keys = set(score_work[key_name].astype(str))
    missing_keys = sorted(trial_keys - score_keys)
    extra_keys = sorted(score_keys - trial_keys)
    mismatch = {
        "match_key": match_key,
        "trial_count": int(len(trial_work)),
        "score_count": int(len(score_work)),
        "missing_score_count": int(len(missing_keys)),
        "extra_score_count": int(len(extra_keys)),
        "missing_score_examples": missing_keys[:25],
        "extra_score_examples": extra_keys[:25],
    }
    if missing_keys and not allow_partial:
        raise ValueError(
            f"Score file is missing {len(missing_keys)} required trials. "
            "Use --allow-partial only for diagnostic runs."
        )
    if allow_partial:
        merged = merged[~missing_mask].copy()

    timestamp = utc_timestamp()
    commit = git_commit()
    merged["source_system"] = source_system
    merged["checkpoint_id"] = checkpoint_id
    merged["run_id"] = run_id
    merged["timestamp_utc"] = timestamp
    merged["git_commit"] = commit
    merged = merged.drop(columns=[key_name])

    for column in SCORE_OUTPUT_COLUMNS:
        if column not in merged.columns:
            merged[column] = ""
    return merged[SCORE_OUTPUT_COLUMNS].copy(), mismatch


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


def classification_metrics(llrs: np.ndarray | None, labels: np.ndarray, threshold: float = 0.0) -> dict[str, float]:
    if llrs is None:
        return {"accuracy": math.nan, "precision": math.nan, "recall": math.nan, "f1": math.nan}
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


def metric_block(scores: np.ndarray, labels: np.ndarray, llrs: np.ndarray | None = None) -> dict[str, float | int]:
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
    row.update({"eer_pct": 100.0 * eer_value, "eer_threshold": eer_threshold})
    row["Cllr"] = math.nan if llrs is None else cllr(llrs, labels)
    for suffix, prior in PRIORS.items():
        row[f"minDCF_{suffix}"] = min_dcf(scores, labels, prior)
        row[f"actDCF_{suffix}"] = math.nan if llrs is None else act_dcf(llrs, labels, prior)
    row.update(classification_metrics(llrs, labels))
    return row


def confidence_from_llr(llrs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    posterior_target = 1.0 / (1.0 + np.exp(-llrs))
    confidence = np.maximum(posterior_target, 1.0 - posterior_target)
    return posterior_target, confidence


def metric_rows(
    frame: pd.DataFrame,
    *,
    score_column: str = "score",
    llr_column: str | None = None,
    source_system: str = "",
    run_id: str = "",
    score_kind: str = "raw_score",
    include_breakdowns: bool = True,
) -> list[dict[str, Any]]:
    ensure_columns(frame, [score_column], context="metric input")
    labels = label_array(frame)
    scores = pd.to_numeric(frame[score_column], errors="coerce").to_numpy(dtype=np.float64)
    llrs = None
    if llr_column:
        ensure_columns(frame, [llr_column], context="metric input")
        llrs = pd.to_numeric(frame[llr_column], errors="coerce").to_numpy(dtype=np.float64)

    valid = np.isfinite(scores)
    if llrs is not None:
        valid &= np.isfinite(llrs)
    labels = labels[valid]
    scores = scores[valid]
    llrs = None if llrs is None else llrs[valid]
    valid_frame = frame.loc[valid].reset_index(drop=True)

    base = {
        "source_system": source_system or first_nonempty(valid_frame.get("source_system"), ""),
        "run_id": run_id or first_nonempty(valid_frame.get("run_id"), ""),
        "timestamp_utc": first_nonempty(valid_frame.get("timestamp_utc"), utc_timestamp()),
        "git_commit": first_nonempty(valid_frame.get("git_commit"), git_commit()),
        "dataset": first_nonempty(valid_frame.get("dataset"), ""),
        "split": first_nonempty(valid_frame.get("split"), ""),
        "score_kind": score_kind,
    }

    rows: list[dict[str, Any]] = []
    overall = {**base, "group_name": "all", "group_value": "all"}
    overall.update(metric_block(scores, labels, llrs))
    rows.append(overall)

    if not include_breakdowns:
        return rows

    for column in ["language_condition", "target"]:
        if column not in valid_frame.columns:
            continue
        for value, subset in valid_frame.groupby(column, sort=True):
            idx = subset.index.to_numpy(dtype=np.int64)
            row = {**base, "group_name": column, "group_value": str(value)}
            row.update(metric_block(scores[idx], labels[idx], None if llrs is None else llrs[idx]))
            rows.append(row)
    if {"target", "language_condition"}.issubset(valid_frame.columns):
        grouped = valid_frame.groupby(["target", "language_condition"], sort=True)
        for (target_value, lang_value), subset in grouped:
            idx = subset.index.to_numpy(dtype=np.int64)
            row = {**base, "group_name": "target_by_language_condition", "group_value": f"{target_value}:{lang_value}"}
            row.update(metric_block(scores[idx], labels[idx], None if llrs is None else llrs[idx]))
            rows.append(row)
    return rows


def first_nonempty(series: Any, default: str) -> str:
    if series is None:
        return default
    if isinstance(series, pd.Series):
        nonempty = series.dropna().astype(str)
        nonempty = nonempty[nonempty.str.len() > 0]
        if not nonempty.empty:
            return str(nonempty.iloc[0])
        return default
    return str(series) if series else default


def write_metric_report(metrics: pd.DataFrame, path: Path | str, title: str) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", f"- generated_utc: {utc_timestamp()}", f"- git_commit: {git_commit()}", ""]
    if metrics.empty:
        lines.append("No metrics were generated.")
    else:
        display_cols = [
            "source_system",
            "score_kind",
            "group_name",
            "group_value",
            "n",
            "target_n",
            "nontarget_n",
            "eer_pct",
            "Cllr",
            "minDCF_p01",
            "actDCF_p01",
        ]
        display_cols = [column for column in display_cols if column in metrics.columns]
        lines.append(metrics[display_cols].to_markdown(index=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_mismatch_log(mismatch: dict[str, Any], path: Path | str) -> None:
    write_json(mismatch, path)


def write_csv_rows(rows: list[dict[str, Any]], path: Path | str) -> None:
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
