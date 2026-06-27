#!/usr/bin/env python3
"""Audit metric redundancy and optionally freeze a selected feature set."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from similarity_scores import SCORE_SPECS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "similarity" / "outputs" / "tidyvoice_similarity_scores_pilot.parquet"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "similarity" / "outputs"
EMBEDDINGS = [
    "speechbrain_ecapa_tdnn_voxceleb",
    "wespeaker_resnet34_cnceleb",
    "funasr_campplus_cn_3k",
    "funasr_eres2netv2_cn_200k",
    "hf_wavlm_base_sv_voxceleb1",
    "hf_wavlm_base_plus_sv_voxceleb1",
]


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-table", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--fallback-split", default="calibration")
    parser.add_argument("--max-rows", type=int, default=200_000)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def available_metrics(columns: list[str]) -> list[str]:
    metrics = {"cosine"}
    for column in columns:
        if column.startswith("score__"):
            metric = column.rsplit("__", 1)[-1]
            if metric in SCORE_SPECS:
                metrics.add(metric)
    return [metric for metric in SCORE_SPECS if metric in metrics]


def metric_column(frame: pd.DataFrame, embedding: str, metric: str) -> str | None:
    wide = f"score__{embedding}__{metric}"
    if wide in frame.columns:
        return wide
    compat = f"score_{embedding}"
    if metric == "cosine" and compat in frame.columns:
        return compat
    return None


def deterministic_sample(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if frame.shape[0] <= max_rows:
        return frame.reset_index(drop=True)
    hashed = pd.util.hash_pandas_object(frame["trial_id"].astype(str), index=False)
    return frame.assign(_rank=hashed).sort_values("_rank", kind="mergesort").head(max_rows).drop(columns=["_rank"]).reset_index(drop=True)


def redundancy_rows(frame: pd.DataFrame) -> list[dict[str, object]]:
    metrics = available_metrics(list(frame.columns))
    rows: list[dict[str, object]] = []
    labels = frame["label"].to_numpy(dtype=np.int8, copy=False) if "label" in frame else None
    for embedding in EMBEDDINGS:
        columns = {metric: metric_column(frame, embedding, metric) for metric in metrics}
        columns = {metric: column for metric, column in columns.items() if column is not None}
        if "cosine" not in columns:
            continue
        cosine_values = pd.to_numeric(frame[columns["cosine"]], errors="coerce")
        for metric, column in columns.items():
            values = pd.to_numeric(frame[column], errors="coerce")
            mask = np.isfinite(cosine_values.to_numpy()) & np.isfinite(values.to_numpy())
            if mask.sum() < 2:
                pearson = spearman = np.nan
            else:
                pearson = float(pd.Series(cosine_values.to_numpy()[mask]).corr(pd.Series(values.to_numpy()[mask]), method="pearson"))
                spearman = float(pd.Series(cosine_values.to_numpy()[mask]).corr(pd.Series(values.to_numpy()[mask]), method="spearman"))
            unique_count = int(pd.Series(values.to_numpy()[mask]).nunique()) if mask.any() else 0
            redundant = bool(metric != "cosine" and abs(spearman) > 0.9999)
            rows.append(
                {
                    "embedding": embedding,
                    "metric": metric,
                    "reference_metric": "cosine",
                    "pearson_with_cosine": pearson,
                    "spearman_with_cosine": spearman,
                    "unique_value_count": unique_count,
                    "finite_count": int(mask.sum()),
                    "redundant_with_cosine": redundant,
                    "diagnostic_only": SCORE_SPECS[metric].diagnostic_only,
                    "family": SCORE_SPECS[metric].family,
                }
            )
        metric_items = sorted(columns.items())
        for left_idx, (left_metric, left_col) in enumerate(metric_items):
            left = pd.to_numeric(frame[left_col], errors="coerce").to_numpy()
            for right_metric, right_col in metric_items[left_idx + 1 :]:
                right = pd.to_numeric(frame[right_col], errors="coerce").to_numpy()
                mask = np.isfinite(left) & np.isfinite(right)
                if mask.sum() < 2:
                    continue
                rows.append(
                    {
                        "embedding": embedding,
                        "metric": left_metric,
                        "reference_metric": right_metric,
                        "pearson_with_cosine": float(pd.Series(left[mask]).corr(pd.Series(right[mask]), method="pearson")),
                        "spearman_with_cosine": float(pd.Series(left[mask]).corr(pd.Series(right[mask]), method="spearman")),
                        "unique_value_count": int(pd.Series(left[mask]).nunique()),
                        "finite_count": int(mask.sum()),
                        "redundant_with_cosine": bool(abs(pd.Series(left[mask]).corr(pd.Series(right[mask]), method="spearman")) > 0.9999),
                        "diagnostic_only": SCORE_SPECS[left_metric].diagnostic_only,
                        "family": SCORE_SPECS[left_metric].family,
                    }
                )
    return rows


def write_report(path: Path, correlation: pd.DataFrame) -> None:
    lines = ["# Metric Redundancy Report", ""]
    if correlation.empty:
        lines.append("No metric correlations were available.")
    else:
        redundant = correlation[
            (correlation["reference_metric"] == "cosine")
            & (correlation["metric"] != "cosine")
            & (correlation["redundant_with_cosine"].astype(bool))
        ]
        if redundant.empty:
            lines.append("No non-cosine metric crossed the automatic Spearman redundancy threshold against cosine.")
        else:
            lines.append("Metrics automatically marked `redundant_with_cosine`:")
            for _, row in redundant.iterrows():
                lines.append(
                    f"- `{row['embedding']}` / `{row['metric']}`: "
                    f"Spearman={row['spearman_with_cosine']:.6f}"
                )
        lines.append("")
        lines.append("Diagnostic-only metrics remain excluded from the default final candidate set unless explicitly selected.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def git_commit_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return None


def freeze_selection(results_dir: Path, output_path: Path, *, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise SystemExit(f"Output already exists: {output_path}. Use --overwrite to replace it.")
    selected_path = results_dir / "selected_feature_results.csv"
    if not selected_path.exists():
        raise FileNotFoundError(selected_path)
    selected = pd.read_csv(selected_path)
    if selected.empty or "status" in selected.columns:
        raise SystemExit("No selected pilot result is available to freeze.")
    best = selected.sort_values(["Cllr", "actDCF_p01", "eer_pct"], ascending=[True, True, True]).iloc[0].to_dict()
    feature_columns = str(best.get("feature_columns", "")).split("|") if best.get("feature_columns") else []
    payload = {
        "status": "frozen",
        "freeze_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit_hash": git_commit_hash(),
        "selected_score_feature_set": best.get("score_feature_set"),
        "selected_model": best.get("model"),
        "selected_seed": int(best.get("seed", 0)),
        "feature_columns": feature_columns,
        "feature_count": int(best.get("feature_count", len(feature_columns))),
        "selection_val_metrics": best,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    results_dir = args.results_dir.expanduser().resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.input_table) if args.input_table.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(args.input_table)
    split_values = set(frame["split"].astype(str).unique())
    split = args.split if args.split in split_values else args.fallback_split
    sample = deterministic_sample(frame[frame["split"].astype(str) == split], args.max_rows)
    correlation = pd.DataFrame(redundancy_rows(sample))
    correlation_path = results_dir / "metric_correlation.csv"
    if correlation_path.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {correlation_path}. Use --overwrite to replace it.")
    correlation.to_csv(correlation_path, index=False)
    write_report(results_dir / "metric_redundancy_report.md", correlation)
    if args.freeze:
        freeze_selection(
            results_dir,
            PROJECT_ROOT / "similarity" / "frozen_similarity_selection.json",
            overwrite=args.overwrite,
        )
    print(f"wrote redundancy audit under {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

