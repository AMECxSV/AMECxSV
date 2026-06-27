#!/usr/bin/env python3
"""Build a wide Parquet trial table with multiple similarity metrics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PREP_ROOT = PROJECT_ROOT / "data_prep"
if str(DATA_PREP_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_PREP_ROOT))

from build_c0_c5_trial_table import (
    DEFAULT_CENTER_PROTOCOL,
    DEFAULT_EMBEDDINGS,
    DEFAULT_SCORES_DIR,
    FEATURE_COLUMNS,
    add_features,
    assert_aligned,
    compute_duration_center,
    iter_limited_chunks,
    resolve_output_format,
    table_format,
)
from similarity_scores import SCORE_SPECS, parse_metric_list


DEFAULT_OUTPUT = PROJECT_ROOT / "similarity" / "outputs" / "tidyvoice_similarity_scores_pilot.parquet"
REFERENCE_USECOLS = [
    "split",
    "trial_id",
    "label",
    "score",
    "language_condition",
    "enroll_duration_sec",
    "test_duration_sec",
]


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-dir", type=Path, default=DEFAULT_SCORES_DIR)
    parser.add_argument("--score-glob", default="tidyvoice_*.parquet")
    parser.add_argument("--score-file", type=Path, action="append")
    parser.add_argument("--metrics", help="Comma-separated metrics. Omit to infer from score files.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output-format", choices=("auto", "csv", "parquet"), default="auto")
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--alignment-audit-output", type=Path)
    parser.add_argument("--center-protocol", type=Path, default=DEFAULT_CENTER_PROTOCOL)
    parser.add_argument("--duration-center", type=float)
    parser.add_argument("--center-strategy", choices=("median", "mean", "geomean"), default="median")
    parser.add_argument("--dataset-name", default="tidyvoice_asv")
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--max-rows-per-split", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def read_score_file_info(path: Path) -> tuple[str, str, list[str]]:
    fmt = table_format(path)
    if fmt == "parquet":
        parquet_file = pq.ParquetFile(path)
        columns = parquet_file.schema.names
        batches = parquet_file.iter_batches(batch_size=1, columns=["split", "embedding_name"])
        try:
            frame = next(batches).to_pandas()
        except StopIteration:
            frame = pd.DataFrame()
        row = None if frame.empty else frame.iloc[0].to_dict()
    else:
        frame = pd.read_csv(path, nrows=1)
        columns = list(frame.columns)
        row = None if frame.empty else frame.iloc[0].to_dict()
    if row is None:
        raise ValueError(f"{path} has no rows.")
    split = str(row.get("split") or "").strip()
    embedding = str(row.get("embedding_name") or "").strip()
    if not split or not embedding:
        raise ValueError(f"{path} must contain split and embedding_name columns.")
    return split, embedding, columns


def discover_score_files(args: argparse.Namespace) -> list[Path]:
    if args.score_file:
        files = [path.expanduser().resolve() for path in args.score_file]
    else:
        files = sorted(args.scores_dir.glob(args.score_glob))
        files = [path for path in files if "smoke" not in path.stem]
    if not files:
        raise FileNotFoundError("No score files matched.")
    missing = [path for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing score files: {missing}")
    return files


def group_score_files(score_files: list[Path]) -> tuple[dict[str, dict[str, Path]], list[str], dict[str, list[str]]]:
    grouped: dict[str, dict[str, Path]] = {}
    discovered_embeddings: set[str] = set()
    columns_by_path: dict[str, list[str]] = {}
    for path in score_files:
        split, embedding, columns = read_score_file_info(path)
        if embedding in grouped.setdefault(split, {}):
            raise ValueError(f"Duplicate score file for split={split}, embedding={embedding}")
        grouped[split][embedding] = path
        discovered_embeddings.add(embedding)
        columns_by_path[str(path)] = columns
    embeddings = [embedding for embedding in DEFAULT_EMBEDDINGS if embedding in discovered_embeddings]
    embeddings.extend(sorted(discovered_embeddings - set(embeddings)))
    expected = set(embeddings)
    for split, files_by_embedding in grouped.items():
        if set(files_by_embedding) != expected:
            raise ValueError(
                f"Split {split} does not have the same embedding set. "
                f"missing={sorted(expected - set(files_by_embedding))} "
                f"extra={sorted(set(files_by_embedding) - expected)}"
            )
    return grouped, embeddings, columns_by_path


def infer_metrics(columns_by_path: dict[str, list[str]]) -> list[str]:
    metric_names = {"cosine"}
    for columns in columns_by_path.values():
        for column in columns:
            if column.startswith("score_"):
                metric = column[len("score_") :]
                if metric in SCORE_SPECS:
                    metric_names.add(metric)
    return [metric for metric in SCORE_SPECS if metric in metric_names]


def wide_score_column(embedding: str, metric: str) -> str:
    return f"score__{embedding}__{metric}"


def compatibility_score_column(embedding: str) -> str:
    return f"score_{embedding}"


def output_columns(embeddings: list[str], metrics: list[str]) -> list[str]:
    columns = ["split", "trial_id", "label", "target"]
    columns.extend(compatibility_score_column(embedding) for embedding in embeddings)
    for embedding in embeddings:
        for metric in metrics:
            columns.append(wide_score_column(embedding, metric))
    columns.extend(FEATURE_COLUMNS)
    return columns


def score_usecols_for_metrics(metrics: list[str], available_columns: list[str]) -> list[str]:
    usecols = ["split", "trial_id", "label", "score"]
    for metric in metrics:
        metric_column = f"score_{metric}"
        if metric == "cosine":
            continue
        if metric_column not in available_columns:
            raise ValueError(f"Requested metric={metric} but column {metric_column} is missing.")
        usecols.append(metric_column)
    return usecols


def reference_usecols_for_metrics(metrics: list[str], available_columns: list[str]) -> list[str]:
    usecols = list(REFERENCE_USECOLS)
    for metric in metrics:
        metric_column = f"score_{metric}"
        if metric == "cosine":
            continue
        if metric_column not in available_columns:
            raise ValueError(f"Requested metric={metric} but column {metric_column} is missing.")
        usecols.append(metric_column)
    return usecols


def score_values(frame: pd.DataFrame, metric: str) -> pd.Series:
    if metric == "cosine":
        if "score_cosine" in frame.columns:
            return pd.to_numeric(frame["score_cosine"], errors="coerce")
        return pd.to_numeric(frame["score"], errors="coerce")
    return pd.to_numeric(frame[f"score_{metric}"], errors="coerce")


def iter_score_chunks(
    path: Path,
    *,
    metrics: list[str],
    columns_by_path: dict[str, list[str]],
    chunksize: int,
    max_rows: int | None,
) -> Iterator[pd.DataFrame]:
    usecols = score_usecols_for_metrics(metrics, columns_by_path[str(path)])
    yield from iter_limited_chunks(path, usecols=usecols, chunksize=chunksize, max_rows=max_rows)


def write_table(
    grouped_files: dict[str, dict[str, Path]],
    embeddings: list[str],
    metrics: list[str],
    columns_by_path: dict[str, list[str]],
    output_path: Path,
    *,
    output_format: str,
    chunksize: int,
    max_rows_per_split: int | None,
    duration_center: float,
) -> tuple[int, dict[str, int], list[dict]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    rows_written = 0
    rows_by_split: dict[str, int] = {}
    alignment_rows: list[dict] = []
    columns = output_columns(embeddings, metrics)
    csv_output_obj = None
    parquet_writer: pq.ParquetWriter | None = None
    write_header = True
    if output_format == "csv":
        csv_output_obj = temp_path.open("w", encoding="utf-8", newline="")
    try:
        for split in sorted(grouped_files):
            files_by_embedding = grouped_files[split]
            reference_embedding = embeddings[0]
            reference_path = files_by_embedding[reference_embedding]
            reference_iter = iter_limited_chunks(
                reference_path,
                usecols=reference_usecols_for_metrics(metrics, columns_by_path[str(reference_path)]),
                chunksize=chunksize,
                max_rows=max_rows_per_split,
            )
            other_iters = {
                embedding: iter_score_chunks(
                    files_by_embedding[embedding],
                    metrics=metrics,
                    columns_by_path=columns_by_path,
                    chunksize=chunksize,
                    max_rows=max_rows_per_split,
                )
                for embedding in embeddings[1:]
            }
            split_rows = 0
            for reference_chunk in reference_iter:
                featured = add_features(reference_chunk, duration_center=duration_center)
                featured[compatibility_score_column(reference_embedding)] = pd.to_numeric(reference_chunk["score"], errors="coerce")
                for metric in metrics:
                    featured[wide_score_column(reference_embedding, metric)] = score_values(reference_chunk, metric)
                for embedding, iterator in other_iters.items():
                    try:
                        chunk = next(iterator)
                    except StopIteration as exc:
                        raise ValueError(f"{files_by_embedding[embedding]} ended before {reference_path}") from exc
                    assert_aligned(reference_chunk, chunk, path=files_by_embedding[embedding])
                    featured[compatibility_score_column(embedding)] = score_values(chunk, "cosine")
                    for metric in metrics:
                        featured[wide_score_column(embedding, metric)] = score_values(chunk, metric)
                featured = featured[columns]
                if output_format == "csv":
                    if csv_output_obj is None:
                        raise RuntimeError("CSV output is not initialized.")
                    featured.to_csv(csv_output_obj, index=False, header=write_header, na_rep="", lineterminator="\n")
                    write_header = False
                else:
                    table = pa.Table.from_pandas(featured, preserve_index=False)
                    if parquet_writer is None:
                        parquet_writer = pq.ParquetWriter(temp_path, table.schema, compression="zstd")
                    parquet_writer.write_table(table)
                count = int(featured.shape[0])
                rows_written += count
                split_rows += count
            for embedding, iterator in other_iters.items():
                try:
                    extra = next(iterator)
                except StopIteration:
                    continue
                raise ValueError(f"{files_by_embedding[embedding]} has extra rows after reference file: {extra.shape[0]}")
            rows_by_split[split] = split_rows
            alignment_rows.append({"split": split, "rows": split_rows, "reference_embedding": reference_embedding})
            print(f"wrote {split_rows} rows for split={split}", flush=True)
        if output_format == "parquet" and parquet_writer is None:
            empty = pa.Table.from_pandas(pd.DataFrame(columns=columns), preserve_index=False)
            parquet_writer = pq.ParquetWriter(temp_path, empty.schema, compression="zstd")
            parquet_writer.write_table(empty)
    finally:
        if parquet_writer is not None:
            parquet_writer.close()
        if csv_output_obj is not None:
            csv_output_obj.close()
    temp_path.replace(output_path)
    return rows_written, rows_by_split, alignment_rows


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    output_path = args.output.expanduser().resolve()
    output_format = resolve_output_format(output_path, args.output_format)
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output_path}. Use --overwrite to replace it.")
    score_files = discover_score_files(args)
    grouped_files, embeddings, columns_by_path = group_score_files(score_files)
    metrics = parse_metric_list(args.metrics) if args.metrics else infer_metrics(columns_by_path)

    if args.duration_center is not None:
        duration_center = float(args.duration_center)
        duration_count = None
        center_source = "explicit --duration-center"
    else:
        duration_center, duration_count = compute_duration_center(
            args.center_protocol,
            strategy=args.center_strategy,
            chunksize=args.chunksize,
        )
        center_source = str(args.center_protocol)

    rows_written, rows_by_split, alignment_rows = write_table(
        grouped_files,
        embeddings,
        metrics,
        columns_by_path,
        output_path,
        output_format=output_format,
        chunksize=args.chunksize,
        max_rows_per_split=args.max_rows_per_split,
        duration_center=duration_center,
    )
    metadata_output = (
        args.metadata_output.expanduser().resolve()
        if args.metadata_output
        else output_path.with_suffix(output_path.suffix + ".settings.json")
    )
    alignment_output = (
        args.alignment_audit_output.expanduser().resolve()
        if args.alignment_audit_output
        else output_path.with_suffix(output_path.suffix + ".alignment_audit.json")
    )
    metadata = {
        "output": str(output_path),
        "mode": "similarity_wide",
        "format": output_format,
        "rows": rows_written,
        "rows_by_split": rows_by_split,
        "embeddings": embeddings,
        "metrics": metrics,
        "columns": output_columns(embeddings, metrics),
        "score_files_by_split": {
            split: {embedding: str(path) for embedding, path in files.items()}
            for split, files in grouped_files.items()
        },
        "dataset_settings": {
            args.dataset_name: {
                "duration_center_sec": duration_center,
                "duration_center_strategy": args.center_strategy,
                "duration_center_source": center_source,
                "duration_count": duration_count,
            }
        },
    }
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    with metadata_output.open("w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2, ensure_ascii=True)
        file_obj.write("\n")
    with alignment_output.open("w", encoding="utf-8") as file_obj:
        json.dump({"alignment": alignment_rows, "rows_by_split": rows_by_split}, file_obj, indent=2, ensure_ascii=True)
        file_obj.write("\n")
    print(f"wrote table: {output_path}")
    print(f"wrote settings: {metadata_output}")
    print(f"wrote alignment audit: {alignment_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
