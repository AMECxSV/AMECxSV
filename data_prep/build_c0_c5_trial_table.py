#!/usr/bin/env python3
"""Build one wide C0-C5 trial table from raw score CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable, Iterator, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORES_DIR = PROJECT_ROOT / "scores"
DEFAULT_CENTER_PROTOCOL = PROJECT_ROOT / "protocols" / "tidyvoice_calibration.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "build" / "c0_c5_trial_table" / "tidyvoice_c0_c5_train.csv"

DEFAULT_EMBEDDINGS = [
    "speechbrain_ecapa_tdnn_voxceleb",
    "wespeaker_resnet34_cnceleb",
    "funasr_campplus_cn_3k",
    "funasr_eres2netv2_cn_200k",
    "hf_wavlm_base_sv_voxceleb1",
    "hf_wavlm_base_plus_sv_voxceleb1",
]

KEY_COLUMNS = [
    "split",
    "trial_id",
    "label",
    "target",
]

FEATURE_COLUMNS = [
    "qmf1",
    "qmf2",
    "qmf3",
    "qmf4_sum",
    "qmf4_diff",
    "min_duration",
    "duration_ratio",
    "short_duration_risk",
]

REFERENCE_USECOLS = [
    "split",
    "trial_id",
    "label",
    "score",
    "language_condition",
    "enroll_duration_sec",
    "test_duration_sec",
]

SCORE_USECOLS = [
    "split",
    "trial_id",
    "label",
    "score",
]


def score_column(embedding: str) -> str:
    return f"score_{embedding}"


def output_columns(embeddings: list[str]) -> list[str]:
    return KEY_COLUMNS + [score_column(embedding) for embedding in embeddings] + FEATURE_COLUMNS


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a compact wide experiment-input table usable by C0-C10 "
            "baselines. One row is one trial; embedding scores are separate columns."
        )
    )
    parser.add_argument("--scores-dir", type=Path, default=DEFAULT_SCORES_DIR)
    parser.add_argument(
        "--score-glob",
        default="tidyvoice_*.csv",
        help="Glob under --scores-dir. Smoke files are skipped unless explicitly passed.",
    )
    parser.add_argument(
        "--score-file",
        type=Path,
        action="append",
        help="Explicit raw score table. Can be repeated. Overrides --score-glob.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--output-format",
        choices=("auto", "csv", "parquet"),
        default="auto",
        help="Output table format. auto uses parquet for .parquet outputs, otherwise CSV.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        help="Sidecar JSON path. Defaults to OUTPUT with .settings.json suffix.",
    )
    parser.add_argument(
        "--center-protocol",
        type=Path,
        default=DEFAULT_CENTER_PROTOCOL,
        help="Calibration protocol/score CSV used to compute duration_center_sec.",
    )
    parser.add_argument(
        "--duration-center",
        type=float,
        help="Override duration_center_sec. If omitted, computed from --center-protocol.",
    )
    parser.add_argument(
        "--center-strategy",
        choices=("median", "mean", "geomean"),
        default="median",
        help="How to compute duration_center_sec from calibration durations.",
    )
    parser.add_argument("--dataset-name", default="tidyvoice_asv")
    parser.add_argument("--center-split", default="calibration")
    parser.add_argument("--reject-prior", type=float, default=0.001)
    parser.add_argument("--reject-cost-miss", type=float, default=1.0)
    parser.add_argument("--reject-cost-fa", type=float, default=1.0)
    parser.add_argument("--reject-cost-reject", type=float, default=0.1)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument(
        "--max-rows-per-split",
        type=int,
        help="Limit output rows per split for smoke tests.",
    )
    parser.add_argument(
        "--max-rows-per-file",
        type=int,
        help="Backward-compatible alias for --max-rows-per-split.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def table_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return "parquet"
    if suffix == ".csv":
        return "csv"
    raise ValueError(f"Unsupported table extension for {path}; expected .csv or .parquet.")


def resolve_output_format(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    return table_format(path)


def discover_score_files(args: argparse.Namespace) -> list[Path]:
    if args.score_file:
        files = [path.expanduser().resolve() for path in args.score_file]
    else:
        files = sorted(args.scores_dir.glob(args.score_glob))
        files = [path for path in files if "smoke" not in path.stem]
    if not files:
        raise FileNotFoundError("No raw score files matched.")
    missing = [path for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing score files: {missing}")
    return files


def read_score_file_info(path: Path) -> tuple[str, str]:
    if table_format(path) == "parquet":
        parquet_file = pq.ParquetFile(path)
        batches = parquet_file.iter_batches(
            batch_size=1,
            columns=["split", "embedding_name"],
        )
        try:
            row_frame = next(batches).to_pandas()
        except StopIteration:
            row_frame = pd.DataFrame()
        row = None if row_frame.empty else row_frame.iloc[0].to_dict()
    else:
        with path.open("r", encoding="utf-8", newline="") as file_obj:
            reader = csv.DictReader(file_obj)
            row = next(reader, None)
    if row is None:
        raise ValueError(f"{path} has no rows.")
    split = str(row.get("split") or "").strip()
    embedding = str(row.get("embedding_name") or "").strip()
    if not split or not embedding:
        raise ValueError(f"{path} must contain split and embedding_name columns.")
    return split, embedding


def group_score_files(score_files: list[Path]) -> tuple[dict[str, dict[str, Path]], list[str]]:
    grouped: dict[str, dict[str, Path]] = {}
    discovered_embeddings: set[str] = set()

    for path in score_files:
        split, embedding = read_score_file_info(path)
        if embedding in grouped.setdefault(split, {}):
            raise ValueError(f"Duplicate score file for split={split}, embedding={embedding}")
        grouped[split][embedding] = path
        discovered_embeddings.add(embedding)

    embeddings = [
        embedding for embedding in DEFAULT_EMBEDDINGS if embedding in discovered_embeddings
    ]
    embeddings.extend(sorted(discovered_embeddings - set(embeddings)))

    expected = set(embeddings)
    for split, files_by_embedding in grouped.items():
        missing = expected - set(files_by_embedding)
        extra = set(files_by_embedding) - expected
        if missing or extra:
            raise ValueError(
                f"Split {split} does not have the same embedding set. "
                f"missing={sorted(missing)} extra={sorted(extra)}"
            )

    return grouped, embeddings


def _valid_durations(frame: pd.DataFrame) -> np.ndarray:
    enroll = pd.to_numeric(frame["enroll_duration_sec"], errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    test = pd.to_numeric(frame["test_duration_sec"], errors="coerce").to_numpy(
        dtype=np.float64, copy=False
    )
    values = np.concatenate([enroll, test])
    return values[np.isfinite(values) & (values > 0.0)]


def compute_duration_center(
    path: Path,
    *,
    strategy: str,
    chunksize: int,
) -> tuple[float, int]:
    values: list[np.ndarray] = []
    total = 0
    sum_value = 0.0
    sum_log = 0.0

    for chunk in pd.read_csv(
        path,
        usecols=["enroll_duration_sec", "test_duration_sec"],
        chunksize=chunksize,
    ):
        durations = _valid_durations(chunk)
        if durations.size == 0:
            continue
        total += int(durations.size)
        if strategy == "median":
            values.append(durations)
        elif strategy == "mean":
            sum_value += float(np.sum(durations))
        elif strategy == "geomean":
            sum_log += float(np.sum(np.log(durations)))

    if total == 0:
        raise ValueError(f"No positive durations found in {path}")

    if strategy == "median":
        center = float(np.median(np.concatenate(values)))
    elif strategy == "mean":
        center = sum_value / total
    elif strategy == "geomean":
        center = float(math.exp(sum_log / total))
    else:
        raise ValueError(f"Unsupported center strategy: {strategy}")

    if not math.isfinite(center) or center <= 0.0:
        raise ValueError(f"Invalid duration center computed from {path}: {center}")
    return center, total


def add_features(frame: pd.DataFrame, *, duration_center: float) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    output["split"] = frame["split"]
    output["trial_id"] = frame["trial_id"]
    output["label"] = pd.to_numeric(frame["label"], errors="raise").astype("int8")
    output["target"] = np.where(frame["language_condition"].eq("same_language"), 1, 0)

    enroll = pd.to_numeric(frame["enroll_duration_sec"], errors="coerce")
    test = pd.to_numeric(frame["test_duration_sec"], errors="coerce")
    valid = (enroll > 0.0) & (test > 0.0)

    min_duration = pd.Series(np.nan, index=frame.index, dtype=np.float64)
    max_duration = pd.Series(np.nan, index=frame.index, dtype=np.float64)
    min_duration.loc[valid] = np.minimum(enroll.loc[valid], test.loc[valid])
    max_duration.loc[valid] = np.maximum(enroll.loc[valid], test.loc[valid])
    output["min_duration"] = min_duration
    output["duration_ratio"] = min_duration / max_duration
    output["short_duration_risk"] = 1.0 / np.sqrt(min_duration)

    log_ratio = pd.Series(np.nan, index=frame.index, dtype=np.float64)
    log_ratio.loc[valid] = np.log(enroll.loc[valid] / test.loc[valid])

    enroll_centered = pd.Series(np.nan, index=frame.index, dtype=np.float64)
    test_centered = pd.Series(np.nan, index=frame.index, dtype=np.float64)
    enroll_centered.loc[valid] = np.log(enroll.loc[valid] / duration_center)
    test_centered.loc[valid] = np.log(test.loc[valid] / duration_center)

    output["qmf1"] = log_ratio.abs()
    output["qmf2"] = log_ratio**2
    output["qmf3"] = enroll_centered * test_centered
    output["qmf4_sum"] = (enroll_centered + test_centered) ** 2
    output["qmf4_diff"] = -((enroll_centered - test_centered) ** 2)

    return output


def iter_limited_chunks(
    path: Path,
    *,
    usecols: list[str],
    chunksize: int,
    max_rows: Optional[int],
) -> Iterator[pd.DataFrame]:
    rows_read = 0
    if table_format(path) == "parquet":
        chunks = (
            batch.to_pandas()
            for batch in pq.ParquetFile(path).iter_batches(batch_size=chunksize, columns=usecols)
        )
    else:
        chunks = pd.read_csv(path, usecols=usecols, chunksize=chunksize)
    for chunk in chunks:
        if max_rows is not None:
            remaining = max_rows - rows_read
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)
        rows_read += int(chunk.shape[0])
        yield chunk


def assert_aligned(reference: pd.DataFrame, candidate: pd.DataFrame, *, path: Path) -> None:
    if reference.shape[0] != candidate.shape[0]:
        raise ValueError(
            f"Chunk row mismatch for {path}: reference={reference.shape[0]} "
            f"candidate={candidate.shape[0]}"
        )
    for column in ("split", "trial_id", "label"):
        left = reference[column].reset_index(drop=True)
        right = candidate[column].reset_index(drop=True)
        if column == "label":
            left = pd.to_numeric(left, errors="raise")
            right = pd.to_numeric(right, errors="raise")
        else:
            left = left.astype(str)
            right = right.astype(str)
        if not left.equals(right):
            raise ValueError(f"{path} is not aligned on column {column}.")


def write_table(
    grouped_files: dict[str, dict[str, Path]],
    embeddings: list[str],
    output_path: Path,
    *,
    output_format: str,
    chunksize: int,
    max_rows_per_split: Optional[int],
    duration_center: float,
) -> tuple[int, dict[str, int], dict[str, dict[str, str]]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    rows_written = 0
    rows_by_split: dict[str, int] = {}
    write_header = True
    columns = output_columns(embeddings)
    score_files_by_split = {
        split: {embedding: str(files_by_embedding[embedding]) for embedding in embeddings}
        for split, files_by_embedding in grouped_files.items()
    }

    csv_output_obj = None
    parquet_writer: pq.ParquetWriter | None = None
    if output_format == "csv":
        csv_output_obj = temp_path.open("w", encoding="utf-8", newline="")

    try:
        for split in sorted(grouped_files):
            files_by_embedding = grouped_files[split]
            reference_embedding = embeddings[0]
            reference_path = files_by_embedding[reference_embedding]
            reference_iter = iter_limited_chunks(
                reference_path,
                usecols=REFERENCE_USECOLS,
                chunksize=chunksize,
                max_rows=max_rows_per_split,
            )
            other_iters = {
                embedding: iter_limited_chunks(
                    files_by_embedding[embedding],
                    usecols=SCORE_USECOLS,
                    chunksize=chunksize,
                    max_rows=max_rows_per_split,
                )
                for embedding in embeddings[1:]
            }

            split_rows = 0
            for reference_chunk in reference_iter:
                featured = add_features(reference_chunk, duration_center=duration_center)
                featured[score_column(reference_embedding)] = pd.to_numeric(
                    reference_chunk["score"], errors="coerce"
                )

                for embedding, iterator in other_iters.items():
                    try:
                        score_chunk = next(iterator)
                    except StopIteration as exc:
                        raise ValueError(
                            f"{files_by_embedding[embedding]} ended before {reference_path}"
                        ) from exc
                    assert_aligned(
                        reference_chunk,
                        score_chunk,
                        path=files_by_embedding[embedding],
                    )
                    featured[score_column(embedding)] = pd.to_numeric(
                        score_chunk["score"], errors="coerce"
                    )

                featured = featured[columns]
                if output_format == "csv":
                    if csv_output_obj is None:
                        raise RuntimeError("CSV output is not initialized.")
                    featured.to_csv(
                        csv_output_obj,
                        index=False,
                        header=write_header,
                        na_rep="",
                        lineterminator="\n",
                    )
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
                raise ValueError(
                    f"{files_by_embedding[embedding]} has extra rows after reference file: "
                    f"{extra.shape[0]}"
                )

            rows_by_split[split] = split_rows
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
    return rows_written, rows_by_split, score_files_by_split


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    score_files = discover_score_files(args)
    grouped_files, embeddings = group_score_files(score_files)
    output_path = args.output.expanduser().resolve()
    output_format = resolve_output_format(output_path, args.output_format)
    metadata_output = (
        args.metadata_output.expanduser().resolve()
        if args.metadata_output
        else output_path.with_suffix(output_path.suffix + ".settings.json")
    )
    max_rows_per_split = (
        args.max_rows_per_split
        if args.max_rows_per_split is not None
        else args.max_rows_per_file
    )

    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    if args.duration_center is not None:
        duration_center = args.duration_center
        duration_count = None
        center_source = "explicit --duration-center"
    else:
        duration_center, duration_count = compute_duration_center(
            args.center_protocol,
            strategy=args.center_strategy,
            chunksize=args.chunksize,
        )
        center_source = str(args.center_protocol)

    rows_written, rows_by_split, score_files_by_split = write_table(
        grouped_files,
        embeddings,
        output_path,
        output_format=output_format,
        chunksize=args.chunksize,
        max_rows_per_split=max_rows_per_split,
        duration_center=duration_center,
    )

    score_columns = {embedding: score_column(embedding) for embedding in embeddings}
    metadata = {
        "output": str(output_path),
        "mode": "wide",
        "format": output_format,
        "rows": rows_written,
        "columns": output_columns(embeddings),
        "score_columns": score_columns,
        "column_semantics": {
            "label": "same-speaker ASV class: 1 target/same speaker, 0 nontarget/different speaker",
            "target": "same-language flag for C4: 1 same language, 0 cross/unknown language",
            "score_*": "raw embedding cosine scores, before calibration",
            "qmf1": "abs(log(enroll_duration_sec / test_duration_sec))",
            "qmf2": "log(enroll_duration_sec / test_duration_sec) squared",
            "qmf3": "log(enroll_duration_sec / duration_center_sec) * log(test_duration_sec / duration_center_sec)",
            "qmf4_sum": "squared sum of centered log durations",
            "qmf4_diff": "negative squared difference of centered log durations",
        },
        "score_files_by_split": score_files_by_split,
        "rows_by_split": rows_by_split,
        "dataset_settings": {
            args.dataset_name: {
                "duration_center_sec": duration_center,
                "duration_center_strategy": args.center_strategy,
                "duration_center_source": center_source,
                "duration_center_split": args.center_split,
                "duration_count": duration_count,
                "reject_prior": args.reject_prior,
                "reject_cost_miss": args.reject_cost_miss,
                "reject_cost_fa": args.reject_cost_fa,
                "reject_cost_reject": args.reject_cost_reject,
            }
        },
    }
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    with metadata_output.open("w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2, ensure_ascii=True)
        file_obj.write("\n")

    print(f"wrote table: {output_path}")
    print(f"wrote settings: {metadata_output}")
    print(f"rows: {rows_written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
