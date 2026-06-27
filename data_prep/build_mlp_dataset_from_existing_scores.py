#!/usr/bin/env python3
"""Build a speaker-disjoint MLP wide table from existing TidyVoice score CSVs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from build_c0_c5_trial_table import (
    DEFAULT_EMBEDDINGS,
    SCORE_USECOLS,
    add_features,
    assert_aligned,
    compute_duration_center,
    group_score_files,
    iter_limited_chunks,
    output_columns,
    resolve_output_format,
    score_column,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORES_DIR = PROJECT_ROOT / "scores"
DEFAULT_MANIFEST = PROJECT_ROOT / "protocols" / "tidyvoice_mlp_speaker_split_manifest.csv"
DEFAULT_CENTER_PROTOCOL = PROJECT_ROOT / "protocols" / "tidyvoice_mlp_calibration.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "tidyvoice_trials.parquet"

REFERENCE_USECOLS = [
    "split",
    "trial_id",
    "label",
    "score",
    "enroll_speaker",
    "test_speaker",
    "language_condition",
    "enroll_duration_sec",
    "test_duration_sec",
]


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-dir", type=Path, default=DEFAULT_SCORES_DIR)
    parser.add_argument(
        "--source-splits",
        nargs="+",
        default=["calibration", "test"],
        help=(
            "Existing score-file split prefixes to read. For the 60/20/20 TidyVoice "
            "MLP split, keep this as calibration test so existing score CSVs can be reused."
        ),
    )
    parser.add_argument(
        "--score-file",
        type=Path,
        action="append",
        help="Explicit existing score file. Can be repeated. Overrides --scores-dir discovery.",
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--output-format",
        choices=("auto", "csv", "parquet"),
        default="auto",
    )
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--center-protocol", type=Path, default=DEFAULT_CENTER_PROTOCOL)
    parser.add_argument("--duration-center", type=float)
    parser.add_argument(
        "--center-strategy",
        choices=("median", "mean", "geomean"),
        default="median",
    )
    parser.add_argument(
        "--output-splits",
        nargs="+",
        default=["calibration", "validation", "test"],
        help="Speaker-disjoint splits to keep in the MLP table.",
    )
    parser.add_argument("--dataset-name", default="tidyvoice_asv_mlp_60_20_20")
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--max-rows-per-source-split", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def discover_score_files(args: argparse.Namespace) -> list[Path]:
    if args.score_file:
        files = [path.expanduser().resolve() for path in args.score_file]
    else:
        files = []
        for split in args.source_splits:
            files.extend(sorted(args.scores_dir.glob(f"tidyvoice_{split}_*.csv")))
        files = [path for path in files if "smoke" not in path.stem and "_mlp_" not in path.stem]
    if not files:
        raise FileNotFoundError("No source score files matched.")
    missing = [path for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing score files: {missing}")
    return files


def load_manifest(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path, usecols=["speaker_id", "speaker_split"])
    if frame.empty:
        raise ValueError(f"Empty split manifest: {path}")
    assignments = dict(zip(frame["speaker_id"].astype(str), frame["speaker_split"].astype(str)))
    allowed = {"calibration", "validation", "test"}
    unexpected = sorted(set(assignments.values()) - allowed)
    if unexpected:
        raise ValueError(f"Unexpected speaker split(s) in {path}: {unexpected}")
    return assignments


def assign_trial_splits(frame: pd.DataFrame, assignments: dict[str, str]) -> pd.Series:
    enroll_split = frame["enroll_speaker"].astype(str).map(assignments)
    test_split = frame["test_speaker"].astype(str).map(assignments)
    same_partition = enroll_split.notna() & enroll_split.eq(test_split)
    return enroll_split.where(same_partition, "unused_cross_split")


def open_writer(path: Path, output_format: str, columns: list[str]):
    if output_format == "csv":
        output_obj = path.open("w", encoding="utf-8", newline="")
        return output_obj, None, True
    return None, None, False


def close_writer(csv_output_obj, parquet_writer) -> None:
    if parquet_writer is not None:
        parquet_writer.close()
    if csv_output_obj is not None:
        csv_output_obj.close()


def write_featured_chunk(
    featured: pd.DataFrame,
    *,
    output_path: Path,
    output_format: str,
    csv_output_obj,
    parquet_writer,
    write_header: bool,
) -> tuple[pq.ParquetWriter | None, bool]:
    if output_format == "csv":
        featured.to_csv(
            csv_output_obj,
            index=False,
            header=write_header,
            na_rep="",
            lineterminator="\n",
        )
        return parquet_writer, False
    table = pa.Table.from_pandas(featured, preserve_index=False)
    if parquet_writer is None:
        parquet_writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
    parquet_writer.write_table(table)
    return parquet_writer, write_header


def build_table(
    *,
    grouped_files: dict[str, dict[str, Path]],
    embeddings: list[str],
    assignments: dict[str, str],
    output_path: Path,
    output_format: str,
    chunksize: int,
    max_rows_per_source_split: int | None,
    duration_center: float,
    output_splits: set[str],
) -> tuple[int, dict[str, int], int, dict[str, dict[str, str]]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    columns = output_columns(embeddings)
    score_files_by_source_split = {
        split: {embedding: str(files_by_embedding[embedding]) for embedding in embeddings}
        for split, files_by_embedding in grouped_files.items()
    }

    csv_output_obj = None
    parquet_writer = None
    write_header = True
    rows_written = 0
    dropped_unused = 0
    rows_by_split: Counter[str] = Counter()
    csv_output_obj, parquet_writer, write_header = open_writer(temp_path, output_format, columns)

    try:
        for source_split in sorted(grouped_files):
            files_by_embedding = grouped_files[source_split]
            reference_embedding = embeddings[0]
            reference_path = files_by_embedding[reference_embedding]
            reference_iter = iter_limited_chunks(
                reference_path,
                usecols=REFERENCE_USECOLS,
                chunksize=chunksize,
                max_rows=max_rows_per_source_split,
            )
            other_iters = {
                embedding: iter_limited_chunks(
                    files_by_embedding[embedding],
                    usecols=SCORE_USECOLS,
                    chunksize=chunksize,
                    max_rows=max_rows_per_source_split,
                )
                for embedding in embeddings[1:]
            }

            source_rows_written = 0
            source_rows_dropped = 0
            for reference_chunk in reference_iter:
                new_splits = assign_trial_splits(reference_chunk, assignments)
                keep_mask = new_splits.isin(output_splits)
                source_rows_dropped += int((~keep_mask).sum())

                score_chunks = {}
                for embedding, iterator in other_iters.items():
                    try:
                        score_chunk = next(iterator)
                    except StopIteration as exc:
                        raise ValueError(
                            f"{files_by_embedding[embedding]} ended before {reference_path}"
                        ) from exc
                    assert_aligned(reference_chunk, score_chunk, path=files_by_embedding[embedding])
                    score_chunks[embedding] = score_chunk

                if not keep_mask.any():
                    continue

                selected_reference = reference_chunk.loc[keep_mask].copy()
                selected_reference["split"] = new_splits.loc[keep_mask].astype(str).to_numpy()
                featured = add_features(selected_reference, duration_center=duration_center)
                featured[score_column(reference_embedding)] = pd.to_numeric(
                    selected_reference["score"], errors="coerce"
                ).to_numpy()

                for embedding, score_chunk in score_chunks.items():
                    featured[score_column(embedding)] = pd.to_numeric(
                        score_chunk.loc[keep_mask, "score"], errors="coerce"
                    ).to_numpy()

                featured = featured[columns]
                split_counts = Counter(featured["split"].astype(str))
                rows_by_split.update(split_counts)
                count = int(featured.shape[0])
                rows_written += count
                source_rows_written += count
                parquet_writer, write_header = write_featured_chunk(
                    featured,
                    output_path=temp_path,
                    output_format=output_format,
                    csv_output_obj=csv_output_obj,
                    parquet_writer=parquet_writer,
                    write_header=write_header,
                )

            for embedding, iterator in other_iters.items():
                try:
                    extra = next(iterator)
                except StopIteration:
                    continue
                raise ValueError(
                    f"{files_by_embedding[embedding]} has extra rows after {reference_path}: "
                    f"{extra.shape[0]}"
                )

            dropped_unused += source_rows_dropped
            print(
                f"source_split={source_split} wrote={source_rows_written} "
                f"dropped_unused={source_rows_dropped}",
                flush=True,
            )
    finally:
        if rows_written == 0 and output_format == "parquet" and parquet_writer is None:
            empty = pa.Table.from_pandas(pd.DataFrame(columns=columns), preserve_index=False)
            parquet_writer = pq.ParquetWriter(temp_path, empty.schema, compression="zstd")
            parquet_writer.write_table(empty)
        close_writer(csv_output_obj, parquet_writer)

    temp_path.replace(output_path)
    return rows_written, dict(rows_by_split), dropped_unused, score_files_by_source_split


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    output_path = args.output.expanduser().resolve()
    output_format = resolve_output_format(output_path, args.output_format)
    metadata_output = (
        args.metadata_output.expanduser().resolve()
        if args.metadata_output
        else output_path.with_suffix(output_path.suffix + ".settings.json")
    )
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"Output already exists: {output_path}. Use --overwrite to replace it.")

    score_files = discover_score_files(args)
    grouped_files, discovered_embeddings = group_score_files(score_files)
    embeddings = [embedding for embedding in DEFAULT_EMBEDDINGS if embedding in discovered_embeddings]
    embeddings.extend(sorted(set(discovered_embeddings) - set(embeddings)))
    assignments = load_manifest(args.manifest)

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

    rows_written, rows_by_split, dropped_unused, score_files_by_source_split = build_table(
        grouped_files=grouped_files,
        embeddings=embeddings,
        assignments=assignments,
        output_path=output_path,
        output_format=output_format,
        chunksize=args.chunksize,
        max_rows_per_source_split=args.max_rows_per_source_split,
        duration_center=duration_center,
        output_splits=set(args.output_splits),
    )

    metadata = {
        "output": str(output_path),
        "mode": "wide_mlp_speaker_disjoint",
        "format": output_format,
        "rows": rows_written,
        "rows_by_split": rows_by_split,
        "dropped_unused_cross_split_rows": dropped_unused,
        "columns": output_columns(embeddings),
        "score_columns": {embedding: score_column(embedding) for embedding in embeddings},
        "source_score_files_by_split": score_files_by_source_split,
        "split_manifest": str(args.manifest),
        "output_splits": args.output_splits,
        "dataset_settings": {
            args.dataset_name: {
                "duration_center_sec": duration_center,
                "duration_center_strategy": args.center_strategy,
                "duration_center_source": center_source,
                "duration_center_split": "calibration",
                "duration_count": duration_count,
            }
        },
        "leakage_control": {
            "unit": "speaker",
            "policy": (
                "A trial is kept only when enroll_speaker and test_speaker are assigned "
                "to the same speaker split. Cross-split nontarget trials are excluded."
            ),
        },
    }
    metadata_output.parent.mkdir(parents=True, exist_ok=True)
    with metadata_output.open("w", encoding="utf-8") as file_obj:
        json.dump(metadata, file_obj, indent=2, ensure_ascii=True)
        file_obj.write("\n")

    print(f"wrote table: {output_path}")
    print(f"wrote settings: {metadata_output}")
    print(f"rows: {rows_written}")
    print(f"rows_by_split: {rows_by_split}")
    print(f"dropped_unused_cross_split_rows: {dropped_unused}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
