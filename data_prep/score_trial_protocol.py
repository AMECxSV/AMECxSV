#!/usr/bin/env python3
"""Compute cosine ASV scores for a trial protocol from extracted embeddings."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SIMILARITY_ROOT = PROJECT_ROOT / "similarity"
if str(SIMILARITY_ROOT) not in sys.path:
    sys.path.insert(0, str(SIMILARITY_ROOT))

from similarity_backend_models import (
    asnorm_scores_from_stats,
    asnorm_stats_for_model,
    load_backend_model,
    score_backend_metrics,
)
from similarity_scores import (
    SCORE_SPECS,
    compute_pairwise_scores,
    cosine,
    finite_audit,
    metric_config_hash,
    parse_metric_list,
)


DEFAULT_PROTOCOL = PROJECT_ROOT / "protocols" / "voxceleb1b.csv"
DEFAULT_EMBEDDINGS_ROOT = PROJECT_ROOT / "data" / "embeddings"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "scores" / "voxceleb1b_speechbrain_ecapa_tdnn_voxceleb.csv"
DEFAULT_MODEL = "speechbrain_ecapa_tdnn_voxceleb"


def safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe utterance path in protocol: {value!r}")
    return path


def embedding_path(embeddings_root: Path, model: str, utterance: str) -> Path:
    return embeddings_root / model / safe_relative_path(utterance).with_suffix(".json")


def load_embedding(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as file_obj:
        payload = json.load(file_obj)
    return np.asarray(payload["embedding"], dtype=np.float32).reshape(-1)


def cosine_score(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator == 0.0:
        return math.nan
    return float(np.dot(left, right) / denominator)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute trial cosine scores from protocol-level embedding JSON files."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--embeddings-root", type=Path, default=DEFAULT_EMBEDDINGS_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--metrics",
        help=(
            "Comma-separated similarity metrics. Omit to preserve legacy cosine-only "
            "schema. When set, score_<metric> columns are emitted in addition to legacy score."
        ),
    )
    parser.add_argument(
        "--backend-models-dir",
        type=Path,
        help="Directory containing <model>/<metric>.npz fitted backend artifacts.",
    )
    parser.add_argument(
        "--allow-blocked-backends",
        action="store_true",
        help="Emit NaN columns for backend artifacts whose manifest status is not ok.",
    )
    parser.add_argument(
        "--output-format",
        choices=("auto", "csv", "parquet"),
        default="auto",
        help="Output table format. auto uses parquet for .parquet outputs, otherwise CSV.",
    )
    parser.add_argument("--parquet-row-group-size", type=int, default=100_000)
    parser.add_argument("--score-batch-size", type=int, default=4096)
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--skip-missing-embeddings", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--audit-output",
        type=Path,
        help="Sidecar audit JSON. Defaults to OUTPUT with .audit.json suffix.",
    )
    parser.add_argument("--progress-every", type=int, default=100000)
    return parser.parse_args(argv)


def resolve_output_format(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    if path.suffix.lower() in {".parquet", ".pq"}:
        return "parquet"
    return "csv"


def flush_parquet_rows(
    rows: list[dict],
    output_path: Path,
    writer: pq.ParquetWriter | None,
) -> pq.ParquetWriter | None:
    if not rows:
        return writer
    table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
    writer.write_table(table)
    rows.clear()
    return writer


def table_row_count(path: Path, output_format: str) -> int:
    if not path.exists():
        return 0
    if output_format == "parquet":
        return int(pq.ParquetFile(path).metadata.num_rows)
    with path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.reader(file_obj)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def copy_existing_parquet(source: Path, target: Path) -> pq.ParquetWriter | None:
    parquet_file = pq.ParquetFile(source)
    writer: pq.ParquetWriter | None = None
    for batch in parquet_file.iter_batches(batch_size=100_000):
        table = pa.Table.from_batches([batch])
        if writer is None:
            writer = pq.ParquetWriter(target, table.schema, compression="zstd")
        writer.write_table(table)
    return writer


def append_csv_rows(path: Path, fieldnames: list[str], rows: list[dict], *, write_header: bool) -> None:
    with path.open("a", encoding="utf-8", newline="") as output_obj:
        writer = csv.DictWriter(output_obj, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def load_requested_backends(
    backend_models_dir: Path | None,
    model: str,
    metrics: list[str],
) -> dict[str, object]:
    backend_metrics = [metric for metric in metrics if SCORE_SPECS[metric].requires_fit]
    if not backend_metrics:
        return {}
    if backend_models_dir is None:
        raise ValueError(f"Fitted backend metrics requested without --backend-models-dir: {backend_metrics}")
    loaded = {}
    for metric in backend_metrics:
        path = backend_models_dir / model / f"{metric}.npz"
        loaded[metric] = load_backend_model(path)
    return loaded


def score_asnorm_with_utterance_cache(
    backend_model: object,
    enroll: np.ndarray,
    test: np.ndarray,
    enroll_keys: list[str],
    test_keys: list[str],
    embedding_cache: dict[str, np.ndarray],
    stats_cache: dict[str, tuple[float, float]],
) -> np.ndarray:
    missing_keys: list[str] = []
    seen_missing: set[str] = set()
    for key in [*enroll_keys, *test_keys]:
        if key not in stats_cache and key not in seen_missing:
            missing_keys.append(key)
            seen_missing.add(key)

    if missing_keys:
        values = np.vstack([embedding_cache[key] for key in missing_keys])
        mus, sigmas = asnorm_stats_for_model(backend_model, values)
        for key, mu, sigma in zip(missing_keys, mus, sigmas):
            stats_cache[key] = (float(mu), float(sigma))

    enroll_mu = np.asarray([stats_cache[key][0] for key in enroll_keys], dtype=np.float64)
    enroll_sigma = np.asarray([stats_cache[key][1] for key in enroll_keys], dtype=np.float64)
    test_mu = np.asarray([stats_cache[key][0] for key in test_keys], dtype=np.float64)
    test_sigma = np.asarray([stats_cache[key][1] for key in test_keys], dtype=np.float64)
    return asnorm_scores_from_stats(enroll, test, enroll_mu, enroll_sigma, test_mu, test_sigma)


def ensure_backend_runnable(metric: str, backend_model: object, allow_blocked_backends: bool, n_rows: int) -> np.ndarray | None:
    status = backend_model.metadata.get("status")
    if status == "ok":
        return None
    if allow_blocked_backends:
        return np.full(n_rows, np.nan, dtype=np.float64)
    raise RuntimeError(
        f"Backend metric {metric} is not runnable: "
        f"{status} {backend_model.metadata.get('blocked_reason', '')}"
    )


def transformed_vectors_for_metric(
    metric: str,
    backend_model: object,
    keys: list[str],
    embedding_cache: dict[str, np.ndarray],
    backend_cache: dict[str, dict[str, object]],
) -> np.ndarray:
    metric_cache = backend_cache.setdefault(metric, {})
    missing_keys: list[str] = []
    seen_missing: set[str] = set()
    for key in keys:
        if key not in metric_cache and key not in seen_missing:
            missing_keys.append(key)
            seen_missing.add(key)

    if missing_keys:
        values = np.vstack([embedding_cache[key] for key in missing_keys])
        if metric == "centered_cosine":
            transformed = values - backend_model.arrays["mean"]
        elif metric in {"whitened_cosine", "wccn_cosine"}:
            mean = backend_model.arrays.get("mean", np.zeros((values.shape[1],), dtype=np.float64))
            transformed = (values - mean) @ backend_model.arrays["transform"]
        elif metric == "lda_cosine":
            projected = (values - backend_model.arrays["mean"]) @ backend_model.arrays["projection"]
            norms = np.linalg.norm(projected, axis=1, keepdims=True)
            transformed = projected / np.maximum(norms, 1.0e-12)
        else:
            raise ValueError(f"Unsupported cached transform metric: {metric}")
        for key, vector in zip(missing_keys, transformed):
            metric_cache[key] = np.asarray(vector, dtype=np.float64)

    return np.vstack([metric_cache[key] for key in keys])


def score_neg_mahalanobis_with_utterance_cache(
    backend_model: object,
    enroll: np.ndarray,
    test: np.ndarray,
    enroll_keys: list[str],
    test_keys: list[str],
    embedding_cache: dict[str, np.ndarray],
    backend_cache: dict[str, dict[str, object]],
) -> np.ndarray:
    if "precision_diag" in backend_model.arrays:
        diff = enroll - test
        return -np.sum((diff**2) * backend_model.arrays["precision_diag"], axis=1)

    metric_cache = backend_cache.setdefault("neg_mahalanobis", {})
    precision = backend_model.arrays["precision"]
    missing_keys: list[str] = []
    seen_missing: set[str] = set()
    for key in [*enroll_keys, *test_keys]:
        if key not in metric_cache and key not in seen_missing:
            missing_keys.append(key)
            seen_missing.add(key)

    if missing_keys:
        values = np.vstack([embedding_cache[key] for key in missing_keys])
        projected = values @ precision
        quadratics = np.einsum("ij,ij->i", values, projected)
        for key, pvec, quad in zip(missing_keys, projected, quadratics):
            metric_cache[key] = (np.asarray(pvec, dtype=np.float64), float(quad))

    enroll_p = np.vstack([metric_cache[key][0] for key in enroll_keys])
    enroll_quad = np.asarray([metric_cache[key][1] for key in enroll_keys], dtype=np.float64)
    test_quad = np.asarray([metric_cache[key][1] for key in test_keys], dtype=np.float64)
    cross = np.einsum("ij,ij->i", enroll_p, test)
    return -(enroll_quad + test_quad - (2.0 * cross))


def score_cacheable_backend_metrics(
    backend_models: dict[str, object],
    enroll: np.ndarray,
    test: np.ndarray,
    enroll_keys: list[str],
    test_keys: list[str],
    embedding_cache: dict[str, np.ndarray],
    asnorm_stats_cache: dict[str, tuple[float, float]],
    backend_cache: dict[str, dict[str, object]],
    metrics: list[str],
    *,
    allow_blocked_backends: bool,
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for metric in metrics:
        if metric not in {"centered_cosine", "whitened_cosine", "wccn_cosine", "lda_cosine", "neg_mahalanobis", "asnorm_cosine"}:
            continue
        if metric not in backend_models:
            raise FileNotFoundError(f"No backend model loaded for metric={metric}")
        backend_model = backend_models[metric]
        blocked = ensure_backend_runnable(metric, backend_model, allow_blocked_backends, enroll.shape[0])
        if blocked is not None:
            output[metric] = blocked
            continue
        if metric in {"centered_cosine", "whitened_cosine", "wccn_cosine", "lda_cosine"}:
            enroll_transformed = transformed_vectors_for_metric(
                metric,
                backend_model,
                enroll_keys,
                embedding_cache,
                backend_cache,
            )
            test_transformed = transformed_vectors_for_metric(
                metric,
                backend_model,
                test_keys,
                embedding_cache,
                backend_cache,
            )
            output[metric] = cosine(enroll_transformed, test_transformed)
        elif metric == "neg_mahalanobis":
            output[metric] = score_neg_mahalanobis_with_utterance_cache(
                backend_model,
                enroll,
                test,
                enroll_keys,
                test_keys,
                embedding_cache,
                backend_cache,
            )
        elif metric == "asnorm_cosine":
            output[metric] = score_asnorm_with_utterance_cache(
                backend_model,
                enroll,
                test,
                enroll_keys,
                test_keys,
                embedding_cache,
                asnorm_stats_cache,
            )
    return output


def process_batch(
    rows: list[dict],
    *,
    cache: dict[str, np.ndarray],
    asnorm_stats_cache: dict[str, tuple[float, float]],
    backend_score_cache: dict[str, dict[str, object]],
    embeddings_root: Path,
    model: str,
    metrics: list[str],
    explicit_metrics: bool,
    backend_models: dict[str, object],
    allow_blocked_backends: bool,
    output_format: str,
) -> tuple[list[dict], dict[str, dict[str, int]]]:
    enroll_embeddings = []
    test_embeddings = []
    enroll_keys = []
    test_keys = []
    for row in rows:
        enroll_utt = row["enroll_utt"].strip()
        test_utt = row["test_utt"].strip()
        if enroll_utt not in cache:
            cache[enroll_utt] = load_embedding(embedding_path(embeddings_root, model, enroll_utt))
        if test_utt not in cache:
            cache[test_utt] = load_embedding(embedding_path(embeddings_root, model, test_utt))
        enroll_keys.append(enroll_utt)
        test_keys.append(test_utt)
        enroll_embeddings.append(cache[enroll_utt])
        test_embeddings.append(cache[test_utt])

    enroll = np.vstack(enroll_embeddings)
    test = np.vstack(test_embeddings)
    nofit_metrics = sorted({metric for metric in metrics if not SCORE_SPECS[metric].requires_fit} | {"cosine"})
    nofit_scores = compute_pairwise_scores(enroll, test, nofit_metrics)
    cached_backend_scores = score_cacheable_backend_metrics(
        backend_models,
        enroll,
        test,
        enroll_keys,
        test_keys,
        cache,
        asnorm_stats_cache,
        backend_score_cache,
        metrics,
        allow_blocked_backends=allow_blocked_backends,
    )
    cacheable_backend_metrics = set(cached_backend_scores)
    backend_metric_names = [metric for metric in metrics if metric not in cacheable_backend_metrics]
    backend_scores = score_backend_metrics(
        backend_models,
        enroll,
        test,
        backend_metric_names,
        allow_blocked=allow_blocked_backends,
    )
    scores = {**nofit_scores, **cached_backend_scores, **backend_scores}
    audit = finite_audit(scores)

    output_rows: list[dict] = []
    for idx, row in enumerate(rows):
        score = float(scores["cosine"][idx])
        output_row = dict(row)
        if output_format == "csv":
            output_row.update(
                {
                    "embedding_name": model,
                    "score": f"{score:.8f}",
                    "enroll_embedding_exists": "true",
                    "test_embedding_exists": "true",
                }
            )
            if explicit_metrics:
                for metric in metrics:
                    value = float(scores[metric][idx])
                    output_row[f"score_{metric}"] = "" if not math.isfinite(value) else f"{value:.8f}"
        else:
            output_row.update(
                {
                    "label": int(row["label"]),
                    "embedding_name": model,
                    "score": score,
                    "enroll_embedding_exists": True,
                    "test_embedding_exists": True,
                }
            )
            if explicit_metrics:
                for metric in metrics:
                    output_row[f"score_{metric}"] = np.float32(scores[metric][idx])
        output_rows.append(output_row)
    return output_rows, audit


def merge_audits(total: dict[str, dict[str, int]], chunk: dict[str, dict[str, int]]) -> None:
    for metric, counts in chunk.items():
        target = total.setdefault(metric, {key: 0 for key in counts})
        for key, value in counts.items():
            target[key] = int(target.get(key, 0) + value)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    cache: dict[str, np.ndarray] = {}
    asnorm_stats_cache: dict[str, tuple[float, float]] = {}
    backend_score_cache: dict[str, dict[str, object]] = {}
    written = 0
    skipped_missing = 0
    output_format = resolve_output_format(args.output, args.output_format)
    explicit_metrics = args.metrics is not None
    metrics = parse_metric_list(args.metrics)
    backend_models = load_requested_backends(args.backend_models_dir, args.model, metrics)
    audit_output = (
        args.audit_output.expanduser().resolve()
        if args.audit_output
        else args.output.with_suffix(args.output.suffix + ".audit.json")
    )
    score_audit: dict[str, dict[str, int]] = {}
    metric_hash = metric_config_hash(metrics, {"explicit_metrics": explicit_metrics})
    existing_rows = table_row_count(args.output, output_format) if args.resume else 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.protocol.open("r", encoding="utf-8", newline="") as input_obj:
        reader = csv.DictReader(input_obj)
        required = {"enroll_utt", "test_utt"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"{args.protocol} must contain columns: {sorted(required)}")

        fieldnames = list(reader.fieldnames or [])
        metric_fields = [f"score_{metric}" for metric in metrics] if explicit_metrics else []
        output_fields = fieldnames + [
            "embedding_name",
            "score",
            *metric_fields,
            "enroll_embedding_exists",
            "test_embedding_exists",
        ]

        csv_output_obj = None
        csv_writer: csv.DictWriter | None = None
        parquet_writer: pq.ParquetWriter | None = None
        parquet_rows: list[dict] = []
        pending_rows: list[dict] = []
        output_path = args.output
        temp_path = output_path.with_suffix(output_path.suffix + ".tmp") if output_format == "parquet" and args.resume and output_path.exists() else output_path

        if output_format == "csv":
            mode = "a" if args.resume and args.output.exists() else "w"
            csv_output_obj = args.output.open(mode, encoding="utf-8", newline="")
            csv_writer = csv.DictWriter(csv_output_obj, fieldnames=output_fields)
            if mode == "w" or existing_rows == 0:
                csv_writer.writeheader()
        elif args.resume and output_path.exists():
            parquet_writer = copy_existing_parquet(output_path, temp_path)

        try:
            for trial_index, row in enumerate(reader, start=1):
                if args.max_trials is not None and trial_index > args.max_trials:
                    break
                if existing_rows and trial_index <= existing_rows:
                    continue

                enroll_utt = row["enroll_utt"].strip()
                test_utt = row["test_utt"].strip()
                enroll_path = embedding_path(args.embeddings_root, args.model, enroll_utt)
                test_path = embedding_path(args.embeddings_root, args.model, test_utt)
                enroll_exists = enroll_path.exists()
                test_exists = test_path.exists()

                if not enroll_exists or not test_exists:
                    if args.skip_missing_embeddings:
                        skipped_missing += 1
                        continue
                    missing = []
                    if not enroll_exists:
                        missing.append(str(enroll_path))
                    if not test_exists:
                        missing.append(str(test_path))
                    raise FileNotFoundError("Missing embedding(s): " + ", ".join(missing))

                pending_rows.append(row)
                if len(pending_rows) >= args.score_batch_size:
                    output_rows, batch_audit = process_batch(
                        pending_rows,
                        cache=cache,
                        asnorm_stats_cache=asnorm_stats_cache,
                        backend_score_cache=backend_score_cache,
                        embeddings_root=args.embeddings_root,
                        model=args.model,
                        metrics=metrics,
                        explicit_metrics=explicit_metrics,
                        backend_models=backend_models,
                        allow_blocked_backends=args.allow_blocked_backends,
                        output_format=output_format,
                    )
                    merge_audits(score_audit, batch_audit)
                    if output_format == "csv":
                        if csv_writer is None:
                            raise RuntimeError("CSV writer is not initialized.")
                        csv_writer.writerows(output_rows)
                    else:
                        parquet_rows.extend(output_rows)
                        if len(parquet_rows) >= args.parquet_row_group_size:
                            parquet_writer = flush_parquet_rows(
                                parquet_rows,
                                temp_path,
                                parquet_writer,
                            )
                    written += len(output_rows)
                    pending_rows.clear()

                if args.progress_every and trial_index % args.progress_every == 0:
                    print(
                        f"processed_trials={trial_index} written={written} "
                        f"cached_embeddings={len(cache)} asnorm_cached={len(asnorm_stats_cache)} "
                        f"skipped_missing={skipped_missing}",
                        flush=True,
                    )
            if pending_rows:
                output_rows, batch_audit = process_batch(
                    pending_rows,
                    cache=cache,
                    asnorm_stats_cache=asnorm_stats_cache,
                    backend_score_cache=backend_score_cache,
                    embeddings_root=args.embeddings_root,
                    model=args.model,
                    metrics=metrics,
                    explicit_metrics=explicit_metrics,
                    backend_models=backend_models,
                    allow_blocked_backends=args.allow_blocked_backends,
                    output_format=output_format,
                )
                merge_audits(score_audit, batch_audit)
                if output_format == "csv":
                    if csv_writer is None:
                        raise RuntimeError("CSV writer is not initialized.")
                    csv_writer.writerows(output_rows)
                else:
                    parquet_rows.extend(output_rows)
                written += len(output_rows)
                pending_rows.clear()
            if output_format == "parquet":
                parquet_writer = flush_parquet_rows(parquet_rows, temp_path, parquet_writer)
                if parquet_writer is None:
                    empty_table = pa.Table.from_pydict({field: [] for field in output_fields})
                    parquet_writer = pq.ParquetWriter(temp_path, empty_table.schema, compression="zstd")
                    parquet_writer.write_table(empty_table)
        finally:
            if parquet_writer is not None:
                parquet_writer.close()
            if csv_output_obj is not None:
                csv_output_obj.close()
        if output_format == "parquet" and temp_path != output_path:
            temp_path.replace(output_path)

    audit = {
        "protocol": str(args.protocol),
        "embeddings_root": str(args.embeddings_root),
        "model": args.model,
        "output": str(args.output),
        "output_format": output_format,
        "explicit_metrics": explicit_metrics,
        "metrics": metrics,
        "metric_config_hash": metric_hash,
        "backend_models_dir": str(args.backend_models_dir) if args.backend_models_dir else None,
        "allow_blocked_backends": bool(args.allow_blocked_backends),
        "resume": bool(args.resume),
        "existing_rows_skipped": int(existing_rows),
        "written": int(written),
        "cached_embeddings": int(len(cache)),
        "asnorm_cached_embeddings": int(len(asnorm_stats_cache)),
        "backend_cached_embeddings_by_metric": {
            metric: int(len(metric_cache)) for metric, metric_cache in sorted(backend_score_cache.items())
        },
        "skipped_missing": int(skipped_missing),
        "score_audit": score_audit,
    }
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    with audit_output.open("w", encoding="utf-8") as file_obj:
        json.dump(audit, file_obj, indent=2, ensure_ascii=True)
        file_obj.write("\n")

    print(
        f"done written={written} cached_embeddings={len(cache)} "
        f"skipped_missing={skipped_missing} output={args.output} audit={audit_output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
