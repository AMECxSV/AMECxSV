#!/usr/bin/env python3
"""Fit calibration-only similarity backend artifacts for one extractor."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PREP_ROOT = PROJECT_ROOT / "data_prep"
if str(DATA_PREP_ROOT) not in sys.path:
    sys.path.insert(0, str(DATA_PREP_ROOT))

from score_trial_protocol import embedding_path, load_embedding
from similarity_backend_models import (
    BackendFitError,
    SimilarityBackendModel,
    fit_asnorm_cosine,
    fit_centered_cosine,
    fit_lda_cosine,
    fit_neg_mahalanobis,
    fit_plda_placeholder,
    fit_wccn_cosine,
    fit_whitened_cosine,
    stable_speaker_hash,
)
from similarity_scores import SCORE_SPECS, parse_metric_list


DEFAULT_PROTOCOL = PROJECT_ROOT / "protocols" / "tidyvoice_calibration.csv"
DEFAULT_TEST_PROTOCOL = PROJECT_ROOT / "protocols" / "tidyvoice_test.csv"
DEFAULT_EMBEDDINGS_ROOT = PROJECT_ROOT / "data" / "embeddings"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "similarity" / "outputs" / "backend_models"
DEFAULT_MODEL = "speechbrain_ecapa_tdnn_voxceleb"
DEFAULT_BACKEND_METRICS = (
    "centered_cosine",
    "whitened_cosine",
    "wccn_cosine",
    "lda_cosine",
    "neg_mahalanobis",
    "asnorm_cosine",
    "plda_llr",
)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--test-protocol", type=Path, default=DEFAULT_TEST_PROTOCOL)
    parser.add_argument("--embeddings-root", type=Path, default=DEFAULT_EMBEDDINGS_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metrics", default=",".join(DEFAULT_BACKEND_METRICS))
    parser.add_argument(
        "--fit-speakers",
        type=Path,
        help="Optional text/CSV file limiting backend-fit speakers. CSV must include speaker_id or speaker.",
    )
    parser.add_argument("--max-utterances", type=int)
    parser.add_argument("--covariance-method", choices=("ledoit_wolf", "oas", "empirical"), default="ledoit_wolf")
    parser.add_argument("--eigenvalue-floor", type=float, default=1.0e-5)
    parser.add_argument("--whitened-output-dim", type=int)
    parser.add_argument("--lda-output-dim", type=int, default=128)
    parser.add_argument("--mahalanobis-diagonal", action="store_true")
    parser.add_argument("--asnorm-max-cohort-size", type=int, default=1000)
    parser.add_argument("--asnorm-per-speaker", type=int, default=1)
    parser.add_argument("--asnorm-top-k", type=int, default=100)
    parser.add_argument("--asnorm-sigma-floor", type=float, default=1.0e-4)
    parser.add_argument("--cohort-seed", default="asnorm_cohort_v1")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def read_speaker_limit(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        column = "speaker_id" if "speaker_id" in frame.columns else "speaker"
        if column not in frame.columns:
            raise ValueError(f"{path} must contain speaker_id or speaker column.")
        return {str(value) for value in frame[column].dropna().astype(str)}
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def collect_test_speakers(path: Path) -> set[str]:
    speakers: set[str] = set()
    for chunk in pd.read_csv(path, usecols=["enroll_speaker", "test_speaker"], chunksize=250_000):
        speakers.update(chunk["enroll_speaker"].dropna().astype(str).unique())
        speakers.update(chunk["test_speaker"].dropna().astype(str).unique())
    return speakers


def collect_utterance_speakers(path: Path, fit_speaker_limit: set[str] | None) -> dict[str, str]:
    utterance_to_speaker: dict[str, str] = {}
    usecols = ["enroll_utt", "test_utt", "enroll_speaker", "test_speaker"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=250_000):
        for utt_col, speaker_col in (("enroll_utt", "enroll_speaker"), ("test_utt", "test_speaker")):
            pairs = chunk[[utt_col, speaker_col]].dropna().drop_duplicates()
            for utt, speaker in pairs.itertuples(index=False, name=None):
                speaker_str = str(speaker)
                if fit_speaker_limit is not None and speaker_str not in fit_speaker_limit:
                    continue
                utt_str = str(utt)
                old = utterance_to_speaker.get(utt_str)
                if old is not None and old != speaker_str:
                    raise ValueError(f"Utterance {utt_str} maps to conflicting speakers: {old}, {speaker_str}")
                utterance_to_speaker[utt_str] = speaker_str
    return utterance_to_speaker


def load_backend_fit_embeddings(
    utterance_to_speaker: dict[str, str],
    *,
    embeddings_root: Path,
    model: str,
    max_utterances: int | None,
) -> tuple[np.ndarray, list[str], list[str], dict[str, int]]:
    rows: list[np.ndarray] = []
    speakers: list[str] = []
    utterances: list[str] = []
    missing = 0
    bad_shape = 0
    expected_dim: int | None = None

    items = sorted(utterance_to_speaker.items(), key=lambda item: (item[1], item[0]))
    if max_utterances is not None:
        items = items[:max_utterances]
    for utt, speaker in items:
        path = embedding_path(embeddings_root, model, utt)
        if not path.exists():
            missing += 1
            continue
        emb = load_embedding(path).astype(np.float64, copy=False)
        if expected_dim is None:
            expected_dim = int(emb.shape[0])
        if emb.ndim != 1 or emb.shape[0] != expected_dim:
            bad_shape += 1
            continue
        rows.append(emb)
        speakers.append(speaker)
        utterances.append(utt)
    if not rows:
        raise BackendFitError("No embeddings loaded for backend fit.")
    return np.vstack(rows), speakers, utterances, {
        "missing_embedding_count": missing,
        "bad_shape_count": bad_shape,
    }


def blocked_artifact(metric: str, model: str, fit_speakers: set[str], test_speakers: set[str], reason: str) -> SimilarityBackendModel:
    overlap = sorted(fit_speakers & test_speakers)
    return SimilarityBackendModel(
        metric=metric,
        extractor=model,
        arrays={"placeholder": np.asarray([0.0], dtype=np.float64)},
        metadata={
            "status": "blocked",
            "blocked_reason": reason,
            "fit_speaker_count": len(fit_speakers),
            "fit_speaker_hash": stable_speaker_hash(fit_speakers),
            "test_speaker_count": len(test_speakers),
            "test_speaker_hash": stable_speaker_hash(test_speakers),
            "test_speaker_intersection_count": len(overlap),
            "test_speaker_intersection_preview": overlap[:10],
        },
    )


def fit_one_metric(
    metric: str,
    embeddings: np.ndarray,
    speakers: list[str],
    utterances: list[str],
    *,
    args: argparse.Namespace,
    fit_speaker_set: set[str],
    test_speakers: set[str],
) -> SimilarityBackendModel:
    if metric == "centered_cosine":
        return fit_centered_cosine(
            embeddings,
            extractor=args.model,
            fit_speakers=fit_speaker_set,
            test_speakers=test_speakers,
        )
    if metric == "whitened_cosine":
        return fit_whitened_cosine(
            embeddings,
            extractor=args.model,
            fit_speakers=fit_speaker_set,
            test_speakers=test_speakers,
            covariance_method=args.covariance_method,
            eigenvalue_floor=args.eigenvalue_floor,
            output_dim=args.whitened_output_dim,
        )
    if metric == "wccn_cosine":
        return fit_wccn_cosine(
            embeddings,
            speakers,
            extractor=args.model,
            fit_speakers=fit_speaker_set,
            test_speakers=test_speakers,
            covariance_method=args.covariance_method,
            eigenvalue_floor=args.eigenvalue_floor,
        )
    if metric == "lda_cosine":
        return fit_lda_cosine(
            embeddings,
            speakers,
            extractor=args.model,
            fit_speakers=fit_speaker_set,
            test_speakers=test_speakers,
            output_dim=args.lda_output_dim,
        )
    if metric == "neg_mahalanobis":
        return fit_neg_mahalanobis(
            embeddings,
            speakers,
            extractor=args.model,
            fit_speakers=fit_speaker_set,
            test_speakers=test_speakers,
            covariance_method=args.covariance_method,
            eigenvalue_floor=args.eigenvalue_floor,
            diagonal=args.mahalanobis_diagonal,
        )
    if metric == "asnorm_cosine":
        return fit_asnorm_cosine(
            embeddings,
            speakers,
            utterances,
            extractor=args.model,
            fit_speakers=fit_speaker_set,
            test_speakers=test_speakers,
            max_cohort_size=args.asnorm_max_cohort_size,
            per_speaker=args.asnorm_per_speaker,
            top_k=args.asnorm_top_k,
            sigma_floor=args.asnorm_sigma_floor,
            seed=args.cohort_seed,
        )
    if metric == "plda_llr":
        return fit_plda_placeholder(
            extractor=args.model,
            fit_speakers=fit_speaker_set,
            test_speakers=test_speakers,
        )
    raise ValueError(f"Unsupported backend metric: {metric}")


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump({"artifacts": rows}, file_obj, indent=2, ensure_ascii=True)
        file_obj.write("\n")


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    metrics = [metric for metric in parse_metric_list(args.metrics) if SCORE_SPECS[metric].requires_fit]
    if not metrics:
        raise SystemExit("No fitted backend metrics requested.")

    fit_limit = read_speaker_limit(args.fit_speakers) if args.fit_speakers else None
    test_speakers = collect_test_speakers(args.test_protocol)
    utterance_to_speaker = collect_utterance_speakers(args.protocol, fit_limit)
    fit_speaker_set = set(utterance_to_speaker.values())
    if not fit_speaker_set:
        raise SystemExit("No fit speakers selected.")
    if fit_speaker_set & test_speakers:
        raise SystemExit(
            "Refusing to fit backend: selected fit speakers intersect test speakers. "
            f"overlap_count={len(fit_speaker_set & test_speakers)}"
        )

    embeddings, speakers, utterances, load_audit = load_backend_fit_embeddings(
        utterance_to_speaker,
        embeddings_root=args.embeddings_root,
        model=args.model,
        max_utterances=args.max_utterances,
    )
    loaded_speaker_set = set(speakers)

    model_dir = args.output_dir / args.model
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict] = []
    for metric in metrics:
        output_path = model_dir / f"{metric}.npz"
        if output_path.exists() and not args.overwrite:
            raise SystemExit(f"Artifact already exists: {output_path}. Use --overwrite to replace it.")
        try:
            artifact = fit_one_metric(
                metric,
                embeddings,
                speakers,
                utterances,
                args=args,
                fit_speaker_set=loaded_speaker_set,
                test_speakers=test_speakers,
            )
        except Exception as exc:
            artifact = blocked_artifact(metric, args.model, loaded_speaker_set, test_speakers, repr(exc))
        artifact.save(output_path)
        row = {
            "metric": metric,
            "artifact": str(output_path),
            **artifact.metadata,
            **load_audit,
        }
        manifest_rows.append(row)
        print(f"wrote {metric}: {output_path} status={artifact.metadata.get('status')}", flush=True)

    write_manifest(model_dir / "fit_manifest.json", manifest_rows)
    with (model_dir / "fit_utterance_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["utterance", "speaker"])
        writer.writeheader()
        for utt, speaker in zip(utterances, speakers):
            writer.writerow({"utterance": utt, "speaker": speaker})
    print(f"wrote manifest: {model_dir / 'fit_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
