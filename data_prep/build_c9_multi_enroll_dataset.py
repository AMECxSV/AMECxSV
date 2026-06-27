#!/usr/bin/env python3
"""Build compact C9 multi-enrollment ASV datasets.

C9 is a group-level trial table. One row claims a speaker with N enrollment
utterances and verifies one test utterance. The output intentionally keeps
traceability and model features separate:

- groups.parquet: integer references to speakers/utterances.
- speakers.parquet and utterances.parquet: compact dictionaries.
- features.parquet: numeric group-level score-consistency features only.
"""

from __future__ import annotations

import argparse
import heapq
import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORES_DIR = PROJECT_ROOT / "scores"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "c9"

DEFAULT_EMBEDDINGS = [
    "speechbrain_ecapa_tdnn_voxceleb",
    "wespeaker_resnet34_cnceleb",
    "funasr_campplus_cn_3k",
    "funasr_eres2netv2_cn_200k",
    "hf_wavlm_base_sv_voxceleb1",
    "hf_wavlm_base_plus_sv_voxceleb1",
]

SPLIT_TO_CODE = {"calibration": 0, "test": 1}
CODE_TO_SPLIT = {value: key for key, value in SPLIT_TO_CODE.items()}

REFERENCE_USECOLS = [
    "split",
    "trial_id",
    "label",
    "enroll_utt",
    "test_utt",
    "enroll_speaker",
    "test_speaker",
]
SCORE_USECOLS = ["trial_id", "score"]


@dataclass
class GroupAccumulator:
    count: int = 0
    top: list[tuple[int, int, int, str]] = field(default_factory=list)

    def add(self, enroll_utt_id: int, trial_id: str, rank: int, enroll_count: int) -> None:
        self.count += 1
        entry = (-rank, rank, enroll_utt_id, trial_id)
        if len(self.top) < enroll_count:
            heapq.heappush(self.top, entry)
            return
        if rank < self.top[0][1]:
            heapq.heapreplace(self.top, entry)

    def selected_enrollments(self) -> list[tuple[int, str]]:
        ordered = sorted(self.top, key=lambda item: item[1])
        return [(enroll_utt_id, trial_id) for _, _, enroll_utt_id, trial_id in ordered]


@dataclass(frozen=True)
class SelectedGroup:
    split_code: int
    label: int
    claim_speaker_id: int
    test_speaker_id: int
    test_utt_id: int
    enrollments: tuple[tuple[int, str], ...]
    group_hash: int
    source_pair_count: int


def stable_hash64(seed: str, *parts: object) -> int:
    text = "\x1f".join([seed, *(str(part) for part in parts)])
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8, person=b"amecxsv-c9-v1").digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def binary_entropy(probability: np.ndarray) -> np.ndarray:
    p = np.clip(probability.astype(np.float64, copy=False), 1.0e-7, 1.0 - 1.0e-7)
    entropy = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    return entropy.astype(np.float32)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values64 = values.astype(np.float64, copy=False)
    return (1.0 / (1.0 + np.exp(-values64))).astype(np.float32)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct a compact C9 multi-enrollment dataset from existing "
            "TidyVoice raw score CSV files."
        )
    )
    parser.add_argument("--scores-dir", type=Path, default=DEFAULT_SCORES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-name", default="tidyvoice_asv")
    parser.add_argument("--score-prefix", default="tidyvoice")
    parser.add_argument(
        "--score-format",
        choices=("auto", "csv", "parquet"),
        default="auto",
        help="Input score table format. auto prefers parquet when present, then CSV.",
    )
    parser.add_argument("--dataset-prefix", default="tidyvoice_c9")
    parser.add_argument(
        "--embedding",
        action="append",
        dest="embeddings",
        help="Embedding key to include. Can be repeated. Defaults to all known embeddings.",
    )
    parser.add_argument(
        "--reference-embedding",
        default=DEFAULT_EMBEDDINGS[0],
        help="Embedding score file used to read protocol columns and construct groups.",
    )
    parser.add_argument("--enroll-count", type=int, default=5)
    parser.add_argument("--groups-per-speaker-label", type=int, default=200)
    parser.add_argument("--nontarget-groups-per-speaker-pair", type=int, default=20)
    parser.add_argument(
        "--max-groups-per-split-label",
        type=int,
        help="Optional global cap per split/label, useful for smoke builds.",
    )
    parser.add_argument("--seed", default="tidyvoice_c9_multi_enroll_v1")
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def table_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return "parquet"
    if suffix == ".csv":
        return "csv"
    raise ValueError(f"Unsupported table extension for {path}; expected .csv or .parquet.")


def score_file(args: argparse.Namespace, split: str, embedding: str) -> Path:
    stem = f"{args.score_prefix}_{split}_{embedding}"
    if args.score_format == "csv":
        return args.scores_dir / f"{stem}.csv"
    if args.score_format == "parquet":
        return args.scores_dir / f"{stem}.parquet"
    parquet_path = args.scores_dir / f"{stem}.parquet"
    if parquet_path.exists():
        return parquet_path
    return args.scores_dir / f"{stem}.csv"


def iter_table_chunks(
    path: Path,
    *,
    usecols: list[str],
    chunksize: int,
) -> Iterable[pd.DataFrame]:
    if table_format(path) == "parquet":
        for batch in pq.ParquetFile(path).iter_batches(batch_size=chunksize, columns=usecols):
            yield batch.to_pandas()
    else:
        yield from pd.read_csv(path, usecols=usecols, chunksize=chunksize)


def validate_inputs(args: argparse.Namespace, embeddings: list[str], reference_embedding: str) -> None:
    required_embeddings = list(dict.fromkeys([reference_embedding, *embeddings]))
    missing = []
    for embedding in required_embeddings:
        for split in SPLIT_TO_CODE:
            path = score_file(args, split, embedding)
            if not path.exists():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing score files:\n" + "\n".join(missing))


def get_id(value: str, mapping: dict[str, int], values: list[str]) -> int:
    normalized = sys.intern(value.strip())
    if not normalized:
        raise ValueError("Encountered an empty identifier while building C9.")
    existing = mapping.get(normalized)
    if existing is not None:
        return existing
    new_id = len(values)
    mapping[normalized] = new_id
    values.append(normalized)
    return new_id


def get_utterance_id(
    value: str,
    speaker_id: int,
    mapping: dict[str, int],
    values: list[str],
    utterance_speaker_ids: list[int],
) -> int:
    utterance_id = get_id(value, mapping, values)
    if utterance_id == len(utterance_speaker_ids):
        utterance_speaker_ids.append(speaker_id)
    elif utterance_speaker_ids[utterance_id] != speaker_id:
        raise ValueError(
            f"Utterance speaker mismatch for {value!r}: "
            f"{utterance_speaker_ids[utterance_id]} vs {speaker_id}"
        )
    return utterance_id


def scan_reference_scores(
    args: argparse.Namespace,
    *,
    speaker_to_id: dict[str, int],
    speakers: list[str],
    utterance_to_id: dict[str, int],
    utterances: list[str],
    utterance_speaker_ids: list[int],
) -> tuple[dict[tuple[int, int, int, int, int], GroupAccumulator], dict[str, int]]:
    accumulators: dict[tuple[int, int, int, int, int], GroupAccumulator] = {}
    stats = {
        "rows_seen": 0,
        "rows_used": 0,
        "rows_skipped_self_pair": 0,
        "rows_skipped_empty": 0,
    }

    for split, split_code in SPLIT_TO_CODE.items():
        path = score_file(args, split, args.reference_embedding)
        split_rows = 0
        for chunk in iter_table_chunks(path, usecols=REFERENCE_USECOLS, chunksize=args.chunksize):
            for row in chunk.itertuples(index=False, name=None):
                row_split, trial_id, label_text, enroll_utt, test_utt, enroll_speaker, test_speaker = row
                stats["rows_seen"] += 1
                split_rows += 1

                row_split = str(row_split)
                trial_id = str(trial_id)
                enroll_utt = str(enroll_utt)
                test_utt = str(test_utt)
                enroll_speaker = str(enroll_speaker)
                test_speaker = str(test_speaker)
                if row_split != split:
                    raise ValueError(f"{path} contains split={row_split!r}, expected {split!r}.")
                if (
                    not trial_id
                    or pd.isna(label_text)
                    or str(label_text).strip() == ""
                    or not enroll_utt
                    or not test_utt
                ):
                    stats["rows_skipped_empty"] += 1
                    continue
                if enroll_utt == test_utt:
                    stats["rows_skipped_self_pair"] += 1
                    continue

                label = int(label_text)
                claim_speaker_id = get_id(enroll_speaker, speaker_to_id, speakers)
                test_speaker_id = get_id(test_speaker, speaker_to_id, speakers)
                enroll_utt_id = get_utterance_id(
                    enroll_utt,
                    claim_speaker_id,
                    utterance_to_id,
                    utterances,
                    utterance_speaker_ids,
                )
                test_utt_id = get_utterance_id(
                    test_utt,
                    test_speaker_id,
                    utterance_to_id,
                    utterances,
                    utterance_speaker_ids,
                )

                key = (split_code, claim_speaker_id, test_utt_id, test_speaker_id, label)
                rank = stable_hash64(
                    args.seed,
                    "enroll",
                    split_code,
                    claim_speaker_id,
                    test_utt_id,
                    test_speaker_id,
                    label,
                    enroll_utt_id,
                )
                accumulator = accumulators.setdefault(key, GroupAccumulator())
                accumulator.add(enroll_utt_id, trial_id, rank, args.enroll_count)
                stats["rows_used"] += 1

        print(f"scanned split={split} rows={split_rows} groups_seen={len(accumulators)}", flush=True)

    return accumulators, stats


def select_groups(
    accumulators: dict[tuple[int, int, int, int, int], GroupAccumulator],
    args: argparse.Namespace,
) -> tuple[list[SelectedGroup], dict[str, object]]:
    candidates: list[SelectedGroup] = []
    too_few = 0
    for key, accumulator in accumulators.items():
        if accumulator.count < args.enroll_count or len(accumulator.top) < args.enroll_count:
            too_few += 1
            continue
        split_code, claim_speaker_id, test_utt_id, test_speaker_id, label = key
        enrollments = tuple(accumulator.selected_enrollments())
        group_hash = stable_hash64(args.seed, "group", *key, *(item[0] for item in enrollments))
        candidates.append(
            SelectedGroup(
                split_code=split_code,
                label=label,
                claim_speaker_id=claim_speaker_id,
                test_speaker_id=test_speaker_id,
                test_utt_id=test_utt_id,
                enrollments=enrollments,
                group_hash=group_hash,
                source_pair_count=accumulator.count,
            )
        )

    candidates.sort(key=lambda group: group.group_hash)
    per_claim_label: dict[tuple[int, int, int], int] = {}
    per_nontarget_pair: dict[tuple[int, int, int], int] = {}
    per_split_label: dict[tuple[int, int], int] = {}
    selected: list[SelectedGroup] = []
    dropped_claim_cap = 0
    dropped_pair_cap = 0
    dropped_split_label_cap = 0

    for group in candidates:
        split_label_key = (group.split_code, group.label)
        if (
            args.max_groups_per_split_label is not None
            and per_split_label.get(split_label_key, 0) >= args.max_groups_per_split_label
        ):
            dropped_split_label_cap += 1
            continue

        claim_key = (group.split_code, group.claim_speaker_id, group.label)
        if per_claim_label.get(claim_key, 0) >= args.groups_per_speaker_label:
            dropped_claim_cap += 1
            continue

        if group.label == 0:
            pair_key = (group.split_code, group.claim_speaker_id, group.test_speaker_id)
            if per_nontarget_pair.get(pair_key, 0) >= args.nontarget_groups_per_speaker_pair:
                dropped_pair_cap += 1
                continue
            per_nontarget_pair[pair_key] = per_nontarget_pair.get(pair_key, 0) + 1

        per_claim_label[claim_key] = per_claim_label.get(claim_key, 0) + 1
        per_split_label[split_label_key] = per_split_label.get(split_label_key, 0) + 1
        selected.append(group)

    selected.sort(key=lambda group: (group.split_code, group.group_hash))
    summary = {
        "group_keys_seen": len(accumulators),
        "candidate_groups_with_enough_enrollments": len(candidates),
        "selected_groups": len(selected),
        "dropped": {
            "too_few_enrollments": too_few,
            "groups_per_speaker_label_cap": dropped_claim_cap,
            "nontarget_speaker_pair_cap": dropped_pair_cap,
            "max_groups_per_split_label_cap": dropped_split_label_cap,
        },
        "selected_by_split_label": count_by_split_label(selected),
    }
    return selected, summary


def count_by_split_label(groups: list[SelectedGroup]) -> dict[str, dict[str, int]]:
    counts = {
        split: {"target": 0, "nontarget": 0}
        for split in SPLIT_TO_CODE
    }
    for group in groups:
        split = CODE_TO_SPLIT[group.split_code]
        label_name = "target" if group.label == 1 else "nontarget"
        counts[split][label_name] += 1
    return counts


def build_compact_tables(
    selected: list[SelectedGroup],
    speakers: list[str],
    utterances: list[str],
    utterance_speaker_ids: list[int],
    enroll_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, tuple[int, int]]]:
    used_old_speakers: set[int] = set()
    used_old_utterances: set[int] = set()
    for group in selected:
        used_old_speakers.add(group.claim_speaker_id)
        used_old_speakers.add(group.test_speaker_id)
        used_old_utterances.add(group.test_utt_id)
        for enroll_utt_id, _ in group.enrollments:
            used_old_utterances.add(enroll_utt_id)

    old_to_new_speaker = {
        old_id: new_id
        for new_id, old_id in enumerate(sorted(used_old_speakers, key=lambda item: speakers[item]))
    }
    old_to_new_utterance = {
        old_id: new_id
        for new_id, old_id in enumerate(sorted(used_old_utterances, key=lambda item: utterances[item]))
    }

    speaker_rows = [
        {"speaker_id": new_id, "speaker": speakers[old_id]}
        for old_id, new_id in old_to_new_speaker.items()
    ]
    speaker_rows.sort(key=lambda row: row["speaker_id"])

    utterance_rows = []
    for old_id, new_id in old_to_new_utterance.items():
        old_speaker_id = utterance_speaker_ids[old_id]
        utterance_rows.append(
            {
                "utterance_id": new_id,
                "speaker_id": old_to_new_speaker[old_speaker_id],
                "utterance": utterances[old_id],
            }
        )
    utterance_rows.sort(key=lambda row: row["utterance_id"])

    group_rows = []
    trial_lookup: dict[str, tuple[int, int]] = {}
    for group_id, group in enumerate(selected):
        row = {
            "group_id": group_id,
            "split": group.split_code,
            "label": group.label,
            "claim_speaker_id": old_to_new_speaker[group.claim_speaker_id],
            "test_speaker_id": old_to_new_speaker[group.test_speaker_id],
            "test_utt_id": old_to_new_utterance[group.test_utt_id],
        }
        for position, (enroll_utt_id, trial_id) in enumerate(group.enrollments, start=1):
            row[f"enroll_utt_id_{position}"] = old_to_new_utterance[enroll_utt_id]
            if trial_id in trial_lookup:
                raise ValueError(f"Selected trial_id appears in more than one C9 slot: {trial_id}")
            trial_lookup[trial_id] = (group_id, position - 1)
        if len(group.enrollments) != enroll_count:
            raise ValueError(f"Group {group_id} has {len(group.enrollments)} enrollments.")
        group_rows.append(row)

    groups_frame = pd.DataFrame(group_rows)
    groups_frame = groups_frame.astype(
        {
            "group_id": "uint64",
            "split": "uint8",
            "label": "uint8",
            "claim_speaker_id": "int32",
            "test_speaker_id": "int32",
            "test_utt_id": "int32",
            **{f"enroll_utt_id_{index}": "int32" for index in range(1, enroll_count + 1)},
        }
    )
    speakers_frame = pd.DataFrame(speaker_rows).astype({"speaker_id": "int32", "speaker": "string"})
    utterances_frame = pd.DataFrame(utterance_rows).astype(
        {"utterance_id": "int32", "speaker_id": "int32", "utterance": "string"}
    )
    return groups_frame, speakers_frame, utterances_frame, trial_lookup


def collect_scores(
    args: argparse.Namespace,
    embedding: str,
    trial_lookup: dict[str, tuple[int, int]],
    group_count: int,
) -> tuple[np.ndarray, dict[str, int]]:
    scores = np.full((group_count, args.enroll_count), np.nan, dtype=np.float32)
    selected_ids = set(trial_lookup)
    found = 0
    rows_scanned = 0

    for split in SPLIT_TO_CODE:
        path = score_file(args, split, embedding)
        for chunk in iter_table_chunks(path, usecols=SCORE_USECOLS, chunksize=args.chunksize):
            rows_scanned += int(chunk.shape[0])
            chunk["trial_id"] = chunk["trial_id"].astype(str)
            mask = chunk["trial_id"].isin(selected_ids)
            if not bool(mask.any()):
                continue
            matched = chunk.loc[mask, SCORE_USECOLS]
            for trial_id, score in matched.itertuples(index=False, name=None):
                slot = trial_lookup.get(trial_id)
                if slot is None:
                    continue
                group_id, position = slot
                scores[group_id, position] = float(score)
                found += 1

    missing = int(np.isnan(scores).sum())
    stats = {
        "rows_scanned": rows_scanned,
        "selected_pair_slots": int(scores.size),
        "scores_found": found,
        "scores_missing": missing,
    }
    if missing:
        raise ValueError(f"{embedding} is missing {missing} selected C9 scores.")
    return scores, stats


def rates_for_thresholds(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels_bool = labels.astype(bool)
    target_n = int(labels_bool.sum())
    nontarget_n = int(labels_bool.shape[0] - target_n)
    if target_n == 0 or nontarget_n == 0:
        raise ValueError("Both target and nontarget calibration examples are required.")

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
    return thresholds, pmiss, pfa


def eer_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    thresholds, pmiss, pfa = rates_for_thresholds(scores, labels)
    diff = pfa - pmiss
    crossing = np.flatnonzero(diff >= 0.0)
    if crossing.size == 0:
        best = int(np.argmin(np.abs(diff)))
        return float(thresholds[best]), float((pmiss[best] + pfa[best]) / 2.0)
    right = int(crossing[0])
    if right == 0:
        return float(thresholds[0]), float((pmiss[0] + pfa[0]) / 2.0)
    left = right - 1
    denom = diff[left] - diff[right]
    alpha = 0.0 if denom == 0.0 else float(np.clip(diff[left] / denom, 0.0, 1.0))
    threshold = float(thresholds[left] + alpha * (thresholds[right] - thresholds[left]))
    eer_value = float(pfa[left] + alpha * (pfa[right] - pfa[left]))
    return threshold, eer_value


def fit_score_transform(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    threshold, pair_eer = eer_threshold(scores, labels)
    model = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000)
    model.fit(scores.reshape(-1, 1), labels.astype(np.int8, copy=False))
    return {
        "score_threshold": threshold,
        "pair_eer_pct": 100.0 * pair_eer,
        "platt_a": float(model.coef_[0, 0]),
        "platt_b": float(model.intercept_[0]),
    }


def add_embedding_features(
    features: pd.DataFrame,
    embedding: str,
    scores: np.ndarray,
    labels: np.ndarray,
    split_codes: np.ndarray,
) -> dict[str, float]:
    calibration_mask = split_codes == SPLIT_TO_CODE["calibration"]
    pair_scores = scores[calibration_mask].reshape(-1).astype(np.float64, copy=False)
    pair_labels = np.repeat(labels[calibration_mask], scores.shape[1])
    transform = fit_score_transform(pair_scores, pair_labels)

    threshold = transform["score_threshold"]
    posterior = sigmoid(transform["platt_a"] * scores + transform["platt_b"])
    vote_frac = np.mean(scores >= threshold, axis=1).astype(np.float32)
    posterior_entropy = binary_entropy(posterior)

    prefix = f"{embedding}__"
    sorted_scores = np.sort(scores, axis=1).astype(np.float32, copy=False)
    for index in range(sorted_scores.shape[1]):
        features[f"{prefix}score_sorted_{index + 1}"] = sorted_scores[:, index]
    features[prefix + "score_mean"] = np.mean(scores, axis=1).astype(np.float32)
    features[prefix + "score_median"] = np.median(scores, axis=1).astype(np.float32)
    features[prefix + "score_min"] = np.min(scores, axis=1).astype(np.float32)
    features[prefix + "score_max"] = np.max(scores, axis=1).astype(np.float32)
    features[prefix + "score_std"] = np.std(scores, axis=1).astype(np.float32)
    features[prefix + "vote_frac"] = vote_frac
    features[prefix + "vote_entropy"] = binary_entropy(vote_frac)
    features[prefix + "post_mean"] = np.mean(posterior, axis=1).astype(np.float32)
    features[prefix + "post_std"] = np.std(posterior, axis=1).astype(np.float32)
    features[prefix + "post_entropy_mean"] = np.mean(posterior_entropy, axis=1).astype(np.float32)
    features[prefix + "post_entropy_max"] = np.max(posterior_entropy, axis=1).astype(np.float32)
    return transform


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    frame.to_parquet(path, engine="pyarrow", compression="zstd", index=False)


def output_paths(output_dir: Path, prefix: str) -> dict[str, Path]:
    return {
        "groups": output_dir / f"{prefix}_groups.parquet",
        "speakers": output_dir / f"{prefix}_speakers.parquet",
        "utterances": output_dir / f"{prefix}_utterances.parquet",
        "features": output_dir / f"{prefix}_features.parquet",
        "transforms": output_dir / f"{prefix}_feature_transforms.json",
        "summary": output_dir / f"{prefix}_summary.json",
    }


def ensure_outputs(paths: dict[str, Path], overwrite: bool) -> None:
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise SystemExit(
            "C9 output already exists. Use --overwrite to replace:\n"
            + "\n".join(str(path) for path in existing)
        )
    paths["groups"].parent.mkdir(parents=True, exist_ok=True)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if args.enroll_count < 2:
        raise ValueError("--enroll-count must be at least 2")
    embeddings = args.embeddings or DEFAULT_EMBEDDINGS
    if args.reference_embedding not in embeddings:
        embeddings = [args.reference_embedding, *embeddings]
    embeddings = list(dict.fromkeys(embeddings))

    args.scores_dir = args.scores_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    paths = output_paths(args.output_dir, args.dataset_prefix)
    ensure_outputs(paths, args.overwrite)
    validate_inputs(args, embeddings, args.reference_embedding)

    speaker_to_id: dict[str, int] = {}
    speakers: list[str] = []
    utterance_to_id: dict[str, int] = {}
    utterances: list[str] = []
    utterance_speaker_ids: list[int] = []

    accumulators, scan_stats = scan_reference_scores(
        args,
        speaker_to_id=speaker_to_id,
        speakers=speakers,
        utterance_to_id=utterance_to_id,
        utterances=utterances,
        utterance_speaker_ids=utterance_speaker_ids,
    )
    selected, selection_summary = select_groups(accumulators, args)
    if not selected:
        raise ValueError("No C9 groups selected. Relax sampling caps or inspect input scores.")
    print(f"selected C9 groups={len(selected)}", flush=True)

    groups_frame, speakers_frame, utterances_frame, trial_lookup = build_compact_tables(
        selected,
        speakers,
        utterances,
        utterance_speaker_ids,
        args.enroll_count,
    )

    features = pd.DataFrame(
        {
            "group_id": groups_frame["group_id"].to_numpy(dtype=np.uint64, copy=False),
            "split": groups_frame["split"].to_numpy(dtype=np.uint8, copy=False),
            "label": groups_frame["label"].to_numpy(dtype=np.uint8, copy=False),
        }
    )
    labels = features["label"].to_numpy(dtype=np.int8, copy=False)
    split_codes = features["split"].to_numpy(dtype=np.uint8, copy=False)

    transforms: dict[str, dict[str, float | int | str]] = {}
    score_stats: dict[str, dict[str, int]] = {}
    for embedding in embeddings:
        print(f"collecting C9 scores embedding={embedding}", flush=True)
        scores, stats = collect_scores(args, embedding, trial_lookup, len(groups_frame))
        transform = add_embedding_features(features, embedding, scores, labels, split_codes)
        transforms[embedding] = {
            "embedding": embedding,
            "enroll_count": args.enroll_count,
            **transform,
        }
        score_stats[embedding] = stats

    feature_columns_by_embedding = {
        embedding: [column for column in features.columns if column.startswith(f"{embedding}__")]
        for embedding in embeddings
    }

    write_parquet(groups_frame, paths["groups"])
    write_parquet(speakers_frame, paths["speakers"])
    write_parquet(utterances_frame, paths["utterances"])
    write_parquet(features, paths["features"])

    transform_payload = {
        "dataset": args.dataset_name,
        "calib": "C9",
        "seed": args.seed,
        "score_prefix": args.score_prefix,
        "score_format": args.score_format,
        "split_encoding": CODE_TO_SPLIT,
        "feature_columns_by_embedding": feature_columns_by_embedding,
        "transforms": transforms,
    }
    with paths["transforms"].open("w", encoding="utf-8") as handle:
        json.dump(transform_payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    summary = {
        "dataset": args.dataset_name,
        "calib": "C9",
        "description": "Compact multi-enrollment group dataset for C9-2/C9-3.",
        "seed": args.seed,
        "score_prefix": args.score_prefix,
        "score_format": args.score_format,
        "enroll_count": args.enroll_count,
        "sampling": {
            "groups_per_speaker_label": args.groups_per_speaker_label,
            "nontarget_groups_per_speaker_pair": args.nontarget_groups_per_speaker_pair,
            "max_groups_per_split_label": args.max_groups_per_split_label,
        },
        "split_encoding": CODE_TO_SPLIT,
        "embeddings": embeddings,
        "scan_stats": scan_stats,
        "selection": selection_summary,
        "outputs": {name: str(path) for name, path in paths.items()},
        "rows": {
            "groups": int(groups_frame.shape[0]),
            "features": int(features.shape[0]),
            "speakers": int(speakers_frame.shape[0]),
            "utterances": int(utterances_frame.shape[0]),
            "selected_pair_slots": int(len(trial_lookup)),
        },
        "columns": {
            "groups": list(groups_frame.columns),
            "features": list(features.columns),
            "speakers": list(speakers_frame.columns),
            "utterances": list(utterances_frame.columns),
        },
        "score_collection": score_stats,
    }
    with paths["summary"].open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)
        handle.write("\n")

    print(f"wrote C9 groups: {paths['groups']}", flush=True)
    print(f"wrote C9 features: {paths['features']}", flush=True)
    print(f"wrote C9 summary: {paths['summary']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
