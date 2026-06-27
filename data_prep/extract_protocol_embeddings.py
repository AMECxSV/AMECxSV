#!/usr/bin/env python3
"""Extract ASV embeddings for utterances referenced by a trial protocol."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Iterable, Iterator, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EMBEDDINGS_ROOT = PROJECT_ROOT / "embedding_extraction"
sys.path.insert(0, str(EMBEDDINGS_ROOT))

from embed import MODEL_SOURCES, embeddings  # noqa: E402


DEFAULT_PROTOCOL = PROJECT_ROOT / "protocols" / "voxceleb1b.csv"
DEFAULT_AUDIO_ROOT = PROJECT_ROOT / "data" / "voxceleb1"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "embeddings"
DEFAULT_MODEL = "speechbrain_ecapa_tdnn_voxceleb"


def safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe utterance path in protocol: {value!r}")
    return path


def iter_protocol_utterances(
    protocol: Path,
    *,
    max_trials: Optional[int],
    max_utterances: Optional[int],
    num_shards: int,
    shard_index: int,
) -> Iterator[str]:
    seen: set[str] = set()
    selected = 0
    with protocol.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        required = {"enroll_utt", "test_utt"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"{protocol} must contain columns: {sorted(required)}")

        for trial_index, row in enumerate(reader, start=1):
            if max_trials is not None and trial_index > max_trials:
                break
            for column in ("enroll_utt", "test_utt"):
                utterance = row[column].strip()
                if not utterance or utterance in seen:
                    continue
                safe_relative_path(utterance)
                utterance_index = len(seen)
                seen.add(utterance)
                if utterance_index % num_shards != shard_index:
                    continue
                yield utterance
                selected += 1
                if max_utterances is not None and selected >= max_utterances:
                    return


def embedding_path(output_root: Path, model: str, utterance: str) -> Path:
    relative = safe_relative_path(utterance).with_suffix(".json")
    return output_root / model / relative


def write_embedding(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=True)
        file_obj.write("\n")
    temp_path.replace(path)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract ASV embeddings only for utterances used by a protocol CSV."
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(MODEL_SOURCES))
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda", "mps"),
        help=(
            "Inference device. auto uses CUDA when available, otherwise CPU. "
            "Use mps explicitly on Apple Silicon for supported Torch backends."
        ),
    )
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--max-utterances", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-missing-audio", action="store_true")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--torch-threads", type=int)
    parser.add_argument("--inter-op-threads", type=int)
    return parser.parse_args(argv)


def configure_runtime(args: argparse.Namespace) -> None:
    if args.num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must be in [0, num_shards)")

    if args.torch_threads is None and args.inter_op_threads is None:
        return

    import torch

    if args.torch_threads is not None:
        torch.set_num_threads(args.torch_threads)
    if args.inter_op_threads is not None:
        torch.set_num_interop_threads(args.inter_op_threads)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    configure_runtime(args)
    extractor = embeddings(device=args.device)

    extracted = 0
    skipped_existing = 0
    skipped_missing = 0

    for index, utterance in enumerate(
        iter_protocol_utterances(
            args.protocol,
            max_trials=args.max_trials,
            max_utterances=args.max_utterances,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        ),
        start=1,
    ):
        audio_path = args.audio_root / safe_relative_path(utterance)
        output_path = embedding_path(args.output_root, args.model, utterance)

        if output_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue

        if not audio_path.exists():
            if args.skip_missing_audio:
                skipped_missing += 1
                continue
            raise FileNotFoundError(f"Missing audio for {utterance}: {audio_path}")

        embedding = extractor.getEmbeddings(args.model, str(audio_path))
        write_embedding(
            output_path,
            {
                "utterance": utterance,
                "audio_path": str(audio_path),
                "model_key": args.model,
                "model": MODEL_SOURCES[args.model],
                "device": extractor.device_for_model(args.model),
                "sample_rate": 16000,
                "embedding": embedding.tolist(),
            },
        )
        extracted += 1

        if args.progress_every and index % args.progress_every == 0:
            print(
                f"processed={index} extracted={extracted} "
                f"skipped_existing={skipped_existing} skipped_missing={skipped_missing}",
                flush=True,
            )

    print(
        f"done extracted={extracted} skipped_existing={skipped_existing} "
        f"skipped_missing={skipped_missing}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
