#!/usr/bin/env python3
"""Score AMEC TidyVoice trials with the LI-MSV w2v-BERT GRL checkpoint."""

from __future__ import annotations

import argparse
import contextlib
import csv
import math
import pickle
import sys
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch
from tqdm import tqdm

from external_common import git_commit, make_run_id, resolve_path, utc_timestamp


SOURCE_SYSTEM = "limsv_w2vbert"
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
    "enroll_audio_exists",
    "test_audio_exists",
]


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-dir", type=Path, default=Path("external/repos/LI-MSV-TidyVoice2026"))
    parser.add_argument("--checkpoint", type=Path, default=Path("external/checkpoints/limsv_w2vbert/TidyVoice2026_GRL/ckpt_0044.pth"))
    parser.add_argument("--train-yaml", type=Path, help="Defaults to train.yaml next to --checkpoint.")
    parser.add_argument("--audio-root", type=Path, default=Path("data/tidyvoice/TidyVoiceX_ASV/TidyVoiceX_Dev"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--parquet-row-group-size", type=int, default=100_000)
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--max-utterances", type=int)
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--torch-threads", type=int, help="Limit CPU torch threads for repeatable local runs.")
    parser.add_argument("--checkpoint-id", default="zl389/w2v-bert-2.0_SV/TidyVoice2026_GRL/ckpt_0044.pth")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return device


def add_limsv_paths(repo_dir: Path) -> None:
    repo_dir = resolve_path(repo_dir)
    paths = [
        repo_dir,
        repo_dir / "recipes" / "DeepASV",
        repo_dir / "deeplab" / "pretrained" / "audio2vector" / "module" / "transformers" / "src",
    ]
    for path in reversed(paths):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


def load_model(
    *,
    repo_dir: Path,
    checkpoint: Path,
    train_yaml: Path | None,
    device: torch.device,
) -> tuple[torch.nn.Module, int, int, bool, dict[str, object]]:
    add_limsv_paths(repo_dir)
    from deeplab.utils.fileio import read_hyperyaml

    checkpoint = resolve_path(checkpoint)
    train_yaml = resolve_path(train_yaml or checkpoint.parent / "train.yaml")
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if not train_yaml.exists():
        raise FileNotFoundError(train_yaml)

    hparams = read_hyperyaml(path=str(train_yaml))
    modules = hparams["modules"]
    model = modules["spk_model"]
    ckpt_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ckpt_state_dict = ckpt_data["modules"]["spk_model"]
    curr_state_dict = model.state_dict()
    loaded = 0
    mismatched: list[str] = []
    for key in curr_state_dict:
        if key in ckpt_state_dict and curr_state_dict[key].shape == ckpt_state_dict[key].shape:
            curr_state_dict[key] = ckpt_state_dict[key]
            loaded += 1
        else:
            mismatched.append(key)
    model.load_state_dict(curr_state_dict)
    model.eval()
    model.to(device)

    sample_rate = int(hparams["sample_rate"])
    max_len = int(float(hparams["max_valid_dur"]) * sample_rate)
    metadata = {
        "loaded_spk_model_tensors": loaded,
        "mismatched_spk_model_tensors": len(mismatched),
        "mismatched_examples": mismatched[:25],
    }
    return model, sample_rate, max_len, bool(hparams.get("use_amp", False)), metadata


def safe_relative_path(value: str) -> Path:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe utterance path in protocol: {value!r}")
    return path


def wav_path(audio_root: Path, utterance: str) -> Path:
    return resolve_path(audio_root) / safe_relative_path(utterance)


def read_audio(path: Path, sample_rate: int, max_len: int) -> np.ndarray:
    signal, sr = sf.read(path, dtype="float32")
    if signal.ndim == 2:
        signal = signal[:, 0]
    if sr != sample_rate:
        signal = librosa.resample(signal, orig_sr=sr, target_sr=sample_rate)
    return np.asarray(signal[:max_len], dtype=np.float32)


def extract_embedding(
    *,
    model: torch.nn.Module,
    path: Path,
    sample_rate: int,
    max_len: int,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    signal = read_audio(path, sample_rate, max_len)
    aud_inputs = torch.from_numpy(signal).float()
    if device.type == "cuda" and use_amp:
        autocast = torch.autocast("cuda", dtype=torch.bfloat16)
    else:
        autocast = contextlib.nullcontext()
    with torch.no_grad(), autocast:
        output = model(aud_inputs)
        embedding = output[0] if isinstance(output, (tuple, list)) else output
    if embedding.ndim == 2:
        embedding = embedding[0]
    return embedding.float().detach().cpu().numpy().astype(np.float32, copy=False)


def collect_utterances(protocol: Path, *, chunksize: int, max_trials: int | None, max_utterances: int | None) -> list[str]:
    protocol = resolve_path(protocol)
    utterances: set[str] = set()
    seen_trials = 0
    for chunk in pd.read_csv(protocol, usecols=["enroll_utt", "test_utt"], chunksize=chunksize):
        if max_trials is not None:
            remaining = max_trials - seen_trials
            if remaining <= 0:
                break
            chunk = chunk.head(remaining)
        seen_trials += int(len(chunk))
        utterances.update(chunk["enroll_utt"].astype(str))
        utterances.update(chunk["test_utt"].astype(str))
        if max_utterances is not None and len(utterances) >= max_utterances:
            return sorted(utterances)[:max_utterances]
    return sorted(utterances)


def load_cache(path: Path | None) -> dict[str, np.ndarray]:
    if path is None:
        return {}
    path = resolve_path(path)
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    embeddings = payload.get("embeddings", payload)
    return {str(key): np.asarray(value, dtype=np.float32) for key, value in embeddings.items()}


def save_cache(path: Path | None, embeddings: dict[str, np.ndarray], metadata: dict[str, object]) -> None:
    if path is None:
        return
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        pickle.dump(
            {
                "source_system": SOURCE_SYSTEM,
                "timestamp_utc": utc_timestamp(),
                "git_commit": git_commit(),
                "embedding_count": len(embeddings),
                "model_metadata": metadata,
                "embeddings": embeddings,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    tmp.replace(path)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 0.0 or not math.isfinite(denom):
        return math.nan
    return float(np.dot(left, right) / denom)


def extract_embeddings(
    *,
    model: torch.nn.Module,
    device: torch.device,
    audio_root: Path,
    sample_rate: int,
    max_len: int,
    use_amp: bool,
    utterances: list[str],
    cache: dict[str, np.ndarray],
    cache_path: Path | None,
    progress_every: int,
    metadata: dict[str, object],
) -> dict[str, np.ndarray]:
    missing = [utt for utt in utterances if utt not in cache]
    progress = tqdm(missing, desc="extract LI-MSV embeddings", unit="utt", mininterval=30.0, smoothing=0.0)
    for index, utt in enumerate(progress, start=1):
        path = wav_path(audio_root, utt)
        if not path.exists():
            raise FileNotFoundError(path)
        cache[utt] = extract_embedding(
            model=model,
            path=path,
            sample_rate=sample_rate,
            max_len=max_len,
            device=device,
            use_amp=use_amp,
        )
        if progress_every and index % progress_every == 0:
            save_cache(cache_path, cache, metadata)
            progress.set_postfix({"cached": len(cache)})
    save_cache(cache_path, cache, metadata)
    return cache


def output_fields(input_fields: list[str]) -> list[str]:
    fields = [field for field in TRIAL_COLUMNS if field in input_fields]
    for field in input_fields:
        if field not in fields:
            fields.append(field)
    for field in ["score", "source_system", "checkpoint_id", "run_id", "timestamp_utc", "git_commit"]:
        if field not in fields:
            fields.append(field)
    return fields


def write_scores(
    *,
    protocol: Path,
    output: Path,
    embeddings: dict[str, np.ndarray],
    chunksize: int,
    max_trials: int | None,
    row_group_size: int,
    checkpoint_id: str,
) -> int:
    protocol = resolve_path(protocol)
    output = resolve_path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    run_id = make_run_id(SOURCE_SYSTEM)
    timestamp = utc_timestamp()
    commit = git_commit()
    written = 0

    if output.suffix.lower() in {".parquet", ".pq"}:
        writer: pq.ParquetWriter | None = None
        try:
            for chunk in pd.read_csv(protocol, chunksize=chunksize):
                if max_trials is not None:
                    remaining = max_trials - written
                    if remaining <= 0:
                        break
                    chunk = chunk.head(remaining)
                scores = [
                    cosine(embeddings[str(enroll)], embeddings[str(test)])
                    for enroll, test in zip(chunk["enroll_utt"], chunk["test_utt"])
                ]
                chunk = chunk.copy()
                chunk["score"] = scores
                chunk["source_system"] = SOURCE_SYSTEM
                chunk["checkpoint_id"] = checkpoint_id
                chunk["run_id"] = run_id
                chunk["timestamp_utc"] = timestamp
                chunk["git_commit"] = commit
                fields = output_fields(chunk.columns.tolist())
                chunk = chunk[fields]
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(output, table.schema, compression="zstd")
                writer.write_table(table, row_group_size=row_group_size)
                written += int(len(chunk))
        finally:
            if writer is not None:
                writer.close()
    else:
        with protocol.open("r", encoding="utf-8", newline="") as input_obj, output.open("w", encoding="utf-8", newline="") as output_obj:
            reader = csv.DictReader(input_obj)
            if reader.fieldnames is None:
                raise ValueError(f"{protocol} has no header")
            fields = output_fields(reader.fieldnames)
            writer = csv.DictWriter(output_obj, fieldnames=fields)
            writer.writeheader()
            for row in reader:
                if max_trials is not None and written >= max_trials:
                    break
                score = cosine(embeddings[row["enroll_utt"]], embeddings[row["test_utt"]])
                row.update(
                    {
                        "score": f"{score:.8f}",
                        "source_system": SOURCE_SYSTEM,
                        "checkpoint_id": checkpoint_id,
                        "run_id": run_id,
                        "timestamp_utc": timestamp,
                        "git_commit": commit,
                    }
                )
                writer.writerow({field: row.get(field, "") for field in fields})
                written += 1
    return written


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    output = resolve_path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {output}. Use --overwrite to replace it.")
    device = choose_device(args.device)
    model, sample_rate, max_len, use_amp, metadata = load_model(
        repo_dir=args.repo_dir,
        checkpoint=args.checkpoint,
        train_yaml=args.train_yaml,
        device=device,
    )
    metadata.update(
        {
            "device": str(device),
            "sample_rate": sample_rate,
            "max_len_samples": max_len,
            "use_amp_from_train_yaml": use_amp,
        }
    )
    utterances = collect_utterances(
        args.protocol,
        chunksize=args.chunksize,
        max_trials=args.max_trials,
        max_utterances=args.max_utterances,
    )
    cache = load_cache(args.embedding_cache)
    extract_embeddings(
        model=model,
        device=device,
        audio_root=args.audio_root,
        sample_rate=sample_rate,
        max_len=max_len,
        use_amp=use_amp,
        utterances=utterances,
        cache=cache,
        cache_path=args.embedding_cache,
        progress_every=args.progress_every,
        metadata=metadata,
    )
    missing = [utt for utt in utterances if utt not in cache]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} embeddings after extraction; examples: {missing[:5]}")
    written = write_scores(
        protocol=args.protocol,
        output=args.output,
        embeddings=cache,
        chunksize=args.chunksize,
        max_trials=args.max_trials,
        row_group_size=args.parquet_row_group_size,
        checkpoint_id=args.checkpoint_id,
    )
    print(
        f"done source_system={SOURCE_SYSTEM} device={device} "
        f"utterances={len(utterances)} scores={written} output={output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
