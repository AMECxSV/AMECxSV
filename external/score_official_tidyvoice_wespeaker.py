#!/usr/bin/env python3
"""Score AMEC TidyVoice splits with the official SimAM-ResNet34 WeSpeaker model."""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import wave
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torchaudio.compliance.kaldi as kaldi
import yaml
from tqdm import tqdm

from external_common import git_commit, make_run_id, resolve_path, utc_timestamp

from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.checkpoint import load_checkpoint


SOURCE_SYSTEM = "official_tidyvoice_simam_resnet34"
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
    parser.add_argument("--model-dir", type=Path, default=Path("external/checkpoints/official_tidyvoice_simam_resnet34"))
    parser.add_argument("--audio-root", type=Path, default=Path("data/tidyvoice/TidyVoiceX_ASV/TidyVoiceX_Dev"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--embedding-batch-size", type=int, default=1)
    parser.add_argument("--parquet-row-group-size", type=int, default=100_000)
    parser.add_argument("--max-trials", type=int)
    parser.add_argument("--max-utterances", type=int)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--checkpoint-id", default="areffarhadi/Resnet34-tidyvoiceX-ASV")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_dir: Path, device: torch.device) -> torch.nn.Module:
    model_dir = resolve_path(model_dir)
    config_path = model_dir / "config.yaml"
    model_path = model_dir / "avg_model.pt"
    if not model_path.exists():
        model_path = model_dir / "models" / "avg_model.pt"
    if not config_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Expected config.yaml and avg_model.pt under {model_dir}")
    configs = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = get_speaker_model(configs["model"])(**configs["model_args"])
    load_checkpoint(model, str(model_path))
    model.eval()
    model.to(device)
    return model


def safe_relative_path(value: str) -> Path:
    path = Path(str(value))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe utterance path in protocol: {value!r}")
    return path


def wav_path(audio_root: Path, utterance: str) -> Path:
    return resolve_path(audio_root) / safe_relative_path(utterance)


def read_wav_pcm(path: Path) -> tuple[torch.Tensor, int]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frames = handle.getnframes()
        raw = handle.readframes(frames)
    if sample_width != 2:
        raise ValueError(f"Unsupported WAV sample width {sample_width} in {path}; expected 16-bit PCM")
    data = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return torch.from_numpy(data).unsqueeze(0), sample_rate


def compute_fbank(
    waveform: torch.Tensor,
    *,
    sample_rate: int,
    num_mel_bins: int = 80,
    frame_length: int = 25,
    frame_shift: int = 10,
    cmn: bool = True,
) -> torch.Tensor:
    feat = kaldi.fbank(
        waveform,
        num_mel_bins=num_mel_bins,
        frame_length=frame_length,
        frame_shift=frame_shift,
        sample_frequency=sample_rate,
        window_type="hamming",
    )
    if cmn:
        feat = feat - torch.mean(feat, 0)
    return feat


def extract_one(model: torch.nn.Module, path: Path, device: torch.device) -> np.ndarray:
    waveform, sample_rate = read_wav_pcm(path)
    if sample_rate != 16000:
        raise ValueError(f"Unexpected sample rate {sample_rate} in {path}; official TidyVoice wavs should be 16 kHz")
    feats = compute_fbank(waveform.to(torch.float), sample_rate=sample_rate, cmn=True)
    feats = feats.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(feats)
        outputs = outputs[-1] if isinstance(outputs, tuple) else outputs
    return outputs[0].detach().cpu().numpy().astype(np.float32, copy=False)


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


def save_cache(path: Path | None, embeddings: dict[str, np.ndarray]) -> None:
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
                "embeddings": embeddings,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    tmp.replace(path)


def cosine_normalized(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 0.0 or not math.isfinite(denom):
        return math.nan
    cosine = float(np.dot(left, right) / denom)
    return (cosine + 1.0) / 2.0


def extract_embeddings(
    *,
    model: torch.nn.Module,
    device: torch.device,
    audio_root: Path,
    utterances: list[str],
    cache: dict[str, np.ndarray],
    cache_path: Path | None,
    progress_every: int,
) -> dict[str, np.ndarray]:
    missing = [utt for utt in utterances if utt not in cache]
    progress = tqdm(missing, desc="extract official embeddings", unit="utt")
    for index, utt in enumerate(progress, start=1):
        path = wav_path(audio_root, utt)
        if not path.exists():
            raise FileNotFoundError(path)
        cache[utt] = extract_one(model, path, device)
        if progress_every and index % progress_every == 0:
            save_cache(cache_path, cache)
            progress.set_postfix({"cached": len(cache)})
    save_cache(cache_path, cache)
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
                    cosine_normalized(embeddings[str(enroll)], embeddings[str(test)])
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
                score = cosine_normalized(embeddings[row["enroll_utt"]], embeddings[row["test_utt"]])
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
    output = resolve_path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"Output exists: {output}. Use --overwrite to replace it.")
    device = choose_device(args.device)
    model = load_model(args.model_dir, device)
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
        utterances=utterances,
        cache=cache,
        cache_path=args.embedding_cache,
        progress_every=args.progress_every,
    )
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
