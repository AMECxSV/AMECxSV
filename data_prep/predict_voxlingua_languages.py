from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd
import torch
import torchaudio
from tqdm import tqdm


MODEL_SOURCE = "speechbrain/lang-id-voxlingua107-ecapa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen SpeechBrain VoxLingua107 ECAPA language-ID model "
            "once per unique enrollment/test utterance."
        )
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/voxlingua107_language_predictions.csv"),
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def read_protocol(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(
            path,
            columns=["enroll_utt", "test_utt"],
        )
    return pd.read_csv(
        path,
        usecols=["enroll_utt", "test_utt"],
    )


def load_classifier(device: str):
    try:
        from speechbrain.inference.classifiers import EncoderClassifier
    except ImportError:
        from speechbrain.pretrained import EncoderClassifier

    return EncoderClassifier.from_hparams(
        source=MODEL_SOURCE,
        run_opts={"device": device},
    )


def predicted_label(
    classifier,
    audio_path: Path,
    *,
    device: str,
) -> str:
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0, keepdim=True)
    model_rate = int(classifier.hparams.sample_rate)
    if sample_rate != model_rate:
        waveform = torchaudio.functional.resample(
            waveform,
            sample_rate,
            model_rate,
        )
    waveform = waveform.to(device)
    with torch.inference_mode():
        _, _, _, labels = classifier.classify_batch(waveform)
    label = labels[0] if isinstance(labels, (list, tuple)) else labels
    return str(label)


def main() -> None:
    args = parse_args()
    protocol = read_protocol(args.protocol)
    utterances = sorted(
        set(protocol["enroll_utt"].astype(str))
        | set(protocol["test_utt"].astype(str))
    )
    classifier = load_classifier(args.device)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["utterance_id", "predicted_language"],
        )
        writer.writeheader()
        for utterance in tqdm(
            utterances,
            desc="frozen VoxLingua107 LID",
            unit="utterance",
        ):
            audio_path = args.audio_root / utterance
            if not audio_path.exists():
                raise FileNotFoundError(audio_path)
            writer.writerow(
                {
                    "utterance_id": utterance,
                    "predicted_language": predicted_label(
                        classifier,
                        audio_path,
                        device=args.device,
                    ),
                }
            )
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
