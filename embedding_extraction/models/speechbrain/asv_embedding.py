from pathlib import Path

import numpy as np
import torch
from speechbrain.inference.speaker import EncoderClassifier

from models.device import resolve_torch_device


ASV_MODELS = {
    "speechbrain_ecapa_tdnn_voxceleb": "speechbrain/spkrec-ecapa-voxceleb",
}


class SpeechBrainASVEmbeddings:
    def __init__(
        self,
        device: str | None = None,
        savedir_root: str | None = None,
        normalize: bool = False,
    ):
        self.device = resolve_torch_device(device)
        self.savedir_root = Path(savedir_root) if savedir_root else self._default_savedir_root()
        self.normalize = normalize
        self.models = {}

    def getEmbeddings(self, model: str, audio: str) -> np.ndarray:
        encoder = self._get_model(model)
        signal = encoder.load_audio(audio)

        with torch.no_grad():
            embedding = encoder.encode_batch(signal, normalize=self.normalize)

        return embedding.squeeze().detach().cpu().numpy().astype(np.float32)

    def _get_model(self, model: str):
        source = self._resolve_model(model)
        if source not in self.models:
            savedir = self.savedir_root / source.replace("/", "_")
            self.models[source] = EncoderClassifier.from_hparams(
                source=source,
                savedir=str(savedir),
                run_opts={"device": self.device},
            )
        return self.models[source]

    @staticmethod
    def _default_savedir_root() -> Path:
        return Path(__file__).resolve().parents[2] / "pretrained_models" / "speechbrain"

    @staticmethod
    def _resolve_model(model: str) -> str:
        key = model.lower()
        if key in ASV_MODELS:
            return ASV_MODELS[key]
        raise ValueError(
            f"Unknown SpeechBrain ASV model: {model}. "
            f"Use one of {list(ASV_MODELS)}."
        )


_DEFAULT_EXTRACTOR = None


def getEmbeddings(model: str, audio: str) -> np.ndarray:
    global _DEFAULT_EXTRACTOR
    if _DEFAULT_EXTRACTOR is None:
        _DEFAULT_EXTRACTOR = SpeechBrainASVEmbeddings()
    return _DEFAULT_EXTRACTOR.getEmbeddings(model, audio)
