import os
from pathlib import Path

import numpy as np

from models.device import resolve_torch_device


ASV_MODELS = {
    "wespeaker_resnet34_cnceleb": "chinese",
}


class WeSpeakerASVEmbeddings:
    def __init__(self, device: str | None = None, savedir_root: str | None = None):
        self.device = resolve_torch_device(device, allow_mps=False)
        self.savedir_root = Path(savedir_root) if savedir_root else self._default_savedir_root()
        self.models = {}

    def getEmbeddings(self, model: str, audio: str) -> np.ndarray:
        speaker = self._get_model(model)
        embedding = speaker.extract_embedding(audio)
        if embedding is None:
            raise ValueError(f"No speech detected in audio: {audio}")
        return embedding.squeeze().detach().cpu().numpy().astype(np.float32)

    def _get_model(self, model: str):
        source = self._resolve_model(model)
        if source not in self.models:
            self.savedir_root.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("WESPEAKER_HOME", str(self.savedir_root))

            import wespeaker

            speaker = wespeaker.load_model(source)
            speaker.set_device(self.device)
            self.models[source] = speaker
        return self.models[source]

    @staticmethod
    def _default_savedir_root() -> Path:
        return Path(__file__).resolve().parents[2] / "pretrained_models" / "wespeaker"

    @staticmethod
    def _resolve_model(model: str) -> str:
        key = model.lower()
        if key in ASV_MODELS:
            return ASV_MODELS[key]
        raise ValueError(
            f"Unknown WeSpeaker ASV model: {model}. "
            f"Use one of {list(ASV_MODELS)}."
        )
