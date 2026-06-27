from pathlib import Path

import numpy as np
import torch

from models.device import resolve_torch_device


ASV_MODELS = {
    "hf_wavlm_base_sv_voxceleb1": "microsoft/wavlm-base-sv",
    "hf_wavlm_base_plus_sv_voxceleb1": "microsoft/wavlm-base-plus-sv",
}


class HuggingFaceASVEmbeddings:
    def __init__(self, device: str | None = None, savedir_root: str | None = None):
        self.device = resolve_torch_device(device)
        self.savedir_root = Path(savedir_root) if savedir_root else self._default_savedir_root()
        self.models = {}

    def getEmbeddings(self, model: str, audio: str) -> np.ndarray:
        feature_extractor, sv_model = self._get_model(model)

        import librosa

        waveform, _ = librosa.load(audio, sr=16000, mono=True)
        inputs = feature_extractor(
            waveform,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            embedding = sv_model(**inputs).embeddings
            embedding = torch.nn.functional.normalize(embedding, dim=-1)

        return embedding.squeeze().detach().cpu().numpy().astype(np.float32)

    def _get_model(self, model: str):
        source = self._resolve_model(model)
        if source not in self.models:
            from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

            self.savedir_root.mkdir(parents=True, exist_ok=True)
            feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
                source,
                cache_dir=str(self.savedir_root),
            )
            sv_model = WavLMForXVector.from_pretrained(
                source,
                cache_dir=str(self.savedir_root),
            ).to(self.device)
            sv_model.eval()
            self.models[source] = (feature_extractor, sv_model)
        return self.models[source]

    @staticmethod
    def _default_savedir_root() -> Path:
        return Path(__file__).resolve().parents[2] / "pretrained_models" / "huggingface"

    @staticmethod
    def _resolve_model(model: str) -> str:
        key = model.lower()
        if key in ASV_MODELS:
            return ASV_MODELS[key]
        raise ValueError(
            f"Unknown HuggingFace ASV model: {model}. "
            f"Use one of {list(ASV_MODELS)}."
        )
