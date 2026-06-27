import os
from pathlib import Path

import numpy as np

from models.device import resolve_cpu_cuda_device


ASV_MODELS = {
    "funasr_campplus_cn_3k": "iic/speech_campplus_sv_zh-cn_16k-common",
    "funasr_eres2netv2_cn_200k": "iic/speech_eres2netv2_sv_zh-cn_16k-common",
}


class FunASRASVEmbeddings:
    def __init__(self, device: str | None = None, savedir_root: str | None = None):
        self.device = resolve_cpu_cuda_device(device)
        self.savedir_root = Path(savedir_root) if savedir_root else self._default_savedir_root()
        self.models = {}

    def getEmbeddings(self, model: str, audio: str) -> np.ndarray:
        sv_model = self._get_model(model)
        result = sv_model.generate(input=audio)
        embedding = result[0]["spk_embedding"]
        return self._to_numpy(embedding)

    def _get_model(self, model: str):
        source = self._resolve_model(model)
        if source not in self.models:
            self.savedir_root.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("MODELSCOPE_CACHE", str(self.savedir_root))

            from funasr import AutoModel

            self.models[source] = AutoModel(model=source, device=self.device)
        return self.models[source]

    @staticmethod
    def _default_savedir_root() -> Path:
        return Path(__file__).resolve().parents[2] / "pretrained_models" / "funasr"

    @staticmethod
    def _resolve_model(model: str) -> str:
        key = model.lower()
        if key in ASV_MODELS:
            return ASV_MODELS[key]
        raise ValueError(
            f"Unknown FunASR ASV model: {model}. "
            f"Use one of {list(ASV_MODELS)}."
        )

    @staticmethod
    def _to_numpy(embedding) -> np.ndarray:
        if hasattr(embedding, "detach"):
            embedding = embedding.detach().cpu().numpy()
        return np.asarray(embedding).squeeze().astype(np.float32)
