import json
from pathlib import Path

import numpy as np

MODEL_SOURCES = {
    "speechbrain_ecapa_tdnn_voxceleb": "speechbrain/spkrec-ecapa-voxceleb",
    "wespeaker_resnet34_cnceleb": "chinese",
    "funasr_campplus_cn_3k": "iic/speech_campplus_sv_zh-cn_16k-common",
    "funasr_eres2netv2_cn_200k": "iic/speech_eres2netv2_sv_zh-cn_16k-common",
    "hf_wavlm_base_sv_voxceleb1": "microsoft/wavlm-base-sv",
    "hf_wavlm_base_plus_sv_voxceleb1": "microsoft/wavlm-base-plus-sv",
}


def _speechbrain(device):
    from models.speechbrain.asv_embedding import SpeechBrainASVEmbeddings

    return SpeechBrainASVEmbeddings(device=device)


def _wespeaker(device):
    from models.wespeaker.asv_embedding import WeSpeakerASVEmbeddings

    return WeSpeakerASVEmbeddings(device=device)


def _funasr(device):
    from models.funasr.asv_embedding import FunASRASVEmbeddings

    return FunASRASVEmbeddings(device=device)


def _huggingface(device):
    from models.huggingface.asv_embedding import HuggingFaceASVEmbeddings

    return HuggingFaceASVEmbeddings(device=device)


class embeddings:
    def __init__(self, device: str | None = None):
        self.device = device
        self.model_factories = {
            "speechbrain_ecapa_tdnn_voxceleb": lambda: _speechbrain(device),
            "wespeaker_resnet34_cnceleb": lambda: _wespeaker(device),
            "funasr_campplus_cn_3k": lambda: _funasr(device),
            "funasr_eres2netv2_cn_200k": lambda: _funasr(device),
            "hf_wavlm_base_sv_voxceleb1": lambda: _huggingface(device),
            "hf_wavlm_base_plus_sv_voxceleb1": lambda: _huggingface(device),
        }
        self.models = {}

    def _get_extractor(self, model: str):
        if model not in self.model_factories:
            raise ValueError(f"Unsupported model: {model}")
        if model not in self.models:
            self.models[model] = self.model_factories[model]()
        return self.models[model]

    def getEmbeddings(self, model: str, audio: str) -> np.ndarray:
        return self._get_extractor(model).getEmbeddings(model, audio)

    def device_for_model(self, model: str) -> str:
        return getattr(self._get_extractor(model), "device", "unknown")

    def getEmbeddingsAll(self,input_folder:str,output_folder:str)->None:
        # {
        #     "audio_path": "datasets/xxx.wav",
        #     "model": "speechbrain/spkrec-ecapa-voxceleb",
        #     "sample_rate": 16000,
        #     "embedding": [float, ...]
        # }
        # output one pretty-printed json for each audio file, named by audio path
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        folders = sorted(input_folder.rglob("*.wav"))

        for m in self.model_factories:
            for file in folders:
                emb = self.getEmbeddings(m, str(file))
                relative_name = file.relative_to(input_folder).with_suffix("")
                output_name = str(relative_name).replace("\\", "_").replace("/", "_")
                output_file = output_folder / m / f"{output_name}.json"
                output_file.parent.mkdir(parents=True, exist_ok=True)

                data = {
                    "audio_path": str(file),
                    "model": MODEL_SOURCES[m],
                    "sample_rate": 16000,
                }
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write("{\n")
                    f.write(f'  "audio_path": {json.dumps(data["audio_path"])},\n')
                    f.write(f'  "model": {json.dumps(data["model"])},\n')
                    f.write(f'  "sample_rate": {data["sample_rate"]},\n')
                    f.write(f'  "embedding": {json.dumps(emb.tolist())}\n')
                    f.write("}\n")

    def checkmodel(self,model):
        #run embedding and check if diff and same are different
        same_1 = self.getEmbeddings(model, "datasets/samples/same/same_01.wav")
        same_2 = self.getEmbeddings(model, "datasets/samples/same/same_02.wav")
        diff_1 = self.getEmbeddings(model, "datasets/samples/diff/diff_01.wav")
        diff_2 = self.getEmbeddings(model, "datasets/samples/diff/diff_02.wav")

        same_score = float(np.dot(same_1, same_2) / (np.linalg.norm(same_1) * np.linalg.norm(same_2)))
        diff_score = float(np.dot(diff_1, diff_2) / (np.linalg.norm(diff_1) * np.linalg.norm(diff_2)))
        success = same_score > diff_score
        print(f"{model} same: {same_score:.4f}, diff: {diff_score:.4f}, success: {success}")
        return success

    def checkAllmodel(self):
        failed = []
        for model in self.model_factories:
            try:
                success = self.checkmodel(model)
            except Exception as error:
                success = False
                print(f"{model} failed: {error}")
            if not success:
                failed.append(model)

        all_success = len(failed) == 0
        print(f"all success: {all_success}")
        if failed:
            print(f"failed models: {failed}")
        return all_success, failed

if __name__=='__main__':
    emb=embeddings()
    #emb.getEmbeddingsAll('datasets','embeddings/embeddings')
    emb.checkAllmodel()
