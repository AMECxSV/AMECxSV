from __future__ import annotations


def resolve_torch_device(requested: str | None = None, *, allow_mps: bool = True) -> str:
    import torch

    device = (requested or "auto").lower()
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    if device == "cpu":
        return "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested, but torch.cuda.is_available() is false.")
        return "cuda"
    if device == "mps":
        if not allow_mps:
            raise ValueError("MPS was requested, but this backend does not support MPS.")
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise ValueError("MPS was requested, but torch.backends.mps.is_available() is false.")
        return "mps"

    raise ValueError(f"Unsupported device: {requested!r}. Use auto, cpu, cuda, or mps.")


def resolve_cpu_cuda_device(requested: str | None = None) -> str:
    import torch

    device = (requested or "auto").lower()
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("CUDA was requested, but torch.cuda.is_available() is false.")
        return "cuda"
    if device == "mps":
        raise ValueError("MPS was requested, but this backend only supports CPU/CUDA.")
    raise ValueError(f"Unsupported device: {requested!r}. Use auto, cpu, cuda, or mps.")
