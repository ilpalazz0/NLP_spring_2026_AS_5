from __future__ import annotations

import torch


def get_torch_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_generation_dtype():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def describe_device() -> dict[str, str | bool]:
    device = get_torch_device()
    if device == "cuda":
        return {
            "device": device,
            "cuda_available": True,
            "gpu_name": torch.cuda.get_device_name(0),
        }
    return {
        "device": "cpu",
        "cuda_available": False,
        "gpu_name": "",
    }
