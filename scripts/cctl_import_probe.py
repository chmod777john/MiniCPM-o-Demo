#!/usr/bin/env python3
import importlib.util
import sys


def main():
    print(sys.version)
    modules = [
        "torch",
        "torchaudio",
        "torchvision",
        "transformers",
        "accelerate",
        "safetensors",
        "fastapi",
        "uvicorn",
        "httpx",
        "websockets",
        "librosa",
        "soundfile",
        "pydantic",
        "numpy",
        "stepaudio2",
        "minicpmo",
        "edge_tts",
        "decord",
        "moviepy",
        "onnxruntime",
        "hyperpyyaml",
        "yaml",
    ]
    for module in modules:
        print(module, bool(importlib.util.find_spec(module)))
    try:
        import torch

        print(
            "torch_version",
            torch.__version__,
            torch.version.cuda,
            torch.cuda.is_available(),
            torch.cuda.device_count(),
        )
    except Exception as exc:
        print("torch_error", type(exc).__name__, exc)


if __name__ == "__main__":
    main()
