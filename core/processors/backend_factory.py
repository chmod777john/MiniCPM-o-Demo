"""Factory for the PyTorch backend implementation."""

from __future__ import annotations

import os
from typing import Any, Dict

from o5_paths import resolve_artifacts


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def create_backend(config: Dict[str, Any]) -> Any:
    """Create the PyTorch inference backend."""

    import os as _os
    artifacts = resolve_artifacts(
        model_path=config.get("model_path"),
        pt_path=config.get("pt_path"),
        weights_dir=config.get("weights_dir"),
        backbone_dir=config.get("backbone_dir"),
        assets_dir=config.get("assets_dir"),
    )
    config = {**config, **{key: value for key, value in artifacts.items() if value is not None}}
    # Drive the deployment-mode framework (core.deploy) from config; UnifiedProcessor._load_model
    # reads these env vars. Default single_eager keeps the original single-card eager path.
    _dm = _os.environ.get("O5_DEPLOY_MODE") or config.get("deployment_mode") or "single_eager"
    _os.environ["O5_DEPLOY_MODE"] = _dm
    if not _os.environ.get("O5_BACKBONE_DIR") and config.get("backbone_dir"):
        _os.environ["O5_BACKBONE_DIR"] = str(config["backbone_dir"])
    if not _os.environ.get("O5_ASSETS_DIR") and config.get("assets_dir"):
        _os.environ["O5_ASSETS_DIR"] = str(config["assets_dir"])
    if not _os.environ.get("O5_LLM_CACHE") and config.get("llm_cache_len"):
        _os.environ["O5_LLM_CACHE"] = str(config["llm_cache_len"])

    from core.processors.pytorch_backend import PyTorchBackend

    return PyTorchBackend(
        model_path=config["model_path"],
        gpu_id=config["gpu_id"],
        pt_path=config.get("pt_path"),
        weights_dir=config.get("weights_dir"),
        ref_audio_path=config.get("ref_audio_path"),
        duplex_pause_timeout=config.get("duplex_pause_timeout", 60.0),
        compile=config.get("compile", False),
        chat_vocoder=config.get("chat_vocoder", "token2wav"),
        attn_implementation=config.get("attn_implementation", "auto"),
        preload_both_tts=_env_bool("O5_PRELOAD_BOTH_TTS", config.get("preload_both_tts", True)),
    )
