"""Factory for the PyTorch backend implementation."""

from __future__ import annotations

from typing import Any, Dict


def create_backend(config: Dict[str, Any]) -> Any:
    """Create the PyTorch inference backend."""

    import os as _os
    # Drive the deployment-mode framework (core.deploy) from config; UnifiedProcessor._load_model
    # reads these env vars. Default single_eager keeps the original single-card eager path.
    _dm = config.get("deployment_mode") or "single_eager"
    _os.environ["O5_DEPLOY_MODE"] = _dm
    if config.get("backbone_dir"): _os.environ["O5_BACKBONE_DIR"] = str(config["backbone_dir"])
    if config.get("llm_cache_len"): _os.environ["O5_LLM_CACHE"] = str(config["llm_cache_len"])

    from core.processors.pytorch_backend import PyTorchBackend

    return PyTorchBackend(
        model_path=config["model_path"],
        gpu_id=config["gpu_id"],
        pt_path=config.get("pt_path"),
        ref_audio_path=config.get("ref_audio_path"),
        duplex_pause_timeout=config.get("duplex_pause_timeout", 60.0),
        compile=config.get("compile", False),
        chat_vocoder=config.get("chat_vocoder", "token2wav"),
        attn_implementation=config.get("attn_implementation", "auto"),
    )
