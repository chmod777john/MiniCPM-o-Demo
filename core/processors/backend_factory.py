"""Factory for the PyTorch backend implementation."""

from __future__ import annotations

from typing import Any, Dict


def create_backend(config: Dict[str, Any]) -> Any:
    """Create the PyTorch inference backend from one explicit deployment."""

    import os as _os
    resolved_config = dict(config)
    profile_path = (
        _os.environ.get("FC_DEPLOYMENT_PROFILE")
        or resolved_config.get("fc_deployment_profile_path")
    )
    if profile_path:
        from core.fc_duplex.profiles import (
            O5FcDeploymentProfile,
            apply_fc_deployment_profile_environment,
            load_fc_deployment_profile,
        )

        profile = load_fc_deployment_profile(profile_path)
        apply_fc_deployment_profile_environment(profile)
        resolved_config.update(
            {
                "model_path": profile.model_path,
                "pt_path": profile.pt_path,
                "model_family": profile.model_family,
                "deployment_mode": profile.deployment_mode,
                "fc_deployment_profile_path": str(profile_path),
                "ref_audio_path": profile.reference_audio_path,
            }
        )
        if isinstance(profile, O5FcDeploymentProfile):
            resolved_config.update(
                {
                    "backbone_dir": profile.backbone_dir,
                    "llm_cache_len": profile.llm_cache,
                    "attn_implementation": profile.attn_implementation,
                }
            )

    # Drive the deployment-mode framework (core.deploy) from config; UnifiedProcessor._load_model
    # reads these env vars. Default single_eager keeps the original single-card eager path.
    _dm = (
        _os.environ.get("O5_DEPLOY_MODE")
        or resolved_config.get("deployment_mode")
        or "single_eager"
    )
    _os.environ["O5_DEPLOY_MODE"] = _dm
    if (
        not _os.environ.get("O5_BACKBONE_DIR")
        and resolved_config.get("backbone_dir")
    ):
        _os.environ["O5_BACKBONE_DIR"] = str(resolved_config["backbone_dir"])
    if (
        not _os.environ.get("O5_LLM_CACHE")
        and resolved_config.get("llm_cache_len")
    ):
        _os.environ["O5_LLM_CACHE"] = str(resolved_config["llm_cache_len"])

    from core.processors.pytorch_backend import PyTorchBackend

    return PyTorchBackend(
        model_path=resolved_config["model_path"],
        gpu_id=resolved_config["gpu_id"],
        pt_path=resolved_config.get("pt_path"),
        ref_audio_path=resolved_config.get("ref_audio_path"),
        duplex_pause_timeout=resolved_config.get("duplex_pause_timeout", 60.0),
        compile=resolved_config.get("compile", False),
        chat_vocoder=resolved_config.get("chat_vocoder", "token2wav"),
        attn_implementation=resolved_config.get("attn_implementation", "auto"),
        fc_model_family=resolved_config.get("model_family", "o5"),
    )
