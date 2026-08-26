"""Factory for the PyTorch backend implementation."""

from __future__ import annotations

from typing import Any, Dict

from o5_paths import resolve_artifacts


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
                "model_family": profile.model_family,
                "deployment_mode": profile.deployment_mode,
                "fc_deployment_profile_path": str(profile_path),
                "ref_audio_path": profile.reference_audio_path,
            }
        )
        if isinstance(profile, O5FcDeploymentProfile):
            if not profile.weights_dir:
                raise RuntimeError(
                    "This O5 FC profile still uses legacy model_path/pt_path/backbone_dir; "
                    "export a complete safetensors bundle and set weights_dir before serving."
                )
            resolved_config.update(
                {
                    "weights_dir": profile.weights_dir,
                    "assets_dir": profile.assets_dir,
                    "llm_cache_len": profile.llm_cache,
                    "attn_implementation": profile.attn_implementation,
                }
            )
        else:
            raise RuntimeError(
                "The integrated O5 backend does not serve O45 legacy profiles; "
                "use the O45 checkout for model_family='o45'."
            )

    # Resolve once at the factory boundary so every backend process records and
    # consumes the same complete artifact.  No model_path/.pt/backbone fallback.
    artifacts = resolve_artifacts(
        weights_dir=resolved_config.get("weights_dir"),
        assets_dir=resolved_config.get("assets_dir"),
    )
    resolved_config.update(artifacts)

    # Drive the deployment-mode framework (core.deploy) from config; UnifiedProcessor._load_model
    # reads these env vars. Default single_eager keeps the original single-card eager path.
    _dm = (
        _os.environ.get("O5_DEPLOY_MODE")
        or resolved_config.get("deployment_mode")
        or "single_eager"
    )
    _os.environ["O5_DEPLOY_MODE"] = _dm
    _os.environ.pop("O5_BACKBONE_DIR", None)
    if resolved_config.get("weights_dir"):
        _os.environ["O5_WEIGHTS_DIR"] = str(resolved_config["weights_dir"])
    if resolved_config.get("assets_dir"):
        _os.environ["O5_ASSETS_DIR"] = str(resolved_config["assets_dir"])
    if (
        not _os.environ.get("O5_LLM_CACHE")
        and resolved_config.get("llm_cache_len")
    ):
        _os.environ["O5_LLM_CACHE"] = str(resolved_config["llm_cache_len"])

    from core.processors.pytorch_backend import PyTorchBackend

    return PyTorchBackend(
        gpu_id=resolved_config["gpu_id"],
        weights_dir=resolved_config["weights_dir"],
        assets_dir=resolved_config.get("assets_dir"),
        ref_audio_path=resolved_config.get("ref_audio_path"),
        duplex_pause_timeout=resolved_config.get("duplex_pause_timeout", 60.0),
        compile=resolved_config.get("compile", False),
        chat_vocoder=resolved_config.get("chat_vocoder", "token2wav"),
        attn_implementation=resolved_config.get("attn_implementation", "auto"),
        fc_model_family=resolved_config.get("model_family", "o5"),
    )
