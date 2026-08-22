"""Default model-artifact locations used by the Demo loaders and runners.

The vendored ``MiniCPMO45`` directory is the code/config source.  Runtime
weights are resolved separately so a release can ship safetensors without
requiring callers to pass a canonical model checkout or a legacy ``.pt`` file.
Every default remains overridable by an environment variable or an explicit
argument for evaluation and multi-checkpoint deployments.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "MiniCPMO45"
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "weights"
DEFAULT_FULL_BUNDLE = Path(
    "/user/weihongliang/o5_weights/o5_full_hf_"
    "chenmoye_minicpm_5o_moe_omni_long_context_sft_stage2_sft2_8k_"
    "audio_online_process_on_online_audio_process_v2_iter_100"
)
DEFAULT_CHECKPOINT_PATH = Path(
    "/user/weihongliang/o5_weights/"
    "chenmoye_minicpm_5o_moe_omni_long_context_sft_stage2_sft2_8k_"
    "audio_online_process_on_online_audio_process_v2_iter_100.pt"
)
DEFAULT_BACKBONE_DIR = Path(
    "/user/weihongliang/o5_weights/"
    "o5_backbone_hf_chenmoye_minicpm_5o_moe_omni_long_context_sft_stage2_"
    "sft2_8k_audio_online_process_on_online_audio_process_v2_iter_100"
)
DEFAULT_ASSETS_DIR = Path("/user/weihongliang/MiniCPM-o-4_6/assets")


def _path(value: Optional[str | os.PathLike[str]]) -> Optional[Path]:
    if value is None:
        return None
    text = os.fspath(value).strip()
    return Path(text).expanduser() if text else None


def has_hf_weights(directory: Path) -> bool:
    return any(
        (directory / name).is_file()
        for name in ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin")
    ) or any(directory.glob("model-*.safetensors"))


def resolve_model_path(value: Optional[str] = None) -> Path:
    """Resolve model code/config, defaulting to the vendored Demo copy."""
    return (
        _path(value)
        or _path(os.environ.get("O5_MODEL_PATH"))
        or DEFAULT_MODEL_PATH
    ).resolve()


def resolve_weights_dir(
    value: Optional[str] = None,
    *,
    model_path: Optional[str | os.PathLike[str]] = None,
) -> Optional[Path]:
    """Find a complete HF safetensors directory, if one is available."""
    candidates = [
        _path(value),
        _path(os.environ.get("O5_WEIGHTS_DIR")),
        _path(model_path),
        DEFAULT_WEIGHTS_DIR,
        DEFAULT_FULL_BUNDLE,
    ]
    for candidate in candidates:
        if candidate is not None and has_hf_weights(candidate):
            return candidate.resolve()
    return None


def resolve_checkpoint_path(
    value: Optional[str] = None,
    *,
    weights_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Resolve the legacy checkpoint only when no safetensors bundle exists."""
    if value is not None and str(value).strip():
        return Path(value).expanduser().resolve()
    if weights_dir is not None:
        return None
    configured = _path(os.environ.get("O5_PT_PATH") or os.environ.get("O5_CHECKPOINT_PATH"))
    if configured is not None:
        return configured.resolve()
    if DEFAULT_CHECKPOINT_PATH.is_file():
        return DEFAULT_CHECKPOINT_PATH
    return None


def resolve_backbone_dir(
    value: Optional[str] = None,
    *,
    weights_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Resolve the TP LLM directory, including a bundle's ``llm`` view."""
    candidates = [
        _path(value),
        _path(os.environ.get("O5_BACKBONE_DIR")),
        (weights_dir / "llm") if weights_dir is not None else None,
        DEFAULT_BACKBONE_DIR,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_dir() and (
            has_hf_weights(candidate) or (candidate / "config.json").is_file()
        ):
            return candidate.resolve()
    return None


def resolve_assets_dir(value: Optional[str] = None) -> Optional[Path]:
    candidates = [
        _path(value),
        _path(os.environ.get("O5_ASSETS_DIR")),
        DEFAULT_MODEL_PATH / "assets",
        DEFAULT_ASSETS_DIR,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_dir():
            return candidate.resolve()
    return None


def resolve_artifacts(
    *,
    model_path: Optional[str] = None,
    pt_path: Optional[str] = None,
    weights_dir: Optional[str] = None,
    backbone_dir: Optional[str] = None,
    assets_dir: Optional[str] = None,
) -> dict[str, Optional[str]]:
    model = resolve_model_path(model_path)
    weights = resolve_weights_dir(weights_dir, model_path=model)
    checkpoint = resolve_checkpoint_path(pt_path, weights_dir=weights)
    backbone = resolve_backbone_dir(backbone_dir, weights_dir=weights)
    assets = resolve_assets_dir(assets_dir)
    return {
        "model_path": str(model),
        "pt_path": str(checkpoint) if checkpoint is not None else None,
        "weights_dir": str(weights) if weights is not None else None,
        "backbone_dir": str(backbone) if backbone is not None else None,
        "assets_dir": str(assets) if assets is not None else None,
    }
