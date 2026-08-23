"""Default model-artifact locations used by the Demo loaders and runners.

The vendored ``MiniCPMO45`` directory is the code/config source.  Runtime
weights are a complete safetensors bundle.  The serving path deliberately has
no model-checkout, legacy ``.pt`` or standalone-backbone compatibility mode.
Only the bundle and runtime assets can be selected.
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
DEFAULT_ASSETS_DIR = Path("/user/weihongliang/MiniCPM-o-4_6/assets")


def _path(value: Optional[str | os.PathLike[str]]) -> Optional[Path]:
    if value is None:
        return None
    text = os.fspath(value).strip()
    return Path(text).expanduser() if text else None


def has_complete_bundle(directory: Path) -> bool:
    """Return whether *directory* is the published full Demo artifact."""
    return (
        (directory / "config.json").is_file()
        and (directory / "model.safetensors.index.json").is_file()
        and (directory / "llm" / "config.json").is_file()
        and any(directory.glob("model-*.safetensors"))
    )


def resolve_model_path() -> Path:
    """Return the code/config shipped with this Demo checkout."""
    return DEFAULT_MODEL_PATH.resolve()


def resolve_weights_dir(value: Optional[str] = None) -> Path:
    """Resolve and validate the complete HF safetensors bundle."""
    candidates = [
        _path(value),
        _path(os.environ.get("O5_WEIGHTS_DIR")),
        DEFAULT_WEIGHTS_DIR,
        DEFAULT_FULL_BUNDLE,
    ]
    for candidate in candidates:
        if candidate is not None and has_complete_bundle(candidate):
            return candidate.resolve()
    rendered = ", ".join(str(candidate) for candidate in candidates if candidate is not None)
    raise FileNotFoundError(
        "A complete safetensors bundle is required; checked: " + rendered
    )


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
    weights_dir: Optional[str] = None,
    assets_dir: Optional[str] = None,
) -> dict[str, str | None]:
    model = resolve_model_path()
    weights = resolve_weights_dir(weights_dir)
    assets = resolve_assets_dir(assets_dir)
    return {
        "model_path": str(model),
        "weights_dir": str(weights),
        "assets_dir": str(assets) if assets is not None else None,
    }
