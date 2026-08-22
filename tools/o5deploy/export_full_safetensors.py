#!/usr/bin/env python3
"""Convert a legacy full MiniCPM-O5 checkpoint into HF safetensors shards.

The output contains every parameter and buffer in the Demo model, including
VPM, APM, TTS, resampler, merger, and LLM wrapper state.  It is intentionally
an offline conversion utility; the serving path never needs to deserialize
the 75GB PT file once this bundle has been published.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from accelerate import init_empty_weights  # noqa: E402
from safetensors.torch import save_file  # noqa: E402
from transformers import AutoConfig  # noqa: E402
from transformers.modeling_utils import split_torch_state_dict_into_shards  # noqa: E402

from MiniCPMO45.modeling_minicpmo_unified import MiniCPMO  # noqa: E402
from o5_paths import (  # noqa: E402
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_MODEL_PATH,
)


def load_state_dict(path: Path) -> dict:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    for name in ("state_dict", "model", "module"):
        if isinstance(state, dict) and isinstance(state.get(name), dict):
            return state[name]
    return state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--pt-path", default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-shard-size", default="5GB")
    return parser.parse_args()


def _write_bundle(model, output_dir: Path, max_shard_size: str) -> dict[str, str]:
    """Write one complete bundle with a TP-compatible LLM namespace.

    The outer Demo module names LLM parameters ``llm.model.*`` while native
    Qwen loading expects ``model.*``.  Store the latter once and let the outer
    loader alias it back under ``llm.``.  This is the key detail that avoids a
    duplicated standalone TP backbone.
    """
    state = {
        (key[4:] if key.startswith("llm.") else key): value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
    }
    split = split_torch_state_dict_into_shards(state, max_shard_size=max_shard_size)
    for filename, keys in split.filename_to_tensors.items():
        save_file(
            {key: state[key] for key in keys},
            str(output_dir / filename),
            metadata={"format": "pt"},
        )
    index = {
        "metadata": split.metadata,
        "weight_map": split.tensor_to_filename,
    }
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"keys": str(len(state)), "shards": str(len(split.filename_to_tensors))}


def main() -> int:
    args = parse_args()
    model_path = Path(args.model_path).expanduser().resolve()
    pt_path = Path(args.pt_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if not (model_path / "config.json").is_file():
        raise SystemExit(f"model config not found: {model_path / 'config.json'}")
    if not pt_path.is_file():
        raise SystemExit(f"checkpoint not found: {pt_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    config._name_or_path = str(model_path)
    config.name_or_path = str(model_path)
    with init_empty_weights():
        model = MiniCPMO(config)

    print(f"[export] loading PT metadata/tensors: {pt_path}", flush=True)
    state = load_state_dict(pt_path)
    info = model.load_state_dict(state, strict=False, assign=True)
    print(
        f"[export] loaded keys={len(state)} missing={len(info.missing_keys)} "
        f"unexpected={len(info.unexpected_keys)}",
        flush=True,
    )
    if info.missing_keys:
        print(f"[export] missing sample: {info.missing_keys[:5]}", flush=True)
    if info.unexpected_keys:
        print(f"[export] unexpected sample: {info.unexpected_keys[:5]}", flush=True)
    if info.missing_keys:
        raise RuntimeError("refusing to export an incomplete model")

    model.config.save_pretrained(str(output_dir))
    (output_dir / "llm").mkdir(exist_ok=True)
    model.llm.config.save_pretrained(str(output_dir / "llm"))
    bundle_info = _write_bundle(model, output_dir, args.max_shard_size)
    try:
        source_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        source_commit = os.environ.get("DEMO_GIT_COMMIT")
    manifest = {
        "format": "o5-complete-hf-safetensors-v1",
        "scope": "complete MiniCPMO model state",
        "llm_namespace": "native Qwen model.* stored once; outer loader aliases llm.*",
        "source_checkpoint": str(pt_path),
        "source_model_path": str(model_path),
        "source_commit": source_commit,
        "keys": int(bundle_info["keys"]),
        "shards": int(bundle_info["shards"]),
        "elapsed_seconds": round(time.time() - started, 3),
        "tp_backbone_note": "TP2 reads the same root shards with llm/config.json; no second LLM safetensor copy is required.",
    }
    (output_dir / "o5_artifact_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[export] wrote {output_dir} in {manifest['elapsed_seconds']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
