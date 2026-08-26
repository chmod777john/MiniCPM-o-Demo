#!/usr/bin/env python3
"""Validate a complete O5 safetensors bundle without a GPU model run.

The check constructs both the outer Demo model and the native Qwen LLM on
``meta``.  It then loads every outer shard into CPU tensors and verifies that
the same index contains every native Qwen key needed by TP2.  No checkpoint is
loaded, so this is also useful on a CPU-only packaging machine.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM

from core.deploy.weights import _weight_map, load_safetensors_into
from modeling.o5.modeling_minicpmo_unified import MiniCPMO
from o5_paths import DEFAULT_MODEL_PATH


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    args = parser.parse_args()
    bundle = Path(args.bundle).expanduser().resolve()
    model_path = Path(args.model_path).expanduser().resolve()

    index = _weight_map(bundle)
    outer_config = AutoConfig.from_pretrained(str(model_path), trust_remote_code=True)
    # Validation is intentionally CPU-only; do not let a deployment config
    # selecting FlashAttention make meta construction require CUDA.
    outer_config._attn_implementation = "eager"
    with init_empty_weights():
        outer = MiniCPMO(outer_config)
    missing, unexpected = load_safetensors_into(outer, bundle)
    if missing or unexpected:
        raise RuntimeError(f"outer bundle mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    if any(not tensor.is_meta for tensor in outer.state_dict().values()) is False:
        raise RuntimeError("outer loader did not materialize any tensors")

    llm_config_path = bundle / "llm"
    llm_config = AutoConfig.from_pretrained(str(llm_config_path), trust_remote_code=True)
    llm_config._attn_implementation = "eager"
    with init_empty_weights():
        llm = AutoModelForCausalLM.from_config(llm_config)
    expected_llm = set(llm.state_dict())
    artifact_llm = {key for key in index if key in expected_llm}
    missing_llm = sorted(expected_llm - artifact_llm)
    if missing_llm:
        raise RuntimeError(f"TP2 LLM view missing {len(missing_llm)} keys: {missing_llm[:5]}")

    manifest_path = bundle / "o5_artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    print(
        json.dumps(
            {
                "bundle": str(bundle),
                "manifest_format": manifest.get("format"),
                "index_keys": len(index),
                "outer_keys": len(outer.state_dict()),
                "llm_keys": len(expected_llm),
                "outer_missing": len(missing),
                "outer_unexpected": len(unexpected),
                "llm_missing": len(missing_llm),
                "status": "ok",
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
