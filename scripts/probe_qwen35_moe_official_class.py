#!/usr/bin/env python3
"""Minimal text-only probe using the public Transformers Qwen3.5 MoE class.

This intentionally avoids MiniCPMO and AutoModelForCausalLM. It constructs
Qwen3_5MoeForCausalLM directly, loads the LLM slice from the o5 checkpoint,
then runs normal Transformers generate().
"""

import json
import os
import time
from pathlib import Path

import torch
from accelerate import init_empty_weights
from transformers import AutoTokenizer, Qwen3_5MoeForCausalLM
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig


MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/user/weihongliang/MiniCPM-o-4_6"))
PT_PATH = Path(os.environ.get("PT_PATH", "/user/weihongliang/o5_weights/omni_sft2_main_run_iter1200.pt"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "/user/weihongliang/o5_qwen35_official_class_probe"))
PROMPT = os.environ.get("PROMPT", "请用中文简要介绍一下西安。")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "64"))
ATTN_IMPLEMENTATION = os.environ.get("ATTN_IMPLEMENTATION", "sdpa")
EXPERTS_IMPLEMENTATION = os.environ.get("EXPERTS_IMPLEMENTATION", "auto")
DTYPE = torch.bfloat16


def unwrap_state_dict(obj):
    for key in ("state_dict", "model", "module"):
        if isinstance(obj, dict) and isinstance(obj.get(key), dict):
            return obj[key]
    return obj


def load_llm_state_dict():
    state = unwrap_state_dict(torch.load(PT_PATH, map_location="cpu", weights_only=True, mmap=True))
    return {key[len("llm."):]: value for key, value in state.items() if key.startswith("llm.")}


@torch.inference_mode()
def main():
    torch.cuda.set_device(0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    config = Qwen3_5MoeTextConfig.from_pretrained(MODEL_PATH)
    config._attn_implementation = ATTN_IMPLEMENTATION
    if EXPERTS_IMPLEMENTATION != "auto":
        config._experts_implementation = EXPERTS_IMPLEMENTATION
    if isinstance(getattr(config, "torch_dtype", None), str):
        config.torch_dtype = getattr(torch, config.torch_dtype, None)

    with init_empty_weights():
        model = Qwen3_5MoeForCausalLM(config)

    t0 = time.perf_counter()
    llm_state = load_llm_state_dict()
    info = model.load_state_dict(llm_state, strict=False, assign=True)
    load_state_s = time.perf_counter() - t0
    del llm_state

    t1 = time.perf_counter()
    model.to(device="cuda", dtype=DTYPE).eval()
    to_cuda_s = time.perf_counter() - t1

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    inputs = tokenizer(PROMPT, return_tensors="pt", add_special_tokens=True).to("cuda")

    torch.cuda.synchronize()
    t2 = time.perf_counter()
    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        use_cache=True,
    )
    torch.cuda.synchronize()
    generate_s = time.perf_counter() - t2

    new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    record = {
        "model_class": model.__class__.__name__,
        "config_class": config.__class__.__name__,
        "model_type": config.model_type,
        "attn_implementation": ATTN_IMPLEMENTATION,
        "experts_implementation": getattr(config, "_experts_implementation", None),
        "has_functional_grouped_mm": hasattr(torch.nn.functional, "grouped_mm"),
        "prompt": PROMPT,
        "prompt_tokens": int(inputs["input_ids"].shape[1]),
        "max_new_tokens": MAX_NEW_TOKENS,
        "generated_tokens": int(new_ids.numel()),
        "load_state_s": load_state_s,
        "to_cuda_s": to_cuda_s,
        "generate_s": generate_s,
        "tokens_per_s": float(new_ids.numel() / generate_s) if generate_s else None,
        "cuda_max_mem_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "missing_keys": len(info.missing_keys),
        "unexpected_keys": len(info.unexpected_keys),
        "missing_head": info.missing_keys[:20],
        "unexpected_head": info.unexpected_keys[:20],
        "text": text,
    }
    (OUT_DIR / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
