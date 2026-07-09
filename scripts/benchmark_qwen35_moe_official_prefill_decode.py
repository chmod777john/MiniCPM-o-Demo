#!/usr/bin/env python3
"""Bare Qwen3.5 MoE prefill/decode benchmark for o5 LLM weights.

This intentionally excludes MiniCPMO, audio, TTS, token2wav, and duplex logic.
It constructs the public Transformers Qwen3_5MoeForCausalLM class directly,
loads only `llm.*` weights from the o5 checkpoint, then measures:

  * prompt prefill latency
  * one-token decode latency over a manual KV-cache loop

Use EXPERTS_IMPLEMENTATION=eager or grouped_mm to compare MoE expert kernels.
"""

import gc
import json
import os
import statistics
import time
from pathlib import Path

import torch
from accelerate import init_empty_weights
from torch import nn
from transformers import AutoTokenizer, Qwen3_5MoeForCausalLM
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig


MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/user/weihongliang/MiniCPM-o-4_6"))
PT_PATH = Path(os.environ.get("PT_PATH", "/user/weihongliang/o5_weights/omni_sft2_main_run_iter1200.pt"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "/user/weihongliang/o5_qwen35_moe_prefill_decode_bench"))
PROMPT = os.environ.get("PROMPT", "请用中文简要介绍西安的历史、地理、文化和旅游特色。")
PROMPT_REPEAT = int(os.environ.get("PROMPT_REPEAT", "1"))
DECODE_STEPS = int(os.environ.get("DECODE_STEPS", "64"))
WARMUP_DECODE_STEPS = int(os.environ.get("WARMUP_DECODE_STEPS", "4"))
ATTN_IMPLEMENTATION = os.environ.get("ATTN_IMPLEMENTATION", "sdpa")
EXPERTS_IMPLEMENTATION = os.environ.get("EXPERTS_IMPLEMENTATION", "auto")
DTYPE = torch.bfloat16


def unwrap_state_dict(obj):
    for key in ("state_dict", "model", "module"):
        if isinstance(obj, dict) and isinstance(obj.get(key), dict):
            return obj[key]
    return obj


def set_module_param(root: nn.Module, name: str, value: torch.Tensor) -> None:
    parts = name.split(".")
    module = root
    for part in parts[:-1]:
        module = getattr(module, part)
    setattr(module, parts[-1], nn.Parameter(value, requires_grad=False))


def load_llm_state_dict():
    state = unwrap_state_dict(torch.load(PT_PATH, map_location="cpu", weights_only=True, mmap=True))
    return {key[len("llm."):]: value for key, value in state.items() if key.startswith("llm.")}


def load_model():
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
    gc.collect()

    t1 = time.perf_counter()
    model.to(device="cuda", dtype=DTYPE).eval()
    to_cuda_s = time.perf_counter() - t1
    return model, config, info, load_state_s, to_cuda_s


def sync_time(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - start


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[idx]


@torch.inference_mode()
def run_prefill_decode(model, input_ids):
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device="cuda")

    def prefill_call():
        return model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True, return_dict=True)

    outputs, prefill_s = sync_time(prefill_call)
    past = outputs.past_key_values
    next_token = outputs.logits[:, -1:, :].argmax(dim=-1)

    warmup_times = []
    for _ in range(WARMUP_DECODE_STEPS):
        cache_len = past.get_seq_length() if hasattr(past, "get_seq_length") else attention_mask.shape[1]
        decode_mask = torch.ones((1, cache_len + 1), dtype=torch.bool, device="cuda")

        def warmup_call():
            return model(
                input_ids=next_token,
                attention_mask=decode_mask,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )

        out, dt = sync_time(warmup_call)
        warmup_times.append(dt)
        past = out.past_key_values
        next_token = out.logits[:, -1:, :].argmax(dim=-1)

    decode_times = []
    generated = []
    for _ in range(DECODE_STEPS):
        cache_len = past.get_seq_length() if hasattr(past, "get_seq_length") else attention_mask.shape[1]
        decode_mask = torch.ones((1, cache_len + 1), dtype=torch.bool, device="cuda")

        def decode_call():
            return model(
                input_ids=next_token,
                attention_mask=decode_mask,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )

        out, dt = sync_time(decode_call)
        decode_times.append(dt)
        generated.append(int(next_token.item()))
        past = out.past_key_values
        next_token = out.logits[:, -1:, :].argmax(dim=-1)

    return {
        "prefill_s": prefill_s,
        "warmup_decode_times_s": warmup_times,
        "decode_times_s": decode_times,
        "decode_mean_s": statistics.mean(decode_times) if decode_times else None,
        "decode_median_s": statistics.median(decode_times) if decode_times else None,
        "decode_p90_s": percentile(decode_times, 90),
        "decode_p99_s": percentile(decode_times, 99),
        "decode_total_s": sum(decode_times),
        "decode_tokens_per_s": len(decode_times) / sum(decode_times) if decode_times else None,
        "generated_token_ids": generated,
    }


def main():
    torch.cuda.set_device(0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()

    model, config, info, load_state_s, to_cuda_s = load_model()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    prompt = "\n".join([PROMPT] * PROMPT_REPEAT)
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].to("cuda")
    result = run_prefill_decode(model, input_ids)
    text = tokenizer.decode(result["generated_token_ids"], skip_special_tokens=True)

    record = {
        "model_class": model.__class__.__name__,
        "config_class": config.__class__.__name__,
        "model_type": config.model_type,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
        "attn_implementation": ATTN_IMPLEMENTATION,
        "experts_implementation": getattr(config, "_experts_implementation", None),
        "has_functional_grouped_mm": hasattr(torch.nn.functional, "grouped_mm"),
        "has_private_grouped_mm": hasattr(torch, "_grouped_mm"),
        "prompt": PROMPT,
        "prompt_repeat": PROMPT_REPEAT,
        "prompt_tokens": int(input_ids.shape[1]),
        "decode_steps": DECODE_STEPS,
        "warmup_decode_steps": WARMUP_DECODE_STEPS,
        "load_state_s": load_state_s,
        "to_cuda_s": to_cuda_s,
        "cuda_max_mem_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "missing_keys": len(info.missing_keys),
        "unexpected_keys": len(info.unexpected_keys),
        "missing_head": info.missing_keys[:20],
        "unexpected_head": info.unexpected_keys[:20],
        "generated_text": text,
        **result,
    }
    (OUT_DIR / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
