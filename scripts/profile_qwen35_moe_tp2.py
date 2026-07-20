#!/usr/bin/env python3
"""Profile pure Qwen3.5-MoE with Transformers tensor parallelism.

Launch with:

    torchrun --standalone --nproc_per_node=2 scripts/profile_qwen35_moe_tp2.py

The script records rank-0 wall time for prefill and one-token decode steps.
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, Qwen3_5MoeForCausalLM


MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/user/weihongliang/wangkaiqi/o5_backbone_hf"))
TOKENIZER_PATH = Path(os.environ.get("TOKENIZER_PATH", "/user/weihongliang/MiniCPM-o-4_6"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "/user/weihongliang/o5_qwen35_tp2_profile"))
DECODE_STEPS = int(os.environ.get("DECODE_STEPS", "32"))
WARMUP_DECODE_STEPS = int(os.environ.get("WARMUP_DECODE_STEPS", "8"))
WARMUP_PREFILL_STEPS = int(os.environ.get("WARMUP_PREFILL_STEPS", "1"))
PROMPT_REPEAT = int(os.environ.get("PROMPT_REPEAT", "8"))
EXPERTS_IMPLEMENTATION = os.environ.get("EXPERTS_IMPLEMENTATION", "batched_mm")
TP_PLAN = os.environ.get("TP_PLAN", "auto")
DTYPE = torch.bfloat16
PROMPT = os.environ.get(
    "PROMPT",
    "请用中文简要介绍你自己，并说明你可以如何帮助用户。",
)


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_rank0() -> bool:
    return rank() == 0


def sync_time(fn):
    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    dist.barrier()
    return out, time.perf_counter() - start


def summarize(seconds):
    ms = [x * 1000 for x in seconds]
    return {
        "count": len(ms),
        "mean_ms": statistics.mean(ms),
        "median_ms": statistics.median(ms),
        "min_ms": min(ms),
        "max_ms": max(ms),
    }


@torch.inference_mode()
def main() -> None:
    torch.cuda.set_device(local_rank())
    dist.init_process_group(backend="nccl")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = Qwen3_5MoeForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=DTYPE,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        tp_plan=TP_PLAN,
    ).eval()
    model.config._experts_implementation = EXPERTS_IMPLEMENTATION

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    input_ids = tokenizer("\n".join([PROMPT] * PROMPT_REPEAT), return_tensors="pt", add_special_tokens=True)[
        "input_ids"
    ].to(torch.cuda.current_device())

    warm_prefill_times = []
    for _ in range(WARMUP_PREFILL_STEPS):
        _warm_out, dt = sync_time(
            lambda: model(input_ids=input_ids, use_cache=True, return_dict=True, logits_to_keep=1)
        )
        warm_prefill_times.append(dt)

    prefill_out, prefill_s = sync_time(
        lambda: model(input_ids=input_ids, use_cache=True, return_dict=True, logits_to_keep=1)
    )
    past = prefill_out.past_key_values
    next_token = prefill_out.logits[:, -1:, :].argmax(dim=-1)

    warm_times = []
    for _ in range(WARMUP_DECODE_STEPS):
        out, dt = sync_time(
            lambda: model(input_ids=next_token, past_key_values=past, use_cache=True, return_dict=True, logits_to_keep=1)
        )
        warm_times.append(dt)
        past = out.past_key_values
        next_token = out.logits[:, -1:, :].argmax(dim=-1)

    decode_times = []
    generated = []
    for _ in range(DECODE_STEPS):
        out, dt = sync_time(
            lambda: model(input_ids=next_token, past_key_values=past, use_cache=True, return_dict=True, logits_to_keep=1)
        )
        decode_times.append(dt)
        if is_rank0():
            generated.append(int(next_token.item()))
        past = out.past_key_values
        next_token = out.logits[:, -1:, :].argmax(dim=-1)

    if is_rank0():
        record = {
            "model_path": str(MODEL_PATH),
            "tokenizer_path": str(TOKENIZER_PATH),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": __import__("transformers").__version__,
            "device": torch.cuda.get_device_name(0),
            "world_size": world_size(),
            "tp_plan": TP_PLAN,
            "experts_implementation": EXPERTS_IMPLEMENTATION,
            "prompt_repeat": PROMPT_REPEAT,
            "input_tokens": int(input_ids.shape[-1]),
            "warm_prefill_summary": summarize(warm_prefill_times) if warm_prefill_times else None,
            "prefill_ms": prefill_s * 1000,
            "warm_decode_summary": summarize(warm_times),
            "decode_summary": summarize(decode_times),
            "decode_tokens_per_s": len(decode_times) / sum(decode_times),
            "generated_token_ids": generated,
            "generated_text": tokenizer.decode(generated, skip_special_tokens=True),
        }
        (OUT_DIR / "result.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False), flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
