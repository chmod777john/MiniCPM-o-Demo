#!/usr/bin/env python3
"""Microbenchmark one Qwen3.5 MoE experts block for batch=1 decode."""

import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import Qwen3_5MoeForCausalLM


MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/user/weihongliang/wangkaiqi/o5_backbone_hf"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "/user/weihongliang/o5_moe_experts_microbench"))
LAYER_IDX = int(os.environ.get("LAYER_IDX", "0"))
STEPS = int(os.environ.get("STEPS", "512"))
WARMUP = int(os.environ.get("WARMUP", "64"))
EXPERT_PATTERN = os.environ.get("EXPERT_PATTERN", "fixed")
ENABLE_COMPILE = os.environ.get("ENABLE_COMPILE", "0").lower() in {"1", "true", "yes", "on"}
DTYPE = torch.bfloat16


def summary(values):
    values = [v * 1000 for v in values]
    return {
        "count": len(values),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def sync_time(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - start


def load_experts():
    model = Qwen3_5MoeForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=DTYPE,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).eval().cuda()
    layer = model.model.layers[LAYER_IDX]
    experts = layer.mlp.experts
    return model, experts


def make_inputs(experts):
    hidden = torch.randn(1, experts.hidden_dim, dtype=DTYPE, device="cuda")
    if EXPERT_PATTERN == "random":
        topk = torch.randperm(experts.num_experts, device="cuda")[:8].view(1, -1)
    else:
        topk = torch.arange(8, device="cuda").view(1, -1)
    weights = torch.ones_like(topk, dtype=DTYPE) / topk.shape[-1]
    return hidden, topk, weights


def hf_batched(experts, hidden, topk, weights):
    return experts(hidden, topk, weights)


def simple_bmm(experts, hidden, topk, weights):
    expert_ids = topk.reshape(-1).clamp(0, experts.num_experts - 1)
    sample_weights = weights.reshape(-1).to(hidden.dtype)
    expanded = hidden.expand(expert_ids.numel(), -1)
    gate_up = experts.gate_up_proj[expert_ids]
    proj = torch.bmm(gate_up, expanded.unsqueeze(-1)).squeeze(-1)
    gate, up = proj.chunk(2, dim=-1)
    proj = F.silu(gate) * up
    down = experts.down_proj[expert_ids]
    proj = torch.bmm(down, proj.unsqueeze(-1)).squeeze(-1)
    return (proj * sample_weights.unsqueeze(-1)).sum(dim=0, keepdim=True).to(hidden.dtype)


def loop_linear(experts, hidden, topk, weights):
    out = torch.zeros_like(hidden)
    for i in range(topk.shape[-1]):
        idx = topk[0, i]
        gate, up = F.linear(hidden, experts.gate_up_proj[idx]).chunk(2, dim=-1)
        cur = F.silu(gate) * up
        cur = F.linear(cur, experts.down_proj[idx])
        out = out + cur * weights[0, i]
    return out


def bench_one(name, fn, experts, hidden, topk, weights):
    if ENABLE_COMPILE:
        fn = torch.compile(fn, dynamic=False, mode="reduce-overhead")
    for _ in range(WARMUP):
        fn(experts, hidden, topk, weights)
    torch.cuda.synchronize()
    times = []
    for _ in range(STEPS):
        _out, dt = sync_time(lambda: fn(experts, hidden, topk, weights))
        times.append(dt)
    return {"name": name, **summary(times)}


@torch.inference_mode()
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    model, experts = load_experts()
    model.config._experts_implementation = "batched_mm"
    hidden, topk, weights = make_inputs(experts)
    results = []
    for name, fn in [
        ("hf_batched", hf_batched),
        ("simple_bmm", simple_bmm),
        ("loop_linear", loop_linear),
    ]:
        results.append(bench_one(name, fn, experts, hidden, topk, weights))
    record = {
        "model_path": str(MODEL_PATH),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
        "layer_idx": LAYER_IDX,
        "steps": STEPS,
        "warmup": WARMUP,
        "expert_pattern": EXPERT_PATTERN,
        "compile": ENABLE_COMPILE,
        "results": results,
    }
    (OUT_DIR / "result.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
