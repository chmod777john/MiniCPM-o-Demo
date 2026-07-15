#!/usr/bin/env python3
"""Try a replayable CUDA-graph decode loop for pure Qwen3.5 MoE."""

import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen3_5MoeForCausalLM


MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/user/weihongliang/wangkaiqi/o5_backbone_hf"))
TOKENIZER_PATH = Path(os.environ.get("TOKENIZER_PATH", "/user/weihongliang/MiniCPM-o-4_6"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "/user/weihongliang/o5_decode_graph_loop_probe"))
PROMPT = os.environ.get("PROMPT", "请用中文简要介绍西安的历史、地理、文化和旅游特色。")
PROMPT_REPEAT = int(os.environ.get("PROMPT_REPEAT", "4"))
STEPS = int(os.environ.get("STEPS", "32"))
EXPERTS_IMPLEMENTATION = os.environ.get("EXPERTS_IMPLEMENTATION", "batched_mm")
CUDA_PROFILER_RANGE = os.environ.get("CUDA_PROFILER_RANGE", "0").lower() in {"1", "true", "yes", "on"}
NVTX_RANGES = os.environ.get("NVTX_RANGES", "1").lower() in {"1", "true", "yes", "on"}
SYNC_EACH_STEP = os.environ.get("SYNC_EACH_STEP", "1").lower() in {"1", "true", "yes", "on"}
PATCH_B1_EXPERTS = os.environ.get("PATCH_B1_EXPERTS", "0").lower() in {"1", "true", "yes", "on"}
DTYPE = torch.bfloat16


def sync_time(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - start


class nvtx_range:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        if NVTX_RANGES and torch.cuda.is_available():
            torch.cuda.nvtx.range_push(self.name)

    def __exit__(self, exc_type, exc, tb):
        if NVTX_RANGES and torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()


def batch1_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    if hidden_states.shape[0] != 1 or self.has_bias or not self.has_gate or self.is_transposed:
        return self.__base_forward(hidden_states, top_k_index, top_k_weights)
    expert_ids = top_k_index.reshape(-1).clamp(0, self.num_experts - 1)
    weights = top_k_weights.reshape(-1).to(hidden_states.dtype)
    hidden = hidden_states.expand(expert_ids.shape[0], -1)
    proj = torch.bmm(self.gate_up_proj[expert_ids], hidden.unsqueeze(-1)).squeeze(-1)
    gate, up = proj.chunk(2, dim=-1)
    proj = F.silu(gate) * up
    proj = torch.bmm(self.down_proj[expert_ids], proj.unsqueeze(-1)).squeeze(-1)
    return (proj * weights.unsqueeze(-1)).sum(dim=0, keepdim=True).to(hidden_states.dtype)


def patch_batch1_experts(model):
    for layer in model.model.layers:
        experts = getattr(getattr(layer, "mlp", None), "experts", None)
        if experts is None or hasattr(experts, "__base_forward"):
            continue
        experts.__base_forward = experts.forward
        experts.forward = batch1_experts_forward.__get__(experts, experts.__class__)


@torch.inference_mode()
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    model = Qwen3_5MoeForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=DTYPE,
        attn_implementation="sdpa",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).eval().cuda()
    model.config._experts_implementation = EXPERTS_IMPLEMENTATION
    if PATCH_B1_EXPERTS:
        patch_batch1_experts(model)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    input_ids = tokenizer("\n".join([PROMPT] * PROMPT_REPEAT), return_tensors="pt", add_special_tokens=True)[
        "input_ids"
    ].cuda()

    outputs = model(input_ids=input_ids, use_cache=True, return_dict=True, logits_to_keep=1)
    past = outputs.past_key_values
    token = outputs.logits[:, -1:, :].argmax(dim=-1)

    # Initialize recurrent states and avoid first-token one-off costs.
    eager_generated = []
    eager_times = []
    for _ in range(8):
        out, dt = sync_time(lambda: model(input_ids=token, past_key_values=past, use_cache=True, return_dict=True, logits_to_keep=1))
        eager_times.append(dt * 1000)
        eager_generated.append(int(token.item()))
        past = out.past_key_values
        token = out.logits[:, -1:, :].argmax(dim=-1)

    graph_token = token.clone()
    holder = {}

    def step():
        holder["out"] = model(input_ids=graph_token, past_key_values=past, use_cache=True, return_dict=True, logits_to_keep=1)

    record = {
        "model_path": str(MODEL_PATH),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
        "experts_implementation": EXPERTS_IMPLEMENTATION,
        "patch_b1_experts": PATCH_B1_EXPERTS,
        "cuda_profiler_range": CUDA_PROFILER_RANGE,
        "nvtx_ranges": NVTX_RANGES,
        "sync_each_step": SYNC_EACH_STEP,
        "eager_warm_decode_ms": eager_times,
        "eager_warm_generated": eager_generated,
    }

    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()

        replay_times = []
        generated = []
        if CUDA_PROFILER_RANGE:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
        with nvtx_range("graph_replay_loop"):
            if SYNC_EACH_STEP:
                for step_idx in range(STEPS):
                    with nvtx_range(f"graph_replay_{step_idx:03d}"):
                        _unused, dt = sync_time(graph.replay)
                        replay_times.append(dt * 1000)
                        out = holder["out"]
                        next_token = out.logits[:, -1:, :].argmax(dim=-1)
                        generated.append(int(graph_token.item()))
                        graph_token.copy_(next_token)
            else:
                torch.cuda.synchronize()
                loop_start = time.perf_counter()
                for step_idx in range(STEPS):
                    with nvtx_range(f"graph_replay_{step_idx:03d}"):
                        graph.replay()
                        out = holder["out"]
                        next_token = out.logits[:, -1:, :].argmax(dim=-1)
                        generated.append(int(graph_token.item()))
                        graph_token.copy_(next_token)
                torch.cuda.synchronize()
                loop_dt = (time.perf_counter() - loop_start) * 1000
                replay_times = [loop_dt / STEPS] * STEPS
        if CUDA_PROFILER_RANGE:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()

        record["graph_loop"] = "ok"
        record["graph_loop_ms"] = replay_times
        record["graph_loop_mean_ms"] = sum(replay_times) / len(replay_times)
        record["graph_generated"] = generated
        record["graph_text"] = tokenizer.decode(generated, skip_special_tokens=True)
    except Exception as exc:
        record["graph_loop"] = "failed"
        record["error_type"] = type(exc).__name__
        record["error"] = str(exc)

    (OUT_DIR / "result.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
