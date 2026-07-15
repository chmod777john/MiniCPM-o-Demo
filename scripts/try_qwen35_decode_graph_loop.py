#!/usr/bin/env python3
"""Try a replayable CUDA-graph decode loop for pure Qwen3.5 MoE."""

import json
import os
import time
from pathlib import Path

import torch
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
        "cuda_profiler_range": CUDA_PROFILER_RANGE,
        "nvtx_ranges": NVTX_RANGES,
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
            for step_idx in range(STEPS):
                with nvtx_range(f"graph_replay_{step_idx:03d}"):
                    _unused, dt = sync_time(graph.replay)
                    replay_times.append(dt * 1000)
                    out = holder["out"]
                    next_token = out.logits[:, -1:, :].argmax(dim=-1)
                    generated.append(int(graph_token.item()))
                    graph_token.copy_(next_token)
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
