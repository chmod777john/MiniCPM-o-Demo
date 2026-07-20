#!/usr/bin/env python3
"""Try CUDA Graph replay for Qwen3.5-MoE tensor-parallel decode."""

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
OUT_DIR = Path(os.environ.get("OUT_DIR", "/user/weihongliang/o5_qwen35_tp2_graph_profile"))
STEPS = int(os.environ.get("STEPS", "64"))
EAGER_WARM_STEPS = int(os.environ.get("EAGER_WARM_STEPS", "16"))
GRAPH_WARM_REPLAYS = int(os.environ.get("GRAPH_WARM_REPLAYS", "8"))
PROMPT_REPEAT = int(os.environ.get("PROMPT_REPEAT", "8"))
EXPERTS_IMPLEMENTATION = os.environ.get("EXPERTS_IMPLEMENTATION", "batched_mm")
TP_PLAN = os.environ.get("TP_PLAN", "auto")
SYNC_EACH_STEP = os.environ.get("SYNC_EACH_STEP", "0").lower() in {"1", "true", "yes", "on"}
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


def barrier_time(fn):
    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    dist.barrier()
    return out, time.perf_counter() - start


def summarize_ms(ms):
    return {
        "count": len(ms),
        "mean_ms": statistics.mean(ms),
        "median_ms": statistics.median(ms),
        "min_ms": min(ms),
        "max_ms": max(ms),
    }


def write_result(record):
    if is_rank0():
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "result.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False), flush=True)


@torch.inference_mode()
def main() -> None:
    torch.cuda.set_device(local_rank())
    dist.init_process_group(backend="nccl")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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
        "steps": STEPS,
        "eager_warm_steps": EAGER_WARM_STEPS,
        "graph_warm_replays": GRAPH_WARM_REPLAYS,
        "sync_each_step": SYNC_EACH_STEP,
    }

    try:
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
        record["input_tokens"] = int(input_ids.shape[-1])

        prefill_out, prefill_s = barrier_time(
            lambda: model(input_ids=input_ids, use_cache=True, return_dict=True, logits_to_keep=1)
        )
        record["prefill_ms"] = prefill_s * 1000
        past = prefill_out.past_key_values
        token = prefill_out.logits[:, -1:, :].argmax(dim=-1)

        eager_warm_ms = []
        for _ in range(EAGER_WARM_STEPS):
            out, dt = barrier_time(
                lambda: model(input_ids=token, past_key_values=past, use_cache=True, return_dict=True, logits_to_keep=1)
            )
            eager_warm_ms.append(dt * 1000)
            past = out.past_key_values
            token = out.logits[:, -1:, :].argmax(dim=-1)
        record["eager_warm_summary"] = summarize_ms(eager_warm_ms)

        graph_token = token.clone()
        holder = {}

        def step():
            holder["out"] = model(
                input_ids=graph_token,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )

        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        dist.barrier()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        record["graph_capture"] = "ok"

        for _ in range(GRAPH_WARM_REPLAYS):
            graph.replay()
        torch.cuda.synchronize()
        dist.barrier()

        replay_ms = []
        generated = []
        if SYNC_EACH_STEP:
            for _ in range(STEPS):
                _unused, dt = barrier_time(graph.replay)
                replay_ms.append(dt * 1000)
                out = holder["out"]
                next_token = out.logits[:, -1:, :].argmax(dim=-1)
                if is_rank0():
                    generated.append(int(graph_token.item()))
                graph_token.copy_(next_token)
        else:
            torch.cuda.synchronize()
            dist.barrier()
            start = time.perf_counter()
            for _ in range(STEPS):
                graph.replay()
                out = holder["out"]
                next_token = out.logits[:, -1:, :].argmax(dim=-1)
                if is_rank0():
                    generated.append(int(graph_token.item()))
                graph_token.copy_(next_token)
            torch.cuda.synchronize()
            dist.barrier()
            per_step = (time.perf_counter() - start) * 1000 / STEPS
            replay_ms = [per_step] * STEPS

        record["graph_replay_summary"] = summarize_ms(replay_ms)
        record["graph_tokens_per_s"] = 1000 / record["graph_replay_summary"]["mean_ms"]
        record["generated_token_ids"] = generated
        if is_rank0():
            record["generated_text"] = tokenizer.decode(generated, skip_special_tokens=True)

    except Exception as exc:
        record["graph_capture"] = "failed"
        record["error_type"] = type(exc).__name__
        record["error"] = str(exc)
    finally:
        write_result(record)
        # Do not add a final barrier here: if one rank exits the graph path earlier
        # or the NCCL graph path leaves a rank in a different state, a final barrier
        # can keep the cctl job alive after result.json has already been written.
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
