#!/usr/bin/env python3
"""Capture model decode + greedy token update in one CUDA graph."""

import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3_5MoeForCausalLM


MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/user/weihongliang/wangkaiqi/o5_backbone_hf"))
TOKENIZER_PATH = Path(os.environ.get("TOKENIZER_PATH", "/user/weihongliang/MiniCPM-o-4_6"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "/user/weihongliang/o5_decode_graph_fullstep_probe"))
PROMPT = os.environ.get("PROMPT", "请用中文简要介绍西安的历史、地理、文化和旅游特色。")
PROMPT_REPEAT = int(os.environ.get("PROMPT_REPEAT", "4"))
STEPS = int(os.environ.get("STEPS", "64"))
EXPERTS_IMPLEMENTATION = os.environ.get("EXPERTS_IMPLEMENTATION", "batched_mm")
DTYPE = torch.bfloat16


def sync_time(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - start


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

    eager_times = []
    for _ in range(8):
        out, dt = sync_time(lambda: model(input_ids=token, past_key_values=past, use_cache=True, return_dict=True, logits_to_keep=1))
        eager_times.append(dt * 1000)
        past = out.past_key_values
        token = out.logits[:, -1:, :].argmax(dim=-1)

    graph_token = token.clone()
    generated_buffer = torch.empty((STEPS,), dtype=torch.long, device="cuda")
    step_index = torch.zeros((), dtype=torch.long, device="cuda")
    holder = {}

    def full_step():
        holder["out"] = model(input_ids=graph_token, past_key_values=past, use_cache=True, return_dict=True, logits_to_keep=1)
        next_token = holder["out"].logits[:, -1:, :].argmax(dim=-1)
        generated_buffer[step_index].copy_(graph_token.reshape(()))
        graph_token.copy_(next_token)
        step_index.add_(1)

    record = {
        "model_path": str(MODEL_PATH),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
        "experts_implementation": EXPERTS_IMPLEMENTATION,
        "eager_warm_decode_ms": eager_times,
    }

    try:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                full_step()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        # Reset only the user-visible loop buffers. Recurrent/cache state has intentionally
        # advanced during warmup, matching the previous graph-loop probes.
        step_index.zero_()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            full_step()

        torch.cuda.synchronize()
        loop_start = time.perf_counter()
        for _ in range(STEPS):
            graph.replay()
        torch.cuda.synchronize()
        total_ms = (time.perf_counter() - loop_start) * 1000

        generated = generated_buffer.cpu().tolist()
        record["graph_fullstep"] = "ok"
        record["graph_fullstep_total_ms"] = total_ms
        record["graph_fullstep_mean_ms"] = total_ms / STEPS
        record["graph_generated"] = generated
        record["graph_text"] = tokenizer.decode(generated, skip_special_tokens=True)
        record["final_step_index"] = int(step_index.item())
    except Exception as exc:
        record["graph_fullstep"] = "failed"
        record["error_type"] = type(exc).__name__
        record["error"] = str(exc)

    (OUT_DIR / "result.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
