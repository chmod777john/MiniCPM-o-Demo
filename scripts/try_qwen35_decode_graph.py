#!/usr/bin/env python3
"""Try CUDA graph capture for one-step pure Qwen3.5 MoE decode.

This is an experimental probe. It intentionally targets the pure HF backbone and
does not modify serving/demo code.
"""

import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3_5MoeForCausalLM


MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/user/weihongliang/wangkaiqi/o5_backbone_hf"))
TOKENIZER_PATH = Path(os.environ.get("TOKENIZER_PATH", "/user/weihongliang/MiniCPM-o-4_6"))
OUT_DIR = Path(os.environ.get("OUT_DIR", "/user/weihongliang/o5_decode_graph_probe"))
PROMPT = os.environ.get("PROMPT", "请用中文简要介绍西安的历史、地理、文化和旅游特色。")
PROMPT_REPEAT = int(os.environ.get("PROMPT_REPEAT", "4"))
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
    prompt = "\n".join([PROMPT] * PROMPT_REPEAT)
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].cuda()

    outputs = model(input_ids=input_ids, use_cache=True, return_dict=True, logits_to_keep=1)
    past = outputs.past_key_values
    next_token = outputs.logits[:, -1:, :].argmax(dim=-1)

    # Warm several eager decode steps so recurrent states are initialized.
    eager_times = []
    for _ in range(8):
        out, dt = sync_time(
            lambda: model(input_ids=next_token, past_key_values=past, use_cache=True, return_dict=True, logits_to_keep=1)
        )
        eager_times.append(dt)
        past = out.past_key_values
        next_token = out.logits[:, -1:, :].argmax(dim=-1)

    graph_token = next_token.clone()
    graph_out_holder = {}

    def graph_step():
        graph_out_holder["out"] = model(
            input_ids=graph_token,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )

    record = {
        "model_path": str(MODEL_PATH),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "transformers": __import__("transformers").__version__,
        "experts_implementation": EXPERTS_IMPLEMENTATION,
        "eager_warm_decode_ms": [v * 1000 for v in eager_times],
    }

    try:
        # Capture on a side stream per CUDA graph best practice.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                graph_step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_step()

        replay_times = []
        for _ in range(32):
            _out, dt = sync_time(graph.replay)
            replay_times.append(dt)
        record["graph_capture"] = "ok"
        record["graph_replay_ms"] = [v * 1000 for v in replay_times]
        record["graph_replay_mean_ms"] = sum(record["graph_replay_ms"]) / len(record["graph_replay_ms"])
    except Exception as exc:
        record["graph_capture"] = "failed"
        record["error_type"] = type(exc).__name__
        record["error"] = str(exc)

    (OUT_DIR / "result.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
