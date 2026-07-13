#!/usr/bin/env python3
"""Minimal TP2 turn-based chat smoke.

Runs inside torchrun with both ranks executing the same chat prefill/generate
calls. This isolates TP2 chat model behavior from gateway/worker WebSocket
serving and from SpmdMirror scheduling.
"""
import json
import os
import sys

import numpy as np
import torch

WORKTREE = os.environ["WORKTREE"]
sys.path.insert(0, WORKTREE)


def main():
    import torch.distributed as dist
    import core.deploy as deploy
    from MiniCPMO45.modeling_minicpmo_unified import ProcessorMode

    cfg = {
        "model_path": os.environ["MODEL_PATH"],
        "pt_path": os.environ["PT_PATH"],
        "backbone_dir": os.environ["BACKBONE_DIR"],
        "chat_vocoder": "token2wav",
        "attn_implementation": os.environ.get("ATTN_IMPLEMENTATION", "sdpa"),
        "llm_cache_len": int(os.environ.get("O5_LLM_CACHE", "32768")),
    }
    torch.manual_seed(1234)
    np.random.seed(1234)
    br = deploy.get_mode("tp2").build(cfg)
    model = br.model
    rank = br.rank
    is_driver = br.is_driver

    model.set_mode(ProcessorMode.CHAT)
    msgs = [{"role": "user", "content": os.environ.get("PROMPT", "请简单介绍一下你自己") }]
    if is_driver:
        print("BUILD", json.dumps(br.engine, ensure_ascii=False), flush=True)
        print("PREFILL_START", flush=True)
    prompt = model.non_streaming_prefill(
        session_id="tp2_chat_smoke",
        msgs=msgs,
        omni_mode=False,
        use_tts_template=False,
        enable_thinking=False,
        max_inp_length=8192,
    )
    dist.barrier()
    if is_driver:
        print("PREFILL_OK", len(prompt), flush=True)
        print("GENERATE_START", flush=True)
    result = model.non_streaming_generate(
        session_id="tp2_chat_smoke",
        max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", "64")),
        do_sample=False,
        generate_audio=False,
        use_tts_template=False,
        enable_thinking=False,
    )
    dist.barrier()
    if is_driver:
        text = result[0] if isinstance(result, tuple) else result
        print("GENERATE_OK", repr(text)[:1000], flush=True)


if __name__ == "__main__":
    main()
