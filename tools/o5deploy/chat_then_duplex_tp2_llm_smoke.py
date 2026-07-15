#!/usr/bin/env python3
"""TP2 LLM-boundary smoke: run chat first, then one duplex unit.

This catches worker-cache/session reset bugs that do not appear in a single
chat-only process.
"""
import os
import sys

import numpy as np
import torch

WORKTREE = os.environ["WORKTREE"]
sys.path.insert(0, WORKTREE)
sys.path.insert(0, os.path.join(WORKTREE, "scripts"))


def main():
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
    br = deploy.get_mode("tp2_llm").build(cfg)
    model = br.model
    if getattr(model, "_spmd_worker_loop", None) is not None and not br.is_driver:
        model._spmd_worker_loop()
        sys.stdout.flush()
        os._exit(0)

    print("BUILD", br.engine, flush=True)
    model.set_mode(ProcessorMode.CHAT)
    prompt = model.non_streaming_prefill(
        session_id="chat_then_duplex_chat",
        msgs=[{"role": "user", "content": "请只回答：测试"}],
        omni_mode=False,
        use_tts_template=False,
        enable_thinking=False,
        max_inp_length=8192,
    )
    print("CHAT_PREFILL_OK", len(prompt), flush=True)
    result = model.non_streaming_generate(
        session_id="chat_then_duplex_chat",
        max_new_tokens=8,
        do_sample=False,
        generate_audio=False,
        use_tts_template=False,
        enable_thinking=False,
    )
    text = result[0] if isinstance(result, tuple) else result
    print("CHAT_GENERATE_OK", repr(text), flush=True)

    ref = __import__("minimal_o5_unified_model_duplex").load_16k(
        os.path.join(WORKTREE, "assets/ref_audio/ref_minicpm_signature.wav")
    )
    print("DUPLEX_PREPARE_START", flush=True)
    model.duplex_prepare(
        prefix_system_prompt="<|im_start|>system\nStreaming Omni Conversation.\n<|audio_start|>",
        suffix_system_prompt="<|audio_end|><|im_end|>",
        ref_audio=ref,
        prompt_wav_path=os.path.join(WORKTREE, "assets/ref_audio/ref_minicpm_signature.wav"),
    )
    print("DUPLEX_PREPARE_OK", flush=True)
    audio = np.zeros(16000, dtype=np.float32)
    print("DUPLEX_PREFILL_START", flush=True)
    model.duplex_prefill(audio_waveform=audio, frame_list=None)
    print("DUPLEX_PREFILL_OK", flush=True)
    result = model.duplex_generate(force_listen=True)
    print("DUPLEX_GENERATE_OK", result, flush=True)
    model.duplex_finalize()

    shutdown = getattr(model, "_spmd_shutdown", None)
    if shutdown is not None:
        shutdown()


if __name__ == "__main__":
    main()
