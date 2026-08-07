#!/usr/bin/env python3
"""Extract the LLM backbone (self.llm = Qwen3_5MoeForCausalLM) from the full MiniCPMO .pt
checkpoint into a HF-format directory (config.json + sharded safetensors + index), so it can be
loaded with from_pretrained(tp_plan='auto') for 2-card tensor parallelism.

Approach (robust, no manual config reconstruction): build MiniCPMO on meta exactly like
load_o5_model, load the full state_dict (assign=True), then model.llm.save_pretrained(OUT).
model.llm already carries the correct backbone config; save_pretrained writes model.* + lm_head.
"""
import os, sys, json, time
import torch
WORKTREE = os.environ["WORKTREE"]; sys.path.insert(0, WORKTREE); sys.path.insert(0, os.path.join(WORKTREE, "scripts"))
MODEL_PATH = os.environ["MODEL_PATH"]
PT_PATH = os.environ["PT_PATH"]
OUT = os.environ.get("BACKBONE_DIR", "/user/weihongliang/wangkaiqi/o5_backbone_hf")
# 0-GPU / CPU-only Job 没有 CUDA；若 config 默认 flash_attention_2，
# transformers 会在 CPU 上直接 ImportError。抽取权重不需要真实 attention kernel。
os.environ.setdefault("ATTN_IMPLEMENTATION", "sdpa")

from transformers import AutoConfig
from accelerate import init_empty_weights

def log(*a): print(*a, flush=True)

def _force_attn_implementation(config: object, impl: str) -> None:
    """强制顶层与嵌套 config 使用非 flash attention，兼容 CPU-only 抽取。"""
    if config is None:
        return
    for attr in ("_attn_implementation", "_attn_implementation_internal"):
        if hasattr(config, attr):
            setattr(config, attr, impl)
    if hasattr(config, "attn_implementation"):
        try:
            setattr(config, "attn_implementation", impl)
        except Exception:
            pass
    for nest_name in ("llm_config", "text_config", "vision_config", "audio_config"):
        _force_attn_implementation(getattr(config, nest_name, None), impl)

def main():
    t0 = time.time()
    attn_impl = os.environ["ATTN_IMPLEMENTATION"]
    cfg = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
    _force_attn_implementation(cfg, attn_impl)
    cfg._name_or_path = MODEL_PATH; cfg.name_or_path = MODEL_PATH
    import minimal_o5_unified_model_duplex as probe
    MiniCPMO = probe.MiniCPMO
    log(f"building MiniCPMO on meta (attn={attn_impl}) ...")
    with init_empty_weights():
        model = MiniCPMO(cfg)
    log(f"loading full state_dict from {PT_PATH} ...")
    sd = probe.load_pt_state_dict(PT_PATH)
    log(f"state_dict keys={len(sd)}  (load {time.time()-t0:.0f}s)")
    n_llm = sum(1 for k in sd if k.startswith("llm."))
    log(f"llm.* keys={n_llm}")
    ckpt_vocab = sd.get("llm.model.embed_tokens.weight")
    if ckpt_vocab is not None:
        ckpt_vocab_size = int(ckpt_vocab.shape[0])
        model_vocab_size = int(model.llm.get_input_embeddings().weight.shape[0])
        if ckpt_vocab_size != model_vocab_size:
            log(f"resizing llm token embeddings {model_vocab_size} -> {ckpt_vocab_size}")
            model.llm.resize_token_embeddings(ckpt_vocab_size, mean_resizing=False)
    info = model.load_state_dict(sd, strict=False, assign=True)
    log(f"load_state_dict missing={len(info.missing_keys)} unexpected={len(info.unexpected_keys)}")
    del sd
    llm = model.llm
    llm = llm.bfloat16()
    # sanity: report backbone config dims + param count
    c = llm.config
    log(f"backbone config: type={c.model_type} hidden={c.hidden_size} layers={c.num_hidden_layers} "
        f"experts={getattr(c,'num_experts','?')} topk={getattr(c,'num_experts_per_tok','?')} "
        f"vocab={c.vocab_size}")
    nparam = sum(p.numel() for p in llm.parameters())
    log(f"backbone params={nparam/1e9:.2f}B  bytes(bf16)={nparam*2/1e9:.1f}GB")
    os.makedirs(OUT, exist_ok=True)
    log(f"saving backbone (sharded safetensors) -> {OUT} ...")
    llm.save_pretrained(OUT, safe_serialization=True, max_shard_size="5GB")
    # copy tokenizer for convenience (optional)
    log("saved. dir listing:")
    for f in sorted(os.listdir(OUT)):
        p = os.path.join(OUT, f); log(f"  {f}  {os.path.getsize(p) if os.path.isfile(p) else 'dir'}")
    log(f"DONE in {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
