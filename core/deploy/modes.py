"""Shipped deployment-mode builders. Each build() returns a ready-to-serve MiniCPMO.

config keys used: model_path, pt_path, backbone_dir (tp2), chat_vocoder, attn_implementation,
llm_cache_len (tp2/opt graph StaticCache width), duplex_config (optional override).
"""
from __future__ import annotations
import os
import logging
from typing import Any, Dict

import torch

from .base import DeploymentMode, BuildResult
from .registry import register_mode

logger = logging.getLogger("deploy.modes")

_DEFAULT_DUP = {
    "generate_audio": True, "ls_mode": "explicit", "max_new_speak_tokens_per_chunk": 20,
    "temperature": 0.7, "top_k": 20, "top_p": 0.8, "force_listen_count": 3,
}


# ─────────────────────────── shared helpers ───────────────────────────
def _mk_config(cfg: Dict[str, Any]):
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(cfg["model_path"], trust_remote_code=True)
    c._attn_implementation = cfg.get("attn_implementation", "sdpa")
    c._name_or_path = cfg["model_path"]; c.name_or_path = cfg["model_path"]
    return c


def _load_full(cfg: Dict[str, Any], device: str):
    """Single-card: build MiniCPMO, load the full .pt, place on `device`, init_unified."""
    from accelerate import init_empty_weights
    from MiniCPMO45.modeling_minicpmo_unified import MiniCPMO
    from MiniCPMO45.processing_minicpmo import MiniCPMOProcessor
    with init_empty_weights():
        model = MiniCPMO(_mk_config(cfg))
    sd = torch.load(cfg["pt_path"], map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(sd, strict=False, assign=True); del sd
    model.bfloat16().eval().to(device)
    model.processor = MiniCPMOProcessor.from_pretrained(cfg["model_path"], trust_remote_code=True)
    return model


def _surgery_tp(cfg: Dict[str, Any], device: str):
    """2-card TP: MiniCPMO with non-llm weights (mmap) + TP-sharded backbone via from_pretrained."""
    from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM
    from MiniCPMO45.modeling_minicpmo_unified import MiniCPMO
    from MiniCPMO45.processing_minicpmo import MiniCPMOProcessor
    with init_empty_weights():
        model = MiniCPMO(_mk_config(cfg))
    sd = torch.load(cfg["pt_path"], map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict({k: v for k, v in sd.items() if not k.startswith("llm.")},
                          strict=False, assign=True); del sd
    model.llm = None
    model.to(device=device, dtype=torch.bfloat16)
    tp = AutoModelForCausalLM.from_pretrained(cfg["backbone_dir"], tp_plan="auto", dtype=torch.bfloat16)
    for m in tp.modules():
        c = getattr(m, "config", None)
        if c is not None and hasattr(c, "_experts_implementation"):
            c._experts_implementation = "batched_mm"
    model.llm = tp
    model.processor = MiniCPMOProcessor.from_pretrained(cfg["model_path"], trust_remote_code=True)
    return model


def _enable_engine(model, *, tp: bool, cfg: Dict[str, Any]):
    """Enable the deployed CUDA-graph optimization engine. Returns an engine-info dict.
    tp=True additionally installs the token-broadcast SPMD sync on the decoder."""
    from MiniCPMO45.opt_flags import OPT
    # deployed MoE on both paths
    seen = set()
    for m in model.modules():
        c = getattr(m, "config", None)
        if c is not None and hasattr(c, "_experts_implementation") and id(c) not in seen:
            c._experts_implementation = "batched_mm"; seen.add(id(c))
    for k in list(OPT):
        OPT[k] = False
    OPT.update({"tts_fast": True, "lmhead": True, "tts_graph": True,
                "fuse_vision_audio": True, "llm_graph": True})
    for m in model.tts.model.modules():
        c = getattr(m, "config", None)
        if c is not None and hasattr(c, "_attn_implementation"):
            c._attn_implementation = "eager"
    os.environ["O5_VISION_BATCH"] = "1"
    os.environ["O5_LLM_CACHE"] = str(cfg.get("llm_cache_len", 8192))
    eng = {"experts": "batched_mm", "tts_fast": True, "lmhead": True, "tts_graph": True,
           "fuse_vision_audio": True, "llm_graph": True, "llm_cache": cfg.get("llm_cache_len", 8192)}
    if tp:
        _install_token_broadcast(model)
        eng["tp"] = 2; eng["token_broadcast"] = True
    return eng


def _enable_vocoder_bucket(model):
    """Pre-capture the dominant vocoder chunk sizes off the serving hot path (call after warmup)."""
    try:
        from MiniCPMO45 import vocoder_graph as vg
        vg.enable_bucketed(model, lambda *a: logger.info("%s", " ".join(str(x) for x in a)), bucket=50)
    except Exception as e:
        logger.warning("[deploy] vocoder bucketing failed (staying eager): %s", e)


def _install_token_broadcast(model):
    """SPMD: broadcast rank-0's chosen token to all ranks so every rank feeds the SAME token ->
    identical per-layer NCCL all_reduce stream (non-deterministic MoE kernels would otherwise
    diverge at near-ties and desync the collectives)."""
    import torch.distributed as dist
    dec = model.duplex.decoder
    if getattr(dec, "_tp_broadcast_installed", False):
        return
    _orig = dec.decode
    def _synced(*a, **k):
        t = _orig(*a, **k)
        if torch.is_tensor(t):
            t = t.contiguous(); dist.broadcast(t, src=0)
        return t
    dec.decode = _synced
    dec._tp_broadcast_installed = True


def _init_unified(model, cfg: Dict[str, Any]):
    model.init_unified(
        preload_both_tts=cfg.get("preload_both_tts", True),
        duplex_config=cfg.get("duplex_config", _DEFAULT_DUP),
        device="cuda",
        chat_vocoder=cfg.get("chat_vocoder", "token2wav"),
    )


# ─────────────────────────── builders ───────────────────────────
def build_single_eager(cfg, rank=0, world_size=1) -> BuildResult:
    dev = f"cuda:{cfg.get('gpu_id', 0)}"
    model = _load_full(cfg, dev)
    _init_unified(model, cfg)
    return BuildResult(model=model, world_size=1, rank=0, is_driver=True,
                       engine={"mode": "single_eager"})


def build_single_opt(cfg, rank=0, world_size=1) -> BuildResult:
    dev = f"cuda:{cfg.get('gpu_id', 0)}"
    model = _load_full(cfg, dev)
    _init_unified(model, cfg)
    eng = _enable_engine(model, tp=False, cfg=cfg); eng["mode"] = "single_opt"
    return BuildResult(model=model, world_size=1, rank=0, is_driver=True, engine=eng)


def build_tp2(cfg, rank=0, world_size=2) -> BuildResult:
    import torch.distributed as dist
    from datetime import timedelta
    if not dist.is_initialized():
        dist.init_process_group("nccl", timeout=timedelta(seconds=int(cfg.get("nccl_timeout_s", 120))))
    rank = dist.get_rank(); world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = f"cuda:{rank}"
    model = _surgery_tp(cfg, dev)
    _init_unified(model, cfg)
    eng = _enable_engine(model, tp=True, cfg=cfg); eng["mode"] = "tp2"

    from .spmd import SpmdMirror
    model._spmd_mirror = SpmdMirror(model, is_driver=(rank == 0), rank=rank, world_size=world_size)

    def broadcast_input(obj):
        box = [obj]; dist.broadcast_object_list(box, src=0); return box[0]

    return BuildResult(model=model, world_size=world_size, rank=rank, is_driver=(rank == 0),
                       engine=eng, broadcast_input=broadcast_input)


register_mode(DeploymentMode("single_eager", 1, False, build_single_eager,
                             "1 GPU, trusted eager path (production default)."))
register_mode(DeploymentMode("single_opt", 1, False, build_single_opt,
                             "1 GPU, CUDA-graph optimization engine (batched_mm + tts/llm/vocoder graphs)."))
register_mode(DeploymentMode("tp2", 2, True, build_tp2,
                             "2 GPU tensor-parallel backbone (SPMD/torchrun) + graphs + token broadcast."))
