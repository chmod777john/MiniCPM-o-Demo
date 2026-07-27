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


def _experts_impl() -> str:
    return os.environ.get("O5_EXPERTS_IMPLEMENTATION", "batched_mm")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw not in {"0", "false", "False", "no", "NO", "off", "OFF"}


# ─────────────────────────── shared helpers ───────────────────────────
def _mk_config(cfg: Dict[str, Any]):
    from modeling.o5.configuration_minicpmo import MiniCPMOConfig

    c = MiniCPMOConfig.from_pretrained(cfg["model_path"])
    c._attn_implementation = cfg.get("attn_implementation", "sdpa")
    c._name_or_path = cfg["model_path"]; c.name_or_path = cfg["model_path"]
    return c


def _load_o5_processor(model_path: str):
    """只使用仓库内 O5 processor/tokenizer 代码加载外部资产。"""

    from modeling.o5.processing_minicpmo import (
        MiniCPMAAudioProcessor,
        MiniCPMOProcessor,
        MiniCPMVImageProcessor,
    )
    from modeling.o5.tokenization_minicpmo_fast import MiniCPMOTokenizerFast

    image_processor = MiniCPMVImageProcessor.from_pretrained(model_path)
    audio_processor = MiniCPMAAudioProcessor.from_pretrained(model_path)
    tokenizer = MiniCPMOTokenizerFast.from_pretrained(model_path)
    return MiniCPMOProcessor(
        image_processor=image_processor,
        audio_processor=audio_processor,
        tokenizer=tokenizer,
    )


def _load_full(cfg: Dict[str, Any], device: str):
    """Single-card: build MiniCPMO, load the full .pt, place on `device`, init_unified."""
    from accelerate import init_empty_weights
    from modeling.o5.modeling_minicpmo_unified import MiniCPMO
    with init_empty_weights():
        model = MiniCPMO(_mk_config(cfg))
    sd = torch.load(cfg["pt_path"], map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(sd, strict=False, assign=True); del sd
    model.bfloat16().eval().to(device)
    model.processor = _load_o5_processor(cfg["model_path"])
    return model


def _surgery_tp(
    cfg: Dict[str, Any],
    device: str,
    *,
    rank: int = 0,
    world_size: int = 1,
    sync_llm_calls: bool = False,
):
    """2-card TP: MiniCPMO with non-llm weights (mmap) + TP-sharded backbone via from_pretrained."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    from .llm_wrapper import DistributedTPLLM
    from modeling.o5.modeling_minicpmo_unified import MiniCPMO
    with init_empty_weights():
        model = MiniCPMO(_mk_config(cfg))
    sd = torch.load(cfg["pt_path"], map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict({k: v for k, v in sd.items() if not k.startswith("llm.")},
                          strict=False, assign=True); del sd
    model.llm = None
    model.to(device=device, dtype=torch.bfloat16)
    tp_cfg = AutoConfig.from_pretrained(cfg["backbone_dir"], trust_remote_code=True)
    tp_cfg._attn_implementation = cfg.get("attn_implementation", "sdpa")
    tp = AutoModelForCausalLM.from_pretrained(
        cfg["backbone_dir"],
        config=tp_cfg,
        tp_plan="auto",
        dtype=torch.bfloat16,
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
    )
    attn_impl = cfg.get("attn_implementation", "sdpa")
    experts_impl = _experts_impl()
    for m in tp.modules():
        c = getattr(m, "config", None)
        if c is not None:
            if hasattr(c, "_experts_implementation"):
                c._experts_implementation = experts_impl
            if hasattr(c, "_attn_implementation"):
                c._attn_implementation = attn_impl
    model.llm = DistributedTPLLM(
        tp,
        is_driver=(rank == 0),
        rank=rank,
        world_size=world_size,
        sync_calls=sync_llm_calls,
    )
    model.processor = _load_o5_processor(cfg["model_path"])
    return model


def _enable_engine(model, *, tp: bool, cfg: Dict[str, Any]):
    """Enable the deployed CUDA-graph optimization engine. Returns an engine-info dict.
    TP synchronization is handled by the LLM/graph runner, not by decoder-level
    sampling synchronization or backend method mirroring."""
    from modeling.o5.opt_flags import OPT
    # deployed MoE on both paths
    experts_impl = _experts_impl()
    seen = set()
    for m in model.modules():
        c = getattr(m, "config", None)
        if c is not None and hasattr(c, "_experts_implementation") and id(c) not in seen:
            c._experts_implementation = experts_impl; seen.add(id(c))
    for k in list(OPT):
        OPT[k] = False
    llm_graph = _env_flag("O5_LLM_GRAPH", True)
    tts_fast = _env_flag("O5_TTS_FAST", True)
    lmhead = _env_flag("O5_LMHEAD", True)
    tts_graph = _env_flag("O5_TTS_GRAPH", True)
    fuse_vision_audio = _env_flag("O5_FUSE_VISION_AUDIO", True)
    batch_vision = _env_flag("O5_VISION_BATCH", True)
    OPT.update({"tts_fast": tts_fast, "lmhead": lmhead, "tts_graph": tts_graph,
                "fuse_vision_audio": fuse_vision_audio, "llm_graph": llm_graph})
    for m in model.tts.model.modules():
        c = getattr(m, "config", None)
        if c is not None and hasattr(c, "_attn_implementation"):
            c._attn_implementation = "eager"
    os.environ["O5_VISION_BATCH"] = "1" if batch_vision else "0"
    os.environ["O5_LLM_CACHE"] = str(cfg.get("llm_cache_len", 8192))
    eng = {"experts": experts_impl, "tts_fast": tts_fast, "lmhead": lmhead, "tts_graph": tts_graph,
           "fuse_vision_audio": fuse_vision_audio, "batch_vision_feed": batch_vision,
           "llm_graph": llm_graph, "llm_cache": cfg.get("llm_cache_len", 8192)}
    if tp:
        eng["tp"] = 2; eng["token_broadcast"] = False
    return eng


def _enable_vocoder_bucket(model):
    """Pre-capture the dominant vocoder chunk sizes off the serving hot path (call after warmup)."""
    try:
        from modeling.o5 import vocoder_graph as vg
        vg.enable_bucketed(model, lambda *a: logger.info("%s", " ".join(str(x) for x in a)), bucket=50)
    except Exception as e:
        logger.warning("[deploy] vocoder bucketing failed (staying eager): %s", e)


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
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", rank)))
    torch.cuda.set_device(rank)
    if not dist.is_initialized():
        dist.init_process_group(
            "nccl",
            timeout=timedelta(seconds=int(cfg.get("nccl_timeout_s", 120))),
            device_id=torch.device(f"cuda:{rank}"),
        )
    rank = dist.get_rank(); world_size = dist.get_world_size()
    dev = f"cuda:{rank}"
    llm_graph = os.environ.get("O5_LLM_GRAPH", "1") not in {"0", "false", "False"}
    sync_llm_calls = True
    model = _surgery_tp(cfg, dev, rank=rank, world_size=world_size, sync_llm_calls=sync_llm_calls)
    _init_unified(model, cfg)
    eng = _enable_engine(model, tp=True, cfg=cfg); eng["mode"] = "tp2"
    if llm_graph:
        runner = model.duplex.decoder.ensure_llm_graph_runner()
        model._spmd_worker_loop = runner.worker_loop
        model._spmd_shutdown = runner.shutdown_worker
        model._spmd_noop = runner.noop
        return BuildResult(model=model, world_size=world_size, rank=rank, is_driver=(rank == 0), engine=eng)
    model._spmd_worker_loop = model.llm.worker_loop
    model._spmd_shutdown = model.llm.shutdown_worker
    model._spmd_noop = model.llm.noop
    return BuildResult(model=model, world_size=world_size, rank=rank, is_driver=(rank == 0), engine=eng)


register_mode(DeploymentMode("single_eager", 1, False, build_single_eager,
                             "1 GPU, trusted eager path (production default)."))
register_mode(DeploymentMode("single_opt", 1, False, build_single_opt,
                             "1 GPU, CUDA-graph optimization engine (batched_mm + tts/llm/vocoder graphs)."))
register_mode(DeploymentMode("tp2", 2, True, build_tp2,
                             "2 GPU tensor-parallel backbone with LLM/graph-boundary SPMD synchronization."))
register_mode(DeploymentMode("tp2_llm", 2, True, build_tp2,
                             "Alias for tp2."))
