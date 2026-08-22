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
from .fla_runtime import configure_chunk_output, configure_fused_norm, configure_l2norm
from .registry import register_mode
from .weights import load_safetensors_into

logger = logging.getLogger("deploy.modes")

_DEFAULT_DUP = {
    "generate_audio": True, "ls_mode": "explicit", "max_new_speak_tokens_per_chunk": 20,
    "temperature": 0.7, "top_k": 100, "top_p": 0.8, "force_listen_count": 0,
}


def _experts_impl() -> str:
    return os.environ.get("O5_EXPERTS_IMPLEMENTATION", "batched_mm")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw not in {"0", "false", "False", "no", "NO", "off", "OFF"}


def _place_outer_model(model: torch.nn.Module, device: str) -> torch.nn.Module:
    """Place the outer model, optionally preserving floating-point buffer dtypes."""
    if not _env_flag("O5_PRESERVE_FLOAT_BUFFERS", True):
        return model.bfloat16().eval().to(device)

    for parameter in model.parameters():
        if parameter.is_floating_point():
            parameter.data = parameter.data.to(dtype=torch.bfloat16)
    return model.eval().to(device)


# ─────────────────────────── shared helpers ───────────────────────────
def _mk_config(cfg: Dict[str, Any]):
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(cfg["model_path"], trust_remote_code=True)
    c._attn_implementation = cfg.get("attn_implementation", "sdpa")
    c._name_or_path = cfg["model_path"]; c.name_or_path = cfg["model_path"]
    return c


def _load_full(cfg: Dict[str, Any], device: str):
    """Single-card: build MiniCPMO and load either HF shards or legacy PT."""
    # "auto" leaves FLA's normal autotuning untouched. Fixed values are only
    # for reproducibility investigations on the pinned FLA 0.5.0 runtime.
    configure_fused_norm(cfg.get("fla_fused_norm_config"))
    configure_l2norm(cfg.get("fla_l2norm_config"))
    configure_chunk_output(cfg.get("fla_chunk_output_config"))
    from accelerate import init_empty_weights
    from MiniCPMO45.modeling_minicpmo_unified import MiniCPMO
    from MiniCPMO45.processing_minicpmo import MiniCPMOProcessor
    with init_empty_weights():
        model = MiniCPMO(_mk_config(cfg))
    if cfg.get("weights_dir"):
        load_safetensors_into(model, cfg["weights_dir"])
    else:
        sd = torch.load(cfg["pt_path"], map_location="cpu", weights_only=True, mmap=True)
        model.load_state_dict(sd, strict=False, assign=True); del sd
    _place_outer_model(model, device)
    model.processor = MiniCPMOProcessor.from_pretrained(cfg["model_path"], trust_remote_code=True)
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
    # Apply any requested FLA pin before either rank creates its model.
    configure_fused_norm(cfg.get("fla_fused_norm_config"))
    configure_l2norm(cfg.get("fla_l2norm_config"))
    configure_chunk_output(cfg.get("fla_chunk_output_config"))
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    from .llm_wrapper import DistributedTPLLM
    from MiniCPMO45.modeling_minicpmo_unified import MiniCPMO
    from MiniCPMO45.processing_minicpmo import MiniCPMOProcessor
    with init_empty_weights():
        model = MiniCPMO(_mk_config(cfg))
    if cfg.get("weights_dir"):
        load_safetensors_into(model, cfg["weights_dir"], exclude_prefixes=("llm.",))
    else:
        sd = torch.load(cfg["pt_path"], map_location="cpu", weights_only=True, mmap=True)
        model.load_state_dict({k: v for k, v in sd.items() if not k.startswith("llm.")},
                              strict=False, assign=True); del sd
    model.llm = None
    _place_outer_model(model, device)
    # A complete O5 safetensors bundle also contains ``llm/config.json`` and
    # native Qwen ``model.*`` keys.  Reuse its root shards for TP2 so a second
    # 65GB LLM copy is not needed.  The legacy standalone backbone remains a
    # compatible fallback for older deployments.
    tp_root = cfg.get("weights_dir") or cfg["backbone_dir"]
    tp_config_root = (
        os.path.join(cfg["weights_dir"], "llm")
        if cfg.get("weights_dir") and os.path.isfile(os.path.join(cfg["weights_dir"], "llm", "config.json"))
        else cfg["backbone_dir"]
    )
    tp_cfg = AutoConfig.from_pretrained(tp_config_root, trust_remote_code=True)
    tp_cfg._attn_implementation = cfg.get("attn_implementation", "sdpa")
    tp = AutoModelForCausalLM.from_pretrained(
        tp_root,
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
    model.processor = MiniCPMOProcessor.from_pretrained(cfg["model_path"], trust_remote_code=True)
    return model


def _enable_engine(model, *, tp: bool, cfg: Dict[str, Any], token_broadcast: bool = True):
    """Enable the deployed CUDA-graph optimization engine. Returns an engine-info dict.
    tp=True additionally installs the token-broadcast SPMD sync on the decoder."""
    from MiniCPMO45.opt_flags import OPT
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
    vocoder_graph = _env_flag("O5_VOCODER_GRAPH", False)
    fuse_vision_audio = _env_flag("O5_FUSE_VISION_AUDIO", True)
    batch_vision = _env_flag("O5_VISION_BATCH", True)
    OPT.update({"tts_fast": tts_fast, "lmhead": lmhead, "tts_graph": tts_graph,
                "vocoder_graph": vocoder_graph, "fuse_vision_audio": fuse_vision_audio,
                "llm_graph": llm_graph})
    for m in model.tts.model.modules():
        c = getattr(m, "config", None)
        if c is not None and hasattr(c, "_attn_implementation"):
            c._attn_implementation = "eager"
    os.environ["O5_VISION_BATCH"] = "1" if batch_vision else "0"
    os.environ["O5_LLM_CACHE"] = str(cfg.get("llm_cache_len", 8192))
    vocoder_graph_enabled = _enable_vocoder_bucket(model) if vocoder_graph else False
    eng = {"experts": experts_impl, "tts_fast": tts_fast, "lmhead": lmhead, "tts_graph": tts_graph,
           "vocoder_graph": vocoder_graph_enabled,
           "fuse_vision_audio": fuse_vision_audio, "batch_vision_feed": batch_vision,
           "llm_graph": llm_graph, "llm_cache": cfg.get("llm_cache_len", 8192)}
    if tp and token_broadcast:
        _install_token_broadcast(model)
        eng["tp"] = 2; eng["token_broadcast"] = True
    elif tp:
        eng["tp"] = 2; eng["token_broadcast"] = False
    return eng


def _enable_vocoder_bucket(model):
    """Pre-capture the dominant vocoder chunk sizes off the serving hot path (call after warmup)."""
    try:
        from MiniCPMO45 import vocoder_graph as vg
        vg.enable_bucketed(model, lambda *a: logger.info("%s", " ".join(str(x) for x in a)), bucket=50)
        return True
    except Exception as e:
        logger.warning("[deploy] vocoder bucketing failed (staying eager): %s", e)
        return False


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
        assets_dir=cfg.get("assets_dir"),
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
    model = _surgery_tp(cfg, dev, rank=rank, world_size=world_size)
    _init_unified(model, cfg)
    eng = _enable_engine(model, tp=True, cfg=cfg); eng["mode"] = "tp2"

    from .spmd import SpmdMirror
    model._spmd_mirror = SpmdMirror(model, is_driver=(rank == 0), rank=rank, world_size=world_size)
    if hasattr(model.llm, "mirror"):
        model.llm.mirror = model._spmd_mirror

    def broadcast_input(obj):
        box = [obj]; dist.broadcast_object_list(box, src=0); return box[0]

    return BuildResult(model=model, world_size=world_size, rank=rank, is_driver=(rank == 0),
                       engine=eng, broadcast_input=broadcast_input)


def build_tp2_llm(cfg, rank=0, world_size=2) -> BuildResult:
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
    # LLM graph state is owned by StreamDecoder.feed(), outside the HF LLM
    # module.  When graph is enabled, mirror whole backend calls so every rank
    # advances decoder graph/cache state in lockstep; LLM-boundary-only sync does
    # not run that outer state on worker ranks.
    sync_llm_calls = not llm_graph
    model = _surgery_tp(cfg, dev, rank=rank, world_size=world_size, sync_llm_calls=sync_llm_calls)
    _init_unified(model, cfg)
    eng = _enable_engine(model, tp=True, cfg=cfg, token_broadcast=llm_graph); eng["mode"] = "tp2_llm"
    if llm_graph:
        from .spmd import SpmdMirror
        model._spmd_mirror = SpmdMirror(model, is_driver=(rank == 0), rank=rank, world_size=world_size)

        def broadcast_input(obj):
            box = [obj]; dist.broadcast_object_list(box, src=0); return box[0]

        return BuildResult(model=model, world_size=world_size, rank=rank, is_driver=(rank == 0),
                           engine=eng, broadcast_input=broadcast_input)
    model._spmd_worker_loop = model.llm.worker_loop
    model._spmd_shutdown = model.llm.shutdown_worker
    model._spmd_noop = model.llm.noop
    return BuildResult(model=model, world_size=world_size, rank=rank, is_driver=(rank == 0), engine=eng)


register_mode(DeploymentMode("single_eager", 1, False, build_single_eager,
                             "1 GPU, trusted eager path (production default)."))
register_mode(DeploymentMode("single_opt", 1, False, build_single_opt,
                             "1 GPU, CUDA-graph optimization engine (batched_mm + tts/llm/vocoder graphs)."))
register_mode(DeploymentMode("tp2", 2, True, build_tp2,
                             "2 GPU tensor-parallel backbone (SPMD/torchrun) + graphs + token broadcast."))
register_mode(DeploymentMode("tp2_llm", 2, True, build_tp2_llm,
                             "Experimental 2 GPU TP mode with synchronization at the LLM object boundary."))
