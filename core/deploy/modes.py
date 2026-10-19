"""Build external HF models and optionally attach the supplied accel package.

All deployment modes run in one Python process. TP/EP implementation and graph
policy belong to accel; this module only loads assets and forwards configuration.
"""
from __future__ import annotations
import os
import logging
from typing import Any, Dict

import torch
from .base import DeploymentMode, BuildResult
from .registry import register_mode
from core.external_o5 import canonical_module, checkpoint_path, checkpoint_state

logger = logging.getLogger("deploy.modes")
_DEFAULT_DUP = {
    "generate_audio": True, "ls_mode": "explicit", "max_new_speak_tokens_per_chunk": 20,
    "temperature": 0.7, "top_k": 20, "top_p": 0.8, "force_listen_count": 3,
}

def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw not in {"0", "false", "False", "no", "NO", "off", "OFF"}


def _mk_config(cfg: Dict[str, Any]):
    MiniCPMOConfig = canonical_module("configuration_minicpmo").MiniCPMOConfig

    c = MiniCPMOConfig.from_pretrained(cfg["model_path"])
    c._attn_implementation = cfg.get("attn_implementation", "sdpa")
    c._name_or_path = cfg["model_path"]; c.name_or_path = cfg["model_path"]
    return c


def _load_o5_processor(model_path: str):
    """Load processor/tokenizer code from the configured external HF checkout."""
    processing = canonical_module("processing_minicpmo")
    tokenizer_class = canonical_module("tokenization_minicpmo_fast").MiniCPMOTokenizerFast
    return processing.MiniCPMOProcessor(
        image_processor=processing.MiniCPMVImageProcessor.from_pretrained(model_path),
        audio_processor=processing.MiniCPMAAudioProcessor.from_pretrained(model_path),
        tokenizer=tokenizer_class.from_pretrained(model_path),
    )


def _load_full(cfg: Dict[str, Any], device: str):
    """Single-card: build MiniCPMO, load the full .pt, place on `device`, init_unified."""
    from accelerate import init_empty_weights
    from modeling.external_o5_facade import MiniCPMO
    config = _mk_config(cfg)
    sd = checkpoint_state(checkpoint_path(cfg))
    checkpoint_vocab_size = _checkpoint_vocab_size(sd)
    configured_vocab_size = int(
        getattr(config, "vocab_size", checkpoint_vocab_size)
    )
    if checkpoint_vocab_size != configured_vocab_size:
        logger.warning(
            "Checkpoint/config vocabulary mismatch: checkpoint=%d, config=%d. "
            "Building the single-card O5 LLM with checkpoint vocabulary size.",
            checkpoint_vocab_size,
            configured_vocab_size,
        )
        config.vocab_size = checkpoint_vocab_size
    with init_empty_weights():
        model = MiniCPMO(config)
    info = model.load_state_dict(sd, strict=False, assign=True)
    unexpected = [key for key in info.unexpected_keys if not key.startswith("llm.mtp.")]
    if info.missing_keys or unexpected:
        raise ValueError(f"Checkpoint mismatch: missing={info.missing_keys[:10]}, unexpected={unexpected[:10]}")
    model._checkpoint_load_report = {"missing": [], "unexpected": [],
                                     "excluded_training_keys": info.unexpected_keys}
    for param in model.parameters():
        if param.is_floating_point():
            param.data = param.data.to(torch.bfloat16)
    model.eval().to(device).requires_grad_(False)
    del sd
    model.processor = _load_o5_processor(cfg["model_path"])
    return model


def _checkpoint_vocab_size(state_dict: Dict[str, Any]) -> int:
    """读取完整 O5 PT 中 embedding 与 lm_head 的共同词表行数。

    参数:
        state_dict: 完整 O5 checkpoint state_dict。

    返回:
        embedding 与 lm_head 一致的词表行数。
    """

    embedding_key = "llm.model.embed_tokens.weight"
    lm_head_key = "llm.lm_head.weight"
    if embedding_key not in state_dict or lm_head_key not in state_dict:
        raise KeyError(
            "完整 O5 PT 缺少 embedding/lm_head: "
            f"{embedding_key}, {lm_head_key}"
        )
    embedding_rows = int(state_dict[embedding_key].shape[0])
    lm_head_rows = int(state_dict[lm_head_key].shape[0])
    if embedding_rows != lm_head_rows:
        raise ValueError(
            "O5 PT embedding/lm_head 词表行数不一致: "
            f"embedding={embedding_rows}, lm_head={lm_head_rows}"
        )
    return embedding_rows


def _enable_engine(model, *, tp: bool, cfg: Dict[str, Any]):
    """Enable the deployed CUDA-graph optimization engine. Returns an engine-info dict.
    TP synchronization is handled by the LLM/graph runner, not by decoder-level
    sampling synchronization or backend method mirroring."""
    if not os.environ.get("O5_ACCELERATION_PATH"):
        return {"acceleration": "disabled", "tp": 2 if tp else 1}
    from core.external_o5 import acceleration_module
    accel = acceleration_module()
    options = dict(cfg.get("acceleration_config", {}))
    options.setdefault("hardware", os.environ.get("O5_ACCELERATION_HARDWARE", "auto"))
    for option, env in (("llm_graph", "O5_LLM_GRAPH"), ("tts_graph", "O5_TTS_GRAPH"),
                        ("t2w_graph", "O5_VOCODER_GRAPH"), ("compact_logits", "O5_COMPACT_LOGITS"),
                        ("merge_unit_feed", "CANONICAL_UNIT_PREFILL_BATCH")):
        options.setdefault(option, _env_flag(env, True))
    for option, env in (("tts_fa2_whole_decode", "O5_TTS_FA2_WHOLE_DECODE"),
                        ("canvas_window_exact_chain", "O5_CANVAS_WINDOW_EXACT_CHAIN")):
        options.setdefault(option, _env_flag(env, False))
    options.setdefault("mode", cfg.get("parallel", "tp") if tp else "single")
    if tp:
        owner = int(cfg.get("gpu_id", 0))
        options.setdefault("devices", (f"cuda:{owner}", f"cuda:{owner + 1}"))
    else:
        options.setdefault("devices", (str(next(model.llm.parameters()).device),))
    accel.attach(model, accel.AccelerationConfig(**options))
    return {"acceleration": os.environ["O5_ACCELERATION_PATH"], "tp": 2 if tp else 1,
            "acceleration_info": dict(model._acceleration_info)}


def _init_unified(model, cfg: Dict[str, Any]):
    model.init_unified(
        preload_both_tts=cfg.get("preload_both_tts", True),
        duplex_config=cfg.get("duplex_config", _DEFAULT_DUP),
        device=str(next(model.llm.parameters()).device),
        chat_vocoder=cfg.get("chat_vocoder", "token2wav"),
    )


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
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise RuntimeError("External O5 TP/EP uses one process; launch without torchrun")
    if not os.environ.get("O5_ACCELERATION_PATH"):
        raise RuntimeError("External TP/EP requires O5_ACCELERATION_PATH")
    cfg = dict(cfg)
    cfg.setdefault("parallel", os.environ.get("O5_MOE_PARALLEL", "tp").lower())
    if cfg["parallel"] not in {"tp", "ep"}:
        raise ValueError("O5_MOE_PARALLEL must be tp or ep")
    owner = int(cfg.get("gpu_id", 0))
    torch.cuda.set_device(owner)
    if owner + 1 >= torch.cuda.device_count():
        raise RuntimeError("External TP/EP requires two visible GPUs")
    model = _load_full(cfg, f"cuda:{owner}")
    from core.external_o5 import acceleration_module
    acceleration_module()
    from minicpm_o_accel.llm.initialization_memory import reserve_resize_memory
    with reserve_resize_memory(model.llm):
        _init_unified(model, cfg)
    eng = _enable_engine(model, tp=True, cfg=cfg)
    eng["mode"] = cfg["parallel"]
    return BuildResult(model=model, world_size=1, rank=0, is_driver=True, engine=eng)

register_mode(DeploymentMode("single_eager", 1, False, build_single_eager,
                             "External HF on one GPU, without accel."))
register_mode(DeploymentMode("single_opt", 1, False, build_single_opt,
                             "External HF on one GPU with optional external accel."))
register_mode(DeploymentMode("tp2", 1, False, build_tp2,
                             "External HF with two-GPU TP/EP managed by accel in one process."))
register_mode(DeploymentMode("tp2_llm", 1, False, build_tp2, "Alias for tp2."))
