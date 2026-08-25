"""Canonical and Demo runtime adapters for the common replay loop."""

from __future__ import annotations

import importlib
import os
import random
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import init_empty_weights


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _unwrap_state_dict(state: Any) -> Any:
    for key in ("state_dict", "model", "module"):
        if isinstance(state, dict) and isinstance(state.get(key), dict):
            return state[key]
    return state


def _cast_floating_parameters(module: torch.nn.Module, dtype: torch.dtype) -> None:
    """Match HF parameter casting while preserving float32 RoPE buffers."""

    for parameter in module.parameters():
        if parameter.is_floating_point() and parameter.dtype != dtype:
            parameter.data = parameter.data.to(dtype=dtype)


@dataclass
class RuntimeAdapter:
    implementation: str
    model: Any
    duplex: Any
    backend: Any = None

    def prepare(self, *, system_prompt: str, ref_audio_path: str, seed: int, sampling: dict[str, Any]) -> None:
        if self.backend is not None:
            self.backend.duplex_prepare(
                system_prompt_text=system_prompt,
                ref_audio_path=ref_audio_path,
                prompt_wav_path=ref_audio_path,
                sampling=sampling,
                llm_seed=seed,
            )
            return
        import librosa

        ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
        self.duplex.prepare(
            prefix_system_prompt=system_prompt,
            ref_audio=np.asarray(ref_audio, dtype=np.float32),
            prompt_wav_path=ref_audio_path,
            llm_seed=seed,
        )

    def prefill(self, *, audio: np.ndarray, frames: list[Any], max_slice_nums: int) -> Any:
        if self.backend is not None:
            return self.backend.duplex_prefill(
                audio_waveform=audio,
                frame_list=frames or None,
                max_slice_nums=max_slice_nums,
            )
        return self.duplex.streaming_prefill(
            audio_waveform=audio,
            frame_list=frames or None,
            max_slice_nums=max_slice_nums,
            batch_vision_feed=False,
        )

    def generate(self, *, force_listen: bool, prompt_wav_path: str, sampling: dict[str, Any]) -> Any:
        if self.backend is not None:
            return self.backend.duplex_generate(force_listen=force_listen)
        old_force = getattr(self.duplex, "force_listen_count", 0)
        old_count = getattr(self.duplex, "_streaming_generate_count", 0)
        if force_listen:
            self.duplex.force_listen_count = old_count + 1
        try:
            names = (
                "max_new_speak_tokens_per_chunk",
                "decode_mode",
                "temperature",
                "top_k",
                "top_p",
                "listen_prob_scale",
                "text_repetition_penalty",
                "text_repetition_window_size",
            )
            generate_kwargs = {name: sampling[name] for name in names if name in sampling}
            return self.duplex.streaming_generate(
                prompt_wav_path=prompt_wav_path,
                **generate_kwargs,
            )
        finally:
            self.duplex.force_listen_count = old_force

    def finalize_unit(self) -> None:
        if self.backend is not None:
            self.backend.duplex_finalize()

    def shutdown(self) -> None:
        if self.backend is None:
            return
        shutdown = self.backend._get_spmd_shutdown()
        if callable(shutdown) and getattr(self.backend, "spmd_is_driver", False):
            shutdown()
            return
        mirror = self.backend._get_spmd_mirror()
        if mirror is not None and getattr(mirror, "is_driver", False):
            mirror.shutdown()


def load_canonical(args: Any, sampling: dict[str, Any]) -> RuntimeAdapter:
    from core.deploy.fla_runtime import (
        configure_chunk_output,
        configure_fused_norm,
        configure_l2norm,
    )

    # Apply the same pinned FLA autotuner choices before Canonical constructs
    # its model. This makes repeated Canonical runs a controlled baseline.
    configure_fused_norm(args.fla_fused_norm_config)
    configure_l2norm(args.fla_l2norm_config)
    configure_chunk_output(args.fla_chunk_output_config)
    root = Path(args.canonical_root).resolve()
    if not (root / "configuration_minicpmo.py").is_file():
        raise FileNotFoundError(f"canonical model code not found: {root}")
    package_name = f"o5_replay_canonical_{os.getpid()}"
    package = types.ModuleType(package_name)
    package.__path__ = [str(root)]
    sys.modules[package_name] = package
    config_cls = importlib.import_module(f"{package_name}.configuration_minicpmo").MiniCPMOConfig
    modeling = importlib.import_module(f"{package_name}.modeling_minicpmo")
    processor_cls = importlib.import_module(f"{package_name}.processing_minicpmo").MiniCPMOProcessor

    config = config_cls.from_pretrained(root, local_files_only=True)
    config._attn_implementation = args.attn_implementation
    # MiniCPMO creates the Qwen text config independently from the outer
    # multimodal config. The private Transformers field is omitted by
    # ``to_dict()``, so expose the requested value through the public field
    # copied by the canonical model constructor as well.
    config.attn_implementation = args.attn_implementation
    config._name_or_path = str(root)
    config.name_or_path = str(root)
    with init_empty_weights():
        model = modeling.MiniCPMO(config)
    state = torch.load(args.ckpt_path, map_location="cpu", weights_only=True, mmap=True)
    info = model.load_state_dict(_unwrap_state_dict(state), strict=False, assign=True)
    del state
    _cast_floating_parameters(model, torch.bfloat16)
    model.eval().to(args.device)
    rope = model.llm.model.rotary_emb.inv_freq
    if rope.dtype != torch.float32:
        raise RuntimeError(f"canonical RoPE inv_freq must stay float32, got {rope.dtype}")
    model.processor = processor_cls.from_pretrained(root, local_files_only=True, trust_remote_code=True)
    original_init_tts = model.init_tts

    def init_tts_from_assets(model_dir=None, *call_args, **call_kwargs):
        return original_init_tts(
            model_dir=model_dir or str(Path(args.token2wav_dir).resolve()),
            *call_args,
            **call_kwargs,
        )

    model.init_tts = init_tts_from_assets
    try:
        duplex = modeling.MiniCPMODuplex.from_existing_model(
            model,
            device=args.device,
            generate_audio=bool(sampling["generate_audio"]),
            ls_mode=sampling["ls_mode"],
            max_new_speak_tokens_per_chunk=sampling["max_new_speak_tokens_per_chunk"],
            force_listen_count=sampling["force_listen_count"],
            tts_temperature=sampling["tts_temperature"],
            tts_repetition_penalty=sampling["tts_repetition_penalty"],
            n_timesteps=sampling["n_timesteps"],
        )
    finally:
        model.init_tts = original_init_tts
    print(
        f"canonical load_state_dict: missing={len(info.missing_keys)} "
        f"unexpected={len(info.unexpected_keys)}",
        flush=True,
    )
    return RuntimeAdapter("canonical", model, duplex)


def _set_flag(name: str, enabled: bool) -> None:
    os.environ[name] = "1" if enabled else "0"


def configure_demo_environment(args: Any) -> None:
    os.environ["O5_DEPLOY_MODE"] = "tp2" if args.target == "demo-tp2" else args.demo_single_mode
    os.environ["O5_BACKBONE_DIR"] = str(args.backbone_dir)
    os.environ["O5_EXPERTS_IMPLEMENTATION"] = args.experts_implementation
    os.environ["O5_LLM_CACHE"] = str(args.llm_cache)
    # Keep the Demo deployment on the same pinned FLA kernels as Canonical.
    # core.deploy reads these controls before constructing the model; recording
    # them only in the replay manifest is insufficient because autotuning can
    # otherwise choose a different reduction/tiling on the Demo path.
    os.environ["O5_FLA_FUSED_NORM_CONFIG"] = args.fla_fused_norm_config
    os.environ["O5_FLA_L2NORM_CONFIG"] = args.fla_l2norm_config
    os.environ["O5_FLA_CHUNK_OUTPUT_CONFIG"] = args.fla_chunk_output_config
    _set_flag("O5_LLM_GRAPH", args.llm_graph)
    _set_flag("O5_TTS_GRAPH", args.tts_graph)
    _set_flag("O5_VOCODER_GRAPH", args.vocoder_graph)
    _set_flag("O5_TTS_FAST", args.tts_fast)
    _set_flag("O5_LMHEAD", args.lmhead)
    _set_flag("O5_FUSE_VISION_AUDIO", args.fuse_vision_audio)
    _set_flag("O5_VISION_BATCH", args.batch_vision_feed)
    _set_flag("O5_LAYER_TRACE", args.capture_layers)


def load_demo(args: Any, sampling: dict[str, Any]) -> RuntimeAdapter:
    configure_demo_environment(args)
    from core.processors.backend_factory import create_backend

    mode = "tp2" if args.target == "demo-tp2" else args.demo_single_mode
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    backend = create_backend({
        "deployment_mode": mode,
        "model_path": str(args.model_path),
        "pt_path": str(args.ckpt_path),
        "backbone_dir": str(args.backbone_dir),
        "gpu_id": rank,
        "chat_vocoder": "token2wav",
        "attn_implementation": args.attn_implementation,
        "llm_cache_len": args.llm_cache,
        "duplex_config": sampling,
        "fla_fused_norm_config": args.fla_fused_norm_config,
        "fla_l2norm_config": args.fla_l2norm_config,
        "fla_chunk_output_config": args.fla_chunk_output_config,
    })
    backend.load_model()
    if getattr(backend, "spmd_is_worker", False):
        worker_loop = backend._get_spmd_worker_loop()
        if not callable(worker_loop):
            raise RuntimeError("TP2 backend worker has no model-provided worker loop")
        worker_loop()
        sys.stdout.flush()
        os._exit(0)
    model = backend.processor.model
    duplex = model.duplex
    for name, value in sampling.items():
        if hasattr(duplex, name):
            setattr(duplex, name, value)
    return RuntimeAdapter(args.target, model, duplex, backend=backend)


def load_runtime(args: Any, sampling: dict[str, Any]) -> RuntimeAdapter:
    if args.target == "canonical":
        return load_canonical(args, sampling)
    return load_demo(args, sampling)
