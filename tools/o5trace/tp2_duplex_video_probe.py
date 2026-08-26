#!/usr/bin/env python3
"""Offline video duplex probe through the TP2 deployment builder."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from PIL import Image

from thin_duplex_video_probe import (
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    audio_meta,
    argmax_multinomial,
    configure_seed,
    extract_video,
    generate_with_optional_argmax,
    install_token_trace,
    load_ref,
    reset_vocoder_static_state,
    split_audio,
)


def _set_env_flag(name: str, enabled: bool) -> None:
    os.environ[name] = "1" if enabled else "0"


def _apply_safe_engine_env(args: argparse.Namespace) -> None:
    os.environ["O5_EXPERTS_IMPLEMENTATION"] = args.experts_implementation
    os.environ["O5_GROUPED_PREFILL_MIN_TOKENS"] = str(args.grouped_prefill_min_tokens)
    _set_env_flag("O5_LLM_GRAPH", args.llm_graph)
    _set_env_flag("O5_TTS_GRAPH", args.tts_graph)
    _set_env_flag("O5_VOCODER_GRAPH", args.vocoder_graph)
    _set_env_flag("O5_TTS_FAST", args.tts_fast)
    _set_env_flag("O5_LMHEAD", args.lmhead)
    _set_env_flag("O5_FUSE_VISION_AUDIO", args.fuse_vision_audio)
    _set_env_flag("O5_VISION_BATCH", args.batch_vision_feed)
    _set_env_flag("O5_UNIT_PREFILL_BATCH", args.unit_prefill_batch)
    os.environ["O5_LLM_CACHE"] = str(args.o5_llm_cache)


def _install_global_argmax_multinomial() -> None:
    torch.multinomial = argmax_multinomial


def _tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    data = tensor.detach().float().cpu()
    flat = data.reshape(-1)
    return {
        "shape": list(data.shape),
        "mean": float(flat.mean()) if flat.numel() else 0.0,
        "std": float(flat.std(unbiased=False)) if flat.numel() else 0.0,
        "min": float(flat.min()) if flat.numel() else 0.0,
        "max": float(flat.max()) if flat.numel() else 0.0,
        "absmax": float(flat.abs().max()) if flat.numel() else 0.0,
        "l2": float(torch.linalg.vector_norm(flat)) if flat.numel() else 0.0,
    }


def _tensor_topk_distribution(scores: torch.Tensor, k: int = 5) -> dict[str, Any]:
    data = scores.detach().float().cpu()
    rows = data.reshape(-1, data.shape[-1])
    top_k = min(k, rows.shape[-1])
    vals, idx = torch.topk(rows, k=top_k, dim=-1)
    top1 = vals[:, 0]
    top2 = vals[:, 1] if top_k > 1 else torch.zeros_like(top1)
    margin = top1 - top2
    entropy = -(rows.clamp_min(1e-45) * rows.clamp_min(1e-45).log()).sum(dim=-1)
    return {
        "shape": list(data.shape),
        "topk_indices": [[int(x) for x in row] for row in idx.tolist()],
        "topk_values": [[float(x) for x in row] for row in vals.tolist()],
        "top1": [float(x) for x in top1.tolist()],
        "top2": [float(x) for x in top2.tolist()],
        "margin": [float(x) for x in margin.tolist()],
        "min_margin": float(margin.min()) if margin.numel() else 0.0,
        "entropy": [float(x) for x in entropy.tolist()],
    }


def install_distribution_trace(duplex: Any, model: Any, out_dir: Path) -> tuple[dict[str, Any], Any]:
    trace: dict[str, Any] = {
        "current_unit": None,
        "tts_conditions": [],
        "multinomial_calls": [],
    }
    tensor_dir = out_dir / "tensor_trace"
    tensor_dir.mkdir(parents=True, exist_ok=True)

    original_convert = duplex._convert_results_to_tts_input

    def traced_convert(self, results):
        unit_id = trace["current_unit"]
        call_id = len(trace["tts_conditions"])
        record: dict[str, Any] = {
            "unit_id": unit_id,
            "call_id": call_id,
            "result_len": len(results),
            "tokens": [int(item[0]) for item in results],
            "end_of_turn": [bool(item[2]) for item in results],
        }
        arrays: dict[str, np.ndarray] = {}
        if results:
            raw_hidden = torch.cat([item[1].squeeze(0) for item in results], dim=0)
            token_tensor = torch.tensor(record["tokens"], device=self.device, dtype=torch.long)
            llm_embeds = self.model.tts.emb_text(token_tensor)
            projected = self.model.tts.projector_semantic(raw_hidden)
            normalized = torch.nn.functional.normalize(projected, p=2, dim=-1)
            tts_embeds = llm_embeds + normalized
            record.update({
                "raw_hidden": _tensor_stats(raw_hidden),
                "projected_hidden": _tensor_stats(projected),
                "normalized_hidden": _tensor_stats(normalized),
                "llm_embeds": _tensor_stats(llm_embeds),
                "tts_embeds_no_bos": _tensor_stats(tts_embeds),
            })
            arrays.update({
                "raw_hidden": raw_hidden.detach().float().cpu().numpy(),
                "projected_hidden": projected.detach().float().cpu().numpy(),
                "normalized_hidden": normalized.detach().float().cpu().numpy(),
                "llm_embeds": llm_embeds.detach().float().cpu().numpy(),
                "tts_embeds_no_bos": tts_embeds.detach().float().cpu().numpy(),
            })
        output = original_convert(results)
        record["tts_condition"] = _tensor_stats(output)
        arrays["tts_condition"] = output.detach().float().cpu().numpy()
        npz_name = f"unit_{int(unit_id) if unit_id is not None else -1:03d}_tts_condition_{call_id:02d}.npz"
        np.savez_compressed(tensor_dir / npz_name, **arrays)
        record["npz"] = f"tensor_trace/{npz_name}"
        trace["tts_conditions"].append(record)
        return output

    duplex._convert_results_to_tts_input = types.MethodType(traced_convert, duplex)

    original_multinomial = torch.multinomial

    def traced_multinomial(input_tensor, num_samples, replacement=False, *, generator=None, out=None):
        if num_samples == 1:
            record = {
                "unit_id": trace["current_unit"],
                "call_id": len(trace["multinomial_calls"]),
                "distribution": _tensor_topk_distribution(input_tensor),
            }
            trace["multinomial_calls"].append(record)
            return argmax_multinomial(input_tensor, num_samples, replacement=replacement, generator=generator, out=out)
        return original_multinomial(
            input_tensor,
            num_samples,
            replacement=replacement,
            generator=generator,
            out=out,
        )

    torch.multinomial = traced_multinomial
    return trace, original_multinomial


def _duplex_sampling_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "generate_audio": True,
        "ls_mode": "explicit",
        "force_listen_count": args.force_listen_count,
        "max_new_speak_tokens_per_chunk": args.max_new_speak_tokens_per_chunk,
        "decode_mode": args.decode_mode,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "listen_prob_scale": args.listen_prob_scale,
        "text_repetition_penalty": args.text_repetition_penalty,
        "text_repetition_window_size": args.text_repetition_window_size,
        "length_penalty": args.length_penalty,
        "strategy_hd": args.strategy_hd,
        "strategy_hd_max_slice_nums": args.strategy_hd_max_slice_nums,
    }


def _engine_summary(args: argparse.Namespace) -> dict[str, Any]:
    tp = 2 if args.deployment_mode in {"tp2", "tp2_llm"} else 1
    return {
        "experts": args.experts_implementation,
        "grouped_prefill_min_tokens": args.grouped_prefill_min_tokens,
        "tts_fast": args.tts_fast,
        "lmhead": args.lmhead,
        "tts_graph": args.tts_graph,
        "vocoder_graph": args.vocoder_graph,
        "fuse_vision_audio": args.fuse_vision_audio,
        "batch_vision_feed": args.batch_vision_feed,
        "unit_prefill_batch": args.unit_prefill_batch,
        "llm_graph": args.llm_graph,
        "llm_cache": args.o5_llm_cache,
        "tp": tp,
        "token_broadcast": args.deployment_mode == "tp2",
        "mode": args.deployment_mode,
    }


def _build_tp2_model(args: argparse.Namespace):
    import core.deploy as deploy

    cfg = {
        "model_path": args.model_path,
        "pt_path": args.ckpt_path,
        "backbone_dir": args.backbone_dir,
        "chat_vocoder": "token2wav",
        "attn_implementation": args.attn_implementation,
        "llm_cache_len": args.o5_llm_cache,
        "preload_both_tts": False,
        "duplex_config": {
            "generate_audio": True,
            "ls_mode": "explicit",
            "max_new_speak_tokens_per_chunk": args.max_new_speak_tokens_per_chunk,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "force_listen_count": args.force_listen_count,
            "n_timesteps": args.n_timesteps,
            "strategy_hd": args.strategy_hd,
            "strategy_hd_max_slice_nums": args.strategy_hd_max_slice_nums,
        },
    }
    return deploy.get_mode(args.deployment_mode).build(cfg)


def _build_backend(args: argparse.Namespace):
    from core.processors.backend_factory import create_backend

    os.environ["O5_DEPLOY_MODE"] = args.deployment_mode
    os.environ["O5_BACKBONE_DIR"] = args.backbone_dir
    rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    backend = create_backend({
        "deployment_mode": args.deployment_mode,
        "model_path": args.model_path,
        "pt_path": args.ckpt_path,
        "backbone_dir": args.backbone_dir,
        "gpu_id": rank,
        "chat_vocoder": "token2wav",
        "attn_implementation": args.attn_implementation,
        "llm_cache_len": args.o5_llm_cache,
        "duplex_config": _duplex_sampling_config(args),
    })
    backend.load_model()
    _apply_backend_duplex_runtime_config(backend, args)
    return backend


def _apply_backend_duplex_runtime_config(backend: object, args: argparse.Namespace) -> None:
    model = getattr(getattr(backend, "processor", None), "model", None)
    duplex = getattr(model, "duplex", None)
    if duplex is None:
        return
    for name, value in {
        "generate_audio": True,
        "ls_mode": "explicit",
        "force_listen_count": args.force_listen_count,
        "max_new_speak_tokens_per_chunk": args.max_new_speak_tokens_per_chunk,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "listen_prob_scale": args.listen_prob_scale,
        "text_repetition_penalty": args.text_repetition_penalty,
        "text_repetition_window_size": args.text_repetition_window_size,
    }.items():
        if hasattr(duplex, name):
            setattr(duplex, name, value)


def _run_backend_worker_loop(backend: object) -> None:
    mirror = getattr(backend, "_get_spmd_mirror")()
    if mirror is None:
        raise RuntimeError("backend call-path requires a whole-backend SPMD mirror")
    mirror.worker_loop()
    sys.stdout.flush()
    os._exit(0)


def _run_worker_loop(model: object) -> None:
    worker_loop = getattr(model, "_spmd_worker_loop", None)
    if worker_loop is not None:
        worker_loop()
        sys.stdout.flush()
        os._exit(0)
    mirror = getattr(model, "_spmd_mirror", None)
    if mirror is not None:
        mirror.worker_loop()
        sys.stdout.flush()
        os._exit(0)
    raise RuntimeError("TP2 worker rank has no worker loop")


def _shutdown_tp2(model: object) -> None:
    shutdown = getattr(model, "_spmd_shutdown", None)
    if shutdown is not None:
        shutdown()
        return
    mirror = getattr(model, "_spmd_mirror", None)
    if mirror is not None:
        mirror.shutdown()


def _backend_result_value(result: Any, name: str, default: Any = None) -> Any:
    if isinstance(result, dict):
        return result.get(name, default)
    return getattr(result, name, default)


def _backend_audio_waveform(result: Any) -> np.ndarray | None:
    audio_data = _backend_result_value(result, "audio_data")
    if not audio_data:
        return None
    raw = base64.b64decode(audio_data)
    return np.frombuffer(raw, dtype=np.float32).copy()


def _shutdown_backend(backend: object) -> None:
    mirror = getattr(backend, "_get_spmd_mirror")()
    if mirror is not None and getattr(mirror, "is_driver", False):
        mirror.shutdown()


def _run_backend_probe(args: argparse.Namespace, out_dir: Path) -> int:
    if args.deployment_mode == "tp2_llm":
        raise RuntimeError("--call-path backend is not valid with --deployment-mode tp2_llm")

    configure_seed(args.seed)
    backend = _build_backend(args)
    if getattr(backend, "spmd_is_worker", False):
        _run_backend_worker_loop(backend)

    model = backend.processor.model
    frames, audio = extract_video(Path(args.video), out_dir / "media", args.chunk_ms)
    audio_units = int(np.ceil(len(audio) / float(int(INPUT_SAMPLE_RATE * args.chunk_ms / 1000))))
    num_units = max(len(frames), audio_units)
    if args.max_units > 0:
        num_units = min(num_units, args.max_units)
    chunks = split_audio(audio, num_units, args.chunk_ms)

    token_trace = install_token_trace(model.duplex, model) if args.trace_token2wav else None
    distribution_trace, original_multinomial = (
        install_distribution_trace(model.duplex, model, out_dir)
        if args.trace_distribution
        else (None, None)
    )
    vocoder_state = reset_vocoder_static_state(model, args.seed)

    configure_seed(args.seed)
    backend.duplex_prepare(
        system_prompt_text="Streaming Omni Conversation.",
        ref_audio_path=args.prompt_wav,
        prompt_wav_path=args.prompt_wav,
        sampling=_duplex_sampling_config(args),
    )

    units = []
    engine = _engine_summary(args)
    try:
        for idx in range(num_units):
            if token_trace is not None:
                token_trace["current_unit"] = idx
            if distribution_trace is not None:
                distribution_trace["current_unit"] = idx
            frame = Image.open(frames[min(idx, len(frames) - 1)]).convert("RGB") if frames else None
            frame_list = [frame] if frame is not None else None
            prefill = backend.duplex_prefill(
                audio_waveform=chunks[idx],
                frame_list=frame_list,
                max_slice_nums=None if args.strategy_hd else 1,
            )
            result = backend.duplex_generate()
            backend.duplex_finalize()
            wav = _backend_audio_waveform(result)
            meta = audio_meta(wav)
            if wav is not None and meta["samples"] > 0:
                sf.write(out_dir / f"unit_{idx:03d}.wav", np.asarray(wav, dtype=np.float32), OUTPUT_SAMPLE_RATE)
            units.append({
                "unit_id": idx,
                "prefill_success": bool(prefill.get("success")) if isinstance(prefill, dict) else None,
                "is_listen": bool(_backend_result_value(result, "is_listen")),
                "text": _backend_result_value(result, "text", ""),
                "end_of_turn": bool(_backend_result_value(result, "end_of_turn")),
                "n_tokens": _backend_result_value(result, "n_tokens"),
                "n_tts_tokens": _backend_result_value(result, "n_tts_tokens"),
                "audio": meta,
            })
            if token_trace is not None:
                token_trace["current_unit"] = None
            if distribution_trace is not None:
                distribution_trace["current_unit"] = None

        summary = {
            "deployment_mode": args.deployment_mode,
            "call_path": args.call_path,
            "engine": engine,
            "model_path": args.model_path,
            "ckpt_path": args.ckpt_path,
            "backbone_dir": args.backbone_dir,
            "video": args.video,
            "prompt_wav": args.prompt_wav,
            "seed": args.seed,
            "batch_vision_feed": args.batch_vision_feed,
            "o5_llm_cache": args.o5_llm_cache,
            "vocoder_state": vocoder_state,
            "token_trace": "token_trace.json" if token_trace is not None else None,
            "distribution_trace": "distribution_trace.json" if distribution_trace is not None else None,
            "units": units,
        }
        if token_trace is not None:
            trace_to_write = dict(token_trace)
            trace_to_write.pop("current_unit", None)
            (out_dir / "token_trace.json").write_text(
                json.dumps(trace_to_write, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if distribution_trace is not None:
            trace_to_write = dict(distribution_trace)
            trace_to_write.pop("current_unit", None)
            (out_dir / "distribution_trace.json").write_text(
                json.dumps(trace_to_write, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "canonical_units.json").write_text(json.dumps(units, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({
            "out_dir": str(out_dir),
            "units": len(units),
            "text": "".join(u["text"] for u in units),
            "engine": summary["engine"],
        }, ensure_ascii=False), flush=True)
        return 0
    finally:
        if original_multinomial is not None:
            torch.multinomial = original_multinomial
        _shutdown_backend(backend)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deployment-mode",
        choices=("single_eager", "single_opt", "tp2", "tp2_llm"),
        default="tp2_llm",
    )
    parser.add_argument("--call-path", choices=("model", "backend"), default="model")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--backbone-dir", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--prompt-wav", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-ms", type=int, default=1000)
    parser.add_argument("--max-units", type=int, default=8)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--decode-mode", default="greedy")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--listen-prob-scale", type=float, default=1.0)
    parser.add_argument("--text-repetition-penalty", type=float, default=1.05)
    parser.add_argument("--text-repetition-window-size", type=int, default=512)
    parser.add_argument("--length-penalty", type=float, default=1.1)
    parser.add_argument("--force-listen-count", type=int, default=0)
    parser.add_argument("--max-new-speak-tokens-per-chunk", type=int, default=20)
    parser.add_argument("--n-timesteps", type=int, default=5)
    parser.add_argument("--tts-argmax", action="store_true")
    parser.add_argument("--trace-token2wav", action="store_true")
    parser.add_argument("--trace-distribution", action="store_true")
    parser.add_argument(
        "--experts-implementation",
        choices=("eager", "batched_mm", "grouped_mm", "hybrid"),
        default="eager",
    )
    parser.add_argument("--grouped-prefill-min-tokens", type=int, default=100)
    parser.add_argument("--strategy-hd", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--strategy-hd-max-slice-nums", type=int, default=4)
    parser.add_argument("--o5-llm-cache", type=int, default=32768)
    parser.add_argument("--llm-graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tts-graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vocoder-graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tts-fast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lmhead", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fuse-vision-audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-vision-feed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--unit-prefill-batch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="merge the complete multimodal input unit into one LLM prefill",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    _apply_safe_engine_env(args)
    if args.tts_argmax and not args.trace_distribution:
        _install_global_argmax_multinomial()

    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.overwrite:
        import shutil

        shutil.rmtree(out_dir)
    if int(os.environ.get("LOCAL_RANK", "0")) == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    if args.call_path == "backend":
        return _run_backend_probe(args, out_dir)

    configure_seed(args.seed)
    build = _build_tp2_model(args)
    model = build.model
    if not build.is_driver:
        _run_worker_loop(model)

    frames, audio = extract_video(Path(args.video), out_dir / "media", args.chunk_ms)
    audio_units = int(np.ceil(len(audio) / float(int(INPUT_SAMPLE_RATE * args.chunk_ms / 1000))))
    num_units = max(len(frames), audio_units)
    if args.max_units > 0:
        num_units = min(num_units, args.max_units)
    chunks = split_audio(audio, num_units, args.chunk_ms)

    token_trace = install_token_trace(model.duplex, model) if args.trace_token2wav else None
    distribution_trace, original_multinomial = (
        install_distribution_trace(model.duplex, model, out_dir)
        if args.trace_distribution
        else (None, None)
    )
    vocoder_state = reset_vocoder_static_state(model, args.seed)
    ref_audio = load_ref(Path(args.prompt_wav))

    configure_seed(args.seed)
    model.duplex_prepare(
        prefix_system_prompt="<|im_start|>system\nStreaming Omni Conversation.\n<|audio_start|>",
        suffix_system_prompt="<|audio_end|><|im_end|>",
        ref_audio=ref_audio,
        prompt_wav_path=args.prompt_wav,
        llm_seed=args.seed,
    )

    units = []
    try:
        for idx in range(num_units):
            if token_trace is not None:
                token_trace["current_unit"] = idx
            if distribution_trace is not None:
                distribution_trace["current_unit"] = idx
            frame = Image.open(frames[min(idx, len(frames) - 1)]).convert("RGB") if frames else None
            frame_list = [frame] if frame is not None else None
            prefill = model.duplex_prefill(
                audio_waveform=chunks[idx],
                frame_list=frame_list,
                max_slice_nums=None if args.strategy_hd else 1,
                batch_vision_feed=args.batch_vision_feed,
            )
            result = generate_with_optional_argmax(
                lambda: model.duplex_generate(
                    decode_mode=args.decode_mode,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    listen_prob_scale=args.listen_prob_scale,
                    text_repetition_penalty=args.text_repetition_penalty,
                    text_repetition_window_size=args.text_repetition_window_size,
                ),
                args.tts_argmax,
            )
            model.duplex_finalize()
            wav = result.get("audio_waveform")
            meta = audio_meta(wav)
            if wav is not None and meta["samples"] > 0:
                sf.write(out_dir / f"unit_{idx:03d}.wav", np.asarray(wav, dtype=np.float32), OUTPUT_SAMPLE_RATE)
            units.append({
                "unit_id": idx,
                "prefill_success": bool(prefill.get("success")) if isinstance(prefill, dict) else None,
                "effective_max_slice_nums": (
                    prefill.get("effective_max_slice_nums") if isinstance(prefill, dict) else None
                ),
                "is_listen": bool(result.get("is_listen")),
                "text": result.get("text", ""),
                "end_of_turn": bool(result.get("end_of_turn")),
                "n_tokens": result.get("n_tokens"),
                "n_tts_tokens": result.get("n_tts_tokens"),
                "audio": meta,
            })
            if token_trace is not None:
                token_trace["current_unit"] = None
            if distribution_trace is not None:
                distribution_trace["current_unit"] = None

        summary = {
            "deployment_mode": args.deployment_mode,
            "engine": build.engine,
            "model_path": args.model_path,
            "ckpt_path": args.ckpt_path,
            "backbone_dir": args.backbone_dir,
            "video": args.video,
            "prompt_wav": args.prompt_wav,
            "seed": args.seed,
            "batch_vision_feed": args.batch_vision_feed,
            "o5_llm_cache": args.o5_llm_cache,
            "vocoder_state": vocoder_state,
            "token_trace": "token_trace.json" if token_trace is not None else None,
            "distribution_trace": "distribution_trace.json" if distribution_trace is not None else None,
            "units": units,
        }
        if token_trace is not None:
            trace_to_write = dict(token_trace)
            trace_to_write.pop("current_unit", None)
            (out_dir / "token_trace.json").write_text(
                json.dumps(trace_to_write, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if distribution_trace is not None:
            trace_to_write = dict(distribution_trace)
            trace_to_write.pop("current_unit", None)
            (out_dir / "distribution_trace.json").write_text(
                json.dumps(trace_to_write, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "canonical_units.json").write_text(json.dumps(units, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({
            "out_dir": str(out_dir),
            "units": len(units),
            "text": "".join(u["text"] for u in units),
            "engine": build.engine,
        }, ensure_ascii=False), flush=True)
        return 0
    finally:
        if original_multinomial is not None:
            torch.multinomial = original_multinomial
        _shutdown_tp2(model)


if __name__ == "__main__":
    raise SystemExit(main())
