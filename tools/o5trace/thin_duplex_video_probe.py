#!/usr/bin/env python3
"""Shared offline video duplex probe for raw vs thin-unified paths."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import subprocess
import sys
import types
import wave
from pathlib import Path
from typing import Any, Optional

import numpy as np
import soundfile as sf
import torch
from accelerate import init_empty_weights
from PIL import Image
from transformers import AutoConfig


INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def audio_meta(waveform: Any) -> dict[str, Any]:
    arr = np.asarray(waveform if waveform is not None else np.zeros(0, dtype=np.float32), dtype=np.float32)
    return {
        "samples": int(arr.size),
        "duration_s": float(arr.size / OUTPUT_SAMPLE_RATE),
        "sha256": sha_bytes(np.ascontiguousarray(arr).tobytes()),
        "max_abs": float(np.max(np.abs(arr))) if arr.size else 0.0,
    }


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def extract_video(video: Path, out_dir: Path, chunk_ms: int) -> tuple[list[Path], np.ndarray]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    wav_path = out_dir / "input_16k.wav"
    fps = 1000.0 / float(chunk_ms)
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video), "-vf", f"fps={fps}", "-q:v", "2",
        str(frame_dir / "unit_%06d.jpg"),
    ])
    run([
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video), "-vn", "-ac", "1", "-ar", str(INPUT_SAMPLE_RATE),
        "-sample_fmt", "s16", str(wav_path),
    ])
    return sorted(frame_dir.glob("unit_*.jpg")), read_wav(wav_path)


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        raw = wf.readframes(wf.getnframes())
    if sample_width != 2:
        raise ValueError(f"expected int16 wav, got sample_width={sample_width}")
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio.astype(np.float32, copy=False)


def split_audio(audio: np.ndarray, num_units: int, chunk_ms: int) -> list[np.ndarray]:
    chunk_len = int(INPUT_SAMPLE_RATE * chunk_ms / 1000)
    chunks = []
    for idx in range(num_units):
        start = idx * chunk_len
        chunk = audio[start : start + chunk_len]
        if len(chunk) < chunk_len:
            chunk = np.pad(chunk, (0, chunk_len - len(chunk)))
        chunks.append(chunk.astype(np.float32, copy=False))
    return chunks


def load_ref(path: Path) -> np.ndarray:
    import librosa

    wav, _ = librosa.load(path, sr=INPUT_SAMPLE_RATE, mono=True)
    return wav.astype(np.float32, copy=False)


def load_state_dict(path: Path) -> dict:
    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location="cpu")
    for key in ("state_dict", "model", "module"):
        if isinstance(state_dict, dict) and isinstance(state_dict.get(key), dict):
            return state_dict[key]
    return state_dict


def import_raw_modeling_from_flat_code():
    for entry in sys.path:
        if not entry:
            continue
        root = Path(entry)
        if (root / "modeling_minicpmo.py").exists() and (root / "configuration_minicpmo.py").exists():
            package = types.ModuleType("MiniCPMO45")
            package.__path__ = [str(root)]
            sys.modules["MiniCPMO45"] = package
            modeling = importlib.import_module("MiniCPMO45.modeling_minicpmo")
            processing = importlib.import_module("MiniCPMO45.processing_minicpmo")
            return modeling.MiniCPMO, modeling.MiniCPMODuplex, processing.MiniCPMOProcessor
    raise ModuleNotFoundError("Cannot find flat MiniCPMO trusted-code root on sys.path")


def configure_seed(seed: int) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")


def reset_vocoder_static_state(model: Any, seed: int) -> Optional[dict[str, Any]]:
    audio_tokenizer = getattr(getattr(model, "tts", None), "audio_tokenizer", None)
    flow = getattr(audio_tokenizer, "flow", None)
    decoder = getattr(flow, "decoder", None)
    rand_noise = getattr(decoder, "rand_noise", None)
    if decoder is None or not torch.is_tensor(rand_noise):
        return None
    devices = [rand_noise.device] if rand_noise.is_cuda else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        rand_noise.copy_(torch.randn_like(rand_noise))
    for obj in (decoder, getattr(decoder, "estimator", None)):
        if obj is None:
            continue
        for attr in ("cnn_cache_buffer", "att_cache_buffer"):
            buf = getattr(obj, attr, None)
            if torch.is_tensor(buf):
                buf.zero_()
    return {
        "rand_noise_shape": list(rand_noise.shape),
        "rand_noise_sha256": sha_bytes(rand_noise.detach().cpu().float().contiguous().numpy().tobytes()),
    }


def load_model(args: argparse.Namespace):
    if args.mode == "raw":
        try:
            from MiniCPMO45.modeling_minicpmo import MiniCPMO, MiniCPMODuplex
            from MiniCPMO45.processing_minicpmo import MiniCPMOProcessor
        except ModuleNotFoundError:
            MiniCPMO, MiniCPMODuplex, MiniCPMOProcessor = import_raw_modeling_from_flat_code()
    else:
        from MiniCPMO45.modeling_minicpmo_unified import MiniCPMO
        from MiniCPMO45.processing_minicpmo import MiniCPMOProcessor
        MiniCPMODuplex = None

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config._attn_implementation = args.attn_implementation
    config._name_or_path = args.model_path
    config.name_or_path = args.model_path

    with init_empty_weights():
        model = MiniCPMO(config)

    info = model.load_state_dict(load_state_dict(Path(args.ckpt_path)), strict=False, assign=True)
    model.bfloat16().eval().to(args.device)
    model.processor = MiniCPMOProcessor.from_pretrained(args.model_path, trust_remote_code=True)

    duplex_cfg = {
        "generate_audio": True,
        "ls_mode": "explicit",
        "max_new_speak_tokens_per_chunk": args.max_new_speak_tokens_per_chunk,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "force_listen_count": args.force_listen_count,
        "n_timesteps": args.n_timesteps,
    }
    if args.mode == "raw":
        duplex = MiniCPMODuplex.from_existing_model(model, device=args.device, **duplex_cfg)
    else:
        model.init_unified(preload_both_tts=False, duplex_config=duplex_cfg, device=args.device)
        duplex = model.duplex
    return model, duplex, {"missing": len(info.missing_keys), "unexpected": len(info.unexpected_keys)}


def argmax_multinomial(input_tensor, num_samples, replacement=False, *, generator=None, out=None):
    if num_samples != 1:
        raise RuntimeError("argmax probe only supports num_samples=1")
    result = torch.argmax(input_tensor, dim=-1, keepdim=True)
    if out is not None:
        out.copy_(result)
        return out
    return result


def generate_with_optional_argmax(fn, enabled: bool):
    if not enabled:
        return fn()
    old = torch.multinomial
    torch.multinomial = argmax_multinomial
    try:
        return fn()
    finally:
        torch.multinomial = old


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("raw", "unified"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--prompt-wav", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
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
    parser.add_argument("--force-listen-count", type=int, default=0)
    parser.add_argument("--max-new-speak-tokens-per-chunk", type=int, default=20)
    parser.add_argument("--n-timesteps", type=int, default=5)
    parser.add_argument("--tts-argmax", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists() and args.overwrite:
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configure_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))

    frames, audio = extract_video(Path(args.video), out_dir / "media", args.chunk_ms)
    audio_units = int(np.ceil(len(audio) / float(int(INPUT_SAMPLE_RATE * args.chunk_ms / 1000))))
    num_units = max(len(frames), audio_units)
    if args.max_units > 0:
        num_units = min(num_units, args.max_units)
    chunks = split_audio(audio, num_units, args.chunk_ms)

    configure_seed(args.seed)
    model, duplex, load_info = load_model(args)
    vocoder_state = reset_vocoder_static_state(model, args.seed)
    ref_audio = load_ref(Path(args.prompt_wav))

    configure_seed(args.seed)
    if args.mode == "raw":
        duplex.prepare(
            prefix_system_prompt="Streaming Omni Conversation.",
            ref_audio=ref_audio,
            prompt_wav_path=args.prompt_wav,
            llm_seed=args.seed,
        )
    else:
        model.duplex_prepare(
            prefix_system_prompt="<|im_start|>system\nStreaming Omni Conversation.\n<|audio_start|>",
            suffix_system_prompt="<|audio_end|><|im_end|>",
            ref_audio=ref_audio,
            prompt_wav_path=args.prompt_wav,
            llm_seed=args.seed,
        )

    units = []
    for idx in range(num_units):
        frame = Image.open(frames[min(idx, len(frames) - 1)]).convert("RGB") if frames else None
        frame_list = [frame] if frame is not None else None
        if args.mode == "raw":
            prefill = duplex.streaming_prefill(
                audio_waveform=chunks[idx],
                frame_list=frame_list,
                max_slice_nums=1,
                batch_vision_feed=False,
            )
            result = generate_with_optional_argmax(
                lambda: duplex.streaming_generate(
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
        else:
            prefill = model.duplex_prefill(
                audio_waveform=chunks[idx],
                frame_list=frame_list,
                max_slice_nums=1,
                batch_vision_feed=False,
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
            "is_listen": bool(result.get("is_listen")),
            "text": result.get("text", ""),
            "end_of_turn": bool(result.get("end_of_turn")),
            "n_tokens": result.get("n_tokens"),
            "n_tts_tokens": result.get("n_tts_tokens"),
            "audio": meta,
        })

    summary = {
        "mode": args.mode,
        "model_path": args.model_path,
        "ckpt_path": args.ckpt_path,
        "video": args.video,
        "prompt_wav": args.prompt_wav,
        "seed": args.seed,
        "load_info": load_info,
        "vocoder_state": vocoder_state,
        "units": units,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "canonical_units.json").write_text(json.dumps(units, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "units": len(units), "text": "".join(u["text"] for u in units)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
