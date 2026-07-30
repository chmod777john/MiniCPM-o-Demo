#!/usr/bin/env python3
"""Replay a recorded realtime video-duplex session through the canonical model.

This is intentionally narrow: it supports the basic browser/API video-duplex
case we need for alignment. The recorded session must contain exact `.f32`
audio blobs and `.jpg` frame blobs produced by the session recorder in this
worktree. Older sessions that only contain WAV files are rejected.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import shutil
import wave
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel

from core.schemas.duplex import DuplexConfig
from thin_duplex_video_probe import (
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    audio_meta,
    configure_seed,
    generate_with_optional_argmax,
    install_token_trace,
    reset_vocoder_static_state,
)


DEFAULT_CANONICAL_ROOT = Path(
    "/user/weihongliang/MiniCPM-o-4_6-modelbest-moe-35b-a3b_debug-session-replay-canonical-2026-07-26"
)
DEFAULT_CKPT_PATH = Path(
    "/user/weihongliang/o5_weights/houyueran_o5_MB_omni-sft2-8k-hyr_a2_i2_0722_iter_0002800.pt"
)


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def config_value(config: dict[str, Any], name: str, default: Any) -> Any:
    value = config.get(name)
    return default if value is None else value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{lineno}: expected JSON object")
            out.append(item)
    return out


def blob_path(session_dir: Path, rel: str) -> Path:
    if not rel.startswith("@blob/"):
        raise ValueError(f"expected @blob pointer, got: {rel!r}")
    path = (session_dir / rel[1:]).resolve()
    if not str(path).startswith(str(session_dir.resolve())):
        raise ValueError(f"blob path escapes session dir: {rel}")
    return path


def read_f32_blob(session_dir: Path, rel: str, expected_sha: Optional[str]) -> np.ndarray:
    path = blob_path(session_dir, rel)
    if path.suffix != ".f32":
        raise ValueError(f"exact replay requires .f32 audio blob, got: {path}")
    raw = path.read_bytes()
    if expected_sha and sha_bytes(raw) != expected_sha:
        raise ValueError(f"sha256 mismatch for {path}")
    return np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)


def read_image_blob(session_dir: Path, rel: str, expected_sha: Optional[str]) -> Image.Image:
    path = blob_path(session_dir, rel)
    if path.suffix.lower() not in {".jpg", ".jpeg"}:
        raise ValueError(f"expected recorded JPEG blob, got: {path}")
    raw = path.read_bytes()
    if expected_sha and sha_bytes(raw) != expected_sha:
        raise ValueError(f"sha256 mismatch for {path}")
    return Image.open(io.BytesIO(raw)).convert("RGB")


def write_wav(path: Path, waveform: np.ndarray, sample_rate: int = OUTPUT_SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.asarray(waveform, dtype=np.float32), -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2", copy=False)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())


def load_ref_audio(path: Path) -> np.ndarray:
    try:
        import librosa

        wav, _ = librosa.load(str(path), sr=INPUT_SAMPLE_RATE, mono=True)
        return wav.astype(np.float32, copy=False)
    except Exception:
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        if sample_rate != INPUT_SAMPLE_RATE:
            raise ValueError(f"{path} sample_rate={sample_rate}, expected {INPUT_SAMPLE_RATE}")
        if sample_width != 2:
            raise ValueError(f"{path} sample_width={sample_width}, expected int16 wav")
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        return audio.astype(np.float32, copy=False)


def restore_ref_audio_from_base64(init_payload: dict[str, Any], out_dir: Path) -> Optional[str]:
    value = init_payload.get("ref_audio_base64")
    if not isinstance(value, str) or not value:
        return None
    raw = base64.b64decode(value)
    ref_audio = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
    path = out_dir / "ref_audio_from_session.wav"
    write_wav(path, ref_audio, sample_rate=INPUT_SAMPLE_RATE)
    return str(path)


def session_init_payload(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in events:
        frame = event.get("frame")
        if isinstance(frame, dict) and frame.get("type") == "session.init":
            payload = frame.get("payload")
            if isinstance(payload, dict):
                return payload
    raise ValueError("recorded session has no session.init payload")


def session_replay_manifest(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in events:
        frame = event.get("frame")
        if not isinstance(frame, dict) or frame.get("type") != "session.created":
            continue
        manifest = frame.get("replay_manifest")
        if isinstance(manifest, dict):
            return manifest
    return {}


def model_dump(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return dict(obj.model_dump())
    if hasattr(obj, "dict"):
        return dict(obj.dict())
    return dict(obj or {})


def resolved_duplex_config(init_payload: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    resolved = manifest.get("resolved") if isinstance(manifest.get("resolved"), dict) else {}
    manifest_config = resolved.get("duplex_config") if isinstance(resolved, dict) else None
    if isinstance(manifest_config, dict) and manifest_config:
        return dict(manifest_config)

    config = init_payload.get("config") if isinstance(init_payload.get("config"), dict) else {}
    config = dict(config)
    if "use_tts" in init_payload:
        config["generate_audio"] = bool(init_payload.get("use_tts"))
    return model_dump(DuplexConfig(**config))


def resolved_seed(init_payload: dict[str, Any], config: dict[str, Any], manifest: dict[str, Any]) -> int:
    resolved = manifest.get("resolved") if isinstance(manifest.get("resolved"), dict) else {}
    raw_config = init_payload.get("config") if isinstance(init_payload.get("config"), dict) else {}
    for value in (
        resolved.get("llm_seed") if isinstance(resolved, dict) else None,
        resolved.get("seed") if isinstance(resolved, dict) else None,
        init_payload.get("seed"),
        raw_config.get("seed"),
        config.get("seed"),
    ):
        if value is not None:
            return int(value)
    return 0


def manifest_tts_argmax(manifest: dict[str, Any]) -> Optional[bool]:
    runtime = manifest.get("runtime") if isinstance(manifest.get("runtime"), dict) else {}
    env = runtime.get("env") if isinstance(runtime.get("env"), dict) else {}
    value = env.get("O5_TTS_ARGMAX") if isinstance(env, dict) else None
    if value is None:
        return None
    return str(value).lower() in {"1", "true", "yes", "on"}


def input_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for event in events:
        if event.get("dir") != "up":
            continue
        frame = event.get("frame")
        if isinstance(frame, dict) and frame.get("type") == "input.append":
            out.append(event)
    if not out:
        raise ValueError("recorded session has no upstream input.append events")
    return out


def event_audio(event: dict[str, Any], session_dir: Path) -> np.ndarray:
    trace = event.get("payload_trace") or {}
    meta = trace.get("audio") if isinstance(trace, dict) else None
    if not isinstance(meta, dict) or not meta.get("blob_f32"):
        raise ValueError(
            f"input event seq={event.get('seq')} has no payload_trace.audio.blob_f32; "
            "record a new session with the replay recorder"
        )
    audio = read_f32_blob(session_dir, str(meta["blob_f32"]), meta.get("sha256"))
    if int(meta.get("sample_rate", INPUT_SAMPLE_RATE)) != INPUT_SAMPLE_RATE:
        raise ValueError(f"input event seq={event.get('seq')} sample rate is not {INPUT_SAMPLE_RATE}")
    return audio


def event_frames(event: dict[str, Any], session_dir: Path) -> list[Image.Image]:
    trace = event.get("payload_trace") or {}
    frame_meta = trace.get("video_frames") if isinstance(trace, dict) else None
    if not frame_meta:
        return []
    frames: list[Image.Image] = []
    for item in frame_meta:
        if item is None:
            continue
        if not isinstance(item, dict) or not item.get("blob_jpg"):
            raise ValueError(f"input event seq={event.get('seq')} has malformed video frame trace")
        frames.append(read_image_blob(session_dir, str(item["blob_jpg"]), item.get("sha256")))
    return frames


def event_max_slice_nums(event: dict[str, Any]) -> int:
    frame = event.get("frame") if isinstance(event.get("frame"), dict) else {}
    inp = frame.get("input") if isinstance(frame, dict) and isinstance(frame.get("input"), dict) else {}
    trace = event.get("payload_trace") if isinstance(event.get("payload_trace"), dict) else {}
    value = inp.get("max_slice_nums", trace.get("max_slice_nums", 1))
    return int(value or 1)


def event_force_listen(event: dict[str, Any]) -> bool:
    frame = event.get("frame") if isinstance(event.get("frame"), dict) else {}
    inp = frame.get("input") if isinstance(frame, dict) and isinstance(frame.get("input"), dict) else {}
    trace = event.get("payload_trace") if isinstance(event.get("payload_trace"), dict) else {}
    return bool(inp.get("force_listen", trace.get("force_listen", False)))


def event_input_id(event: dict[str, Any], idx: int) -> str:
    frame = event.get("frame") if isinstance(event.get("frame"), dict) else {}
    inp = frame.get("input") if isinstance(frame, dict) and isinstance(frame.get("input"), dict) else {}
    return str(inp.get("input_id") or f"unit_{idx:03d}")


def load_canonical_duplex(
    *,
    canonical_root: Path,
    ckpt_path: Path,
    tmp_model_dir: Path,
    token2wav_dir: Path,
    device: str,
    duplex_kwargs: dict[str, Any],
    attn_implementation: str,
):
    canonical_root = canonical_root.resolve()
    ckpt_path = ckpt_path.resolve()
    tmp_model_dir = tmp_model_dir.resolve()
    if not (canonical_root / "config.json").is_file():
        raise FileNotFoundError(f"config.json not found in {canonical_root}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    if tmp_model_dir.exists():
        shutil.rmtree(tmp_model_dir)
    tmp_model_dir.mkdir(parents=True)
    for src in canonical_root.iterdir():
        if src.is_dir() or src.name.startswith(".") or src.name == "pytorch_model.bin":
            continue
        (tmp_model_dir / src.name).symlink_to(src)
    (tmp_model_dir / "pytorch_model.bin").symlink_to(ckpt_path)

    model = AutoModel.from_pretrained(
        str(tmp_model_dir),
        trust_remote_code=True,
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
    )
    model.eval().to(device)
    original_init_tts = model.init_tts

    def init_tts_with_recorded_assets(model_dir=None, *args, **kwargs):
        return original_init_tts(model_dir=model_dir or str(token2wav_dir), *args, **kwargs)

    model.init_tts = init_tts_with_recorded_assets
    try:
        duplex = model.as_duplex(
            device=device,
            **duplex_kwargs,
        )
    finally:
        model.init_tts = original_init_tts
    return model, duplex


def generate_with_force_listen(duplex: Any, force_listen: bool, kwargs: dict[str, Any]) -> dict[str, Any]:
    if not force_listen:
        return duplex.streaming_generate(**kwargs)

    old_force = getattr(duplex, "force_listen_count", 0)
    old_count = getattr(duplex, "_streaming_generate_count", 0)
    duplex.force_listen_count = old_count + 1
    try:
        return duplex.streaming_generate(**kwargs)
    finally:
        duplex.force_listen_count = old_force


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True, help="data/sessions/sess_xxx directory")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--canonical-root", default=str(DEFAULT_CANONICAL_ROOT))
    parser.add_argument("--ckpt-path", default=str(DEFAULT_CKPT_PATH))
    parser.add_argument("--tmp-model-dir", default=None)
    parser.add_argument("--token2wav-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--max-units", type=int, default=0)
    parser.add_argument("--first-chunk-ms", type=int, default=1035)
    parser.add_argument("--cnn-redundancy-ms", type=int, default=20)
    parser.add_argument("--trace-token2wav", action="store_true")
    parser.add_argument(
        "--tts-argmax",
        choices=("auto", "0", "1"),
        default="auto",
        help="Replay TTS sampling as argmax; auto follows session replay_manifest when available.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    session_dir = Path(args.session_dir).resolve()
    stream_path = session_dir / "stream.jsonl"
    if not stream_path.is_file():
        raise FileNotFoundError(f"stream.jsonl not found: {stream_path}")

    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = read_jsonl(stream_path)
    init_payload = session_init_payload(events)
    manifest = session_replay_manifest(events)
    config = resolved_duplex_config(init_payload, manifest)
    seed = resolved_seed(init_payload, config, manifest)
    chunk_ms = int(config.get("chunk_ms", 1000) or 1000)
    system_prompt = str(init_payload.get("system_prompt") or "Streaming Omni Conversation.")
    ref_audio_path = init_payload.get("ref_audio_path")
    if not ref_audio_path:
        ref_audio_path = restore_ref_audio_from_base64(init_payload, out_dir)
    if not ref_audio_path:
        raise ValueError(
            "basic canonical replay requires session.init.payload.ref_audio_path "
            "or session.init.payload.ref_audio_base64"
        )
    ref_audio_path = str(ref_audio_path)

    canonical_root = Path(args.canonical_root)
    token2wav_dir = Path(args.token2wav_dir) if args.token2wav_dir else canonical_root / "assets" / "token2wav"
    tmp_model_dir = Path(args.tmp_model_dir) if args.tmp_model_dir else out_dir / "_tmp_hf_model"
    unit_events = input_events(events)
    if args.max_units > 0:
        unit_events = unit_events[: args.max_units]

    duplex_kwargs = {
        "generate_audio": bool(config_value(config, "generate_audio", True)),
        "max_new_speak_tokens_per_chunk": int(config_value(config, "max_new_speak_tokens_per_chunk", 20)),
        "text_repetition_penalty": float(config_value(config, "text_repetition_penalty", 1.05)),
        "temperature": float(config_value(config, "temperature", 0.7)),
        "top_k": int(config_value(config, "top_k", 100)),
        "top_p": float(config_value(config, "top_p", 0.8)),
        "text_repetition_window_size": int(config_value(config, "text_repetition_window_size", 512)),
        "listen_prob_scale": float(config_value(config, "listen_prob_scale", 1.0)),
        "force_listen_count": int(config_value(config, "force_listen_count", 0)),
        "tts_temperature": float(config_value(config, "tts_temperature", 0.8)),
        "tts_repetition_penalty": float(config_value(config, "tts_repetition_penalty", 1.05)),
        "chunk_ms": chunk_ms,
        "first_chunk_ms": args.first_chunk_ms,
        "cnn_redundancy_ms": args.cnn_redundancy_ms,
        "sample_rate": int(config_value(config, "sample_rate", INPUT_SAMPLE_RATE)),
    }

    configure_seed(seed)
    if args.device.startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    model, duplex = load_canonical_duplex(
        canonical_root=canonical_root,
        ckpt_path=Path(args.ckpt_path),
        tmp_model_dir=tmp_model_dir,
        token2wav_dir=token2wav_dir,
        device=args.device,
        duplex_kwargs=duplex_kwargs,
        attn_implementation=args.attn_implementation,
    )
    token_trace = install_token_trace(duplex, model) if args.trace_token2wav else None
    vocoder_state = reset_vocoder_static_state(model, seed)
    ref_audio = load_ref_audio(Path(ref_audio_path))

    configure_seed(seed)
    duplex.prepare(
        prefix_system_prompt=system_prompt,
        ref_audio=ref_audio,
        prompt_wav_path=ref_audio_path,
        llm_seed=seed,
    )

    generate_kwargs = {
        "prompt_wav_path": ref_audio_path,
        "max_new_speak_tokens_per_chunk": int(config_value(config, "max_new_speak_tokens_per_chunk", 20)),
        "decode_mode": str(config_value(config, "decode_mode", "sampling")),
        "temperature": float(config_value(config, "temperature", 0.7)),
        "top_k": int(config_value(config, "top_k", 20)),
        "top_p": float(config_value(config, "top_p", 0.8)),
        "listen_prob_scale": float(config_value(config, "listen_prob_scale", 1.0)),
        "listen_top_k": config.get("listen_top_k"),
        "text_repetition_penalty": float(config_value(config, "text_repetition_penalty", 1.05)),
        "text_repetition_window_size": int(config_value(config, "text_repetition_window_size", 512)),
    }
    force_listen_count = int(config.get("force_listen_count", 0) or 0)
    tts_argmax = manifest_tts_argmax(manifest)
    if args.tts_argmax != "auto":
        tts_argmax = args.tts_argmax == "1"
    if tts_argmax is None:
        tts_argmax = False

    units: list[dict[str, Any]] = []
    audio_parts: list[np.ndarray] = []
    for idx, event in enumerate(unit_events):
        if token_trace is not None:
            token_trace["current_unit"] = event_input_id(event, idx)
        audio = event_audio(event, session_dir)
        frames = event_frames(event, session_dir)
        prefill = duplex.streaming_prefill(
            audio_waveform=audio,
            frame_list=frames or None,
            max_slice_nums=event_max_slice_nums(event),
        )
        force_listen = event_force_listen(event) or idx < force_listen_count
        result = generate_with_optional_argmax(
            lambda: generate_with_force_listen(duplex, force_listen, generate_kwargs),
            bool(tts_argmax),
        )
        waveform = result.get("audio_waveform")
        meta = audio_meta(waveform)
        if waveform is not None and meta["samples"] > 0:
            part = np.asarray(waveform, dtype=np.float32)
            audio_parts.append(part)
            write_wav(out_dir / f"unit_{idx:03d}.wav", part)
        units.append({
            "unit_id": idx,
            "input_id": event_input_id(event, idx),
            "prefill_success": bool(prefill.get("success")) if isinstance(prefill, dict) else None,
            "is_listen": bool(result.get("is_listen")),
            "text": result.get("text", ""),
            "end_of_turn": bool(result.get("end_of_turn")),
            "n_tokens": result.get("n_tokens"),
            "n_tts_tokens": result.get("n_tts_tokens"),
            "audio": meta,
        })
        if token_trace is not None:
            token_trace["current_unit"] = None

    summary = {
        "session_dir": str(session_dir),
        "canonical_root": str(canonical_root),
        "ckpt_path": str(Path(args.ckpt_path).resolve()),
        "token2wav_dir": str(token2wav_dir),
        "seed": seed,
        "chunk_ms": chunk_ms,
        "system_prompt": system_prompt,
        "replay_manifest": manifest or None,
        "resolved_duplex_config": config,
        "ref_audio_path": ref_audio_path,
        "duplex_kwargs": duplex_kwargs,
        "generate_kwargs": generate_kwargs,
        "force_listen_count": force_listen_count,
        "tts_argmax": bool(tts_argmax),
        "vocoder_state": vocoder_state,
        "token_trace": "token_trace.json" if token_trace is not None else None,
        "continuous_audio": "continuous.wav" if audio_parts else None,
        "units": units,
    }
    if audio_parts:
        write_wav(out_dir / "continuous.wav", np.concatenate(audio_parts))
    if token_trace is not None:
        trace_to_write = dict(token_trace)
        trace_to_write.pop("current_unit", None)
        (out_dir / "token_trace.json").write_text(
            json.dumps(trace_to_write, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (out_dir / "canonical_units.json").write_text(
        json.dumps(units, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "out_dir": str(out_dir),
        "units": len(units),
        "text": "".join(str(unit.get("text") or "") for unit in units),
        "token_trace": summary["token_trace"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
