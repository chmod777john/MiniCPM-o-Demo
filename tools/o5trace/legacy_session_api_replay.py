#!/usr/bin/env python3
"""Replay an old recorded realtime session through a live realtime API.

Old sessions store upstream media directly in `frame.input` as @blob WAV/JPEG
pointers. This script converts those inputs to the current realtime API wire
format and records the live API response.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import ssl
import time
import wave
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import numpy as np
import websockets


INPUT_SAMPLE_RATE = 16000


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def blob_path(session_dir: Path, rel: str) -> Path:
    if not isinstance(rel, str) or not rel.startswith("@blob/"):
        raise ValueError(f"expected @blob pointer, got: {rel!r}")
    path = (session_dir / rel[1:]).resolve()
    if not str(path).startswith(str(session_dir.resolve())):
        raise ValueError(f"blob path escapes session dir: {rel}")
    return path


def session_init_payload(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in events:
        frame = event.get("frame")
        if isinstance(frame, dict) and frame.get("type") == "session.init":
            payload = frame.get("payload")
            if isinstance(payload, dict):
                return dict(payload)
    raise ValueError("recorded session has no session.init payload")


def input_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for event in events:
        if event.get("dir") != "up":
            continue
        frame = event.get("frame")
        if isinstance(frame, dict) and frame.get("type") == "input.append":
            out.append(event)
    if not out:
        raise ValueError("recorded session has no input.append events")
    return out


def load_wav_float32(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if sample_rate != INPUT_SAMPLE_RATE:
        raise ValueError(f"{path} sample_rate={sample_rate}, expected {INPUT_SAMPLE_RATE}")
    if width != 2:
        raise ValueError(f"{path} sample_width={width}, expected int16")
    audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio.astype(np.float32, copy=False)


def b64_float32(audio: np.ndarray) -> str:
    return base64.b64encode(np.asarray(audio, dtype="<f4").tobytes()).decode("ascii")


def b64_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def response_audio_bytes(msg: dict[str, Any], session_dir: Path) -> Optional[bytes]:
    value = msg.get("audio")
    if not isinstance(value, str) or not value:
        return None
    if value.startswith("@blob/"):
        return blob_path(session_dir, value).read_bytes()
    return base64.b64decode(value)


def normalize_url(url: str, mode: str) -> str:
    if url.startswith("ws://") or url.startswith("wss://"):
        return url
    scheme = "wss" if url.startswith("https://") else "ws"
    return f"{scheme}://{url.split('://', 1)[-1].rstrip('/')}/v1/realtime?mode={mode}"


async def run(args: argparse.Namespace) -> dict[str, Any]:
    session_dir = Path(args.session_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    events = read_jsonl(session_dir / "stream.jsonl")
    init_payload = session_init_payload(events)
    units = input_events(events)
    if args.max_units > 0:
        units = units[: args.max_units]

    # Preserve user-visible session settings but let the target service apply
    # its own deterministic/default replay policy if configured.
    payload = dict(init_payload)
    payload.setdefault("system_prompt", "Streaming Omni Conversation.")
    payload.setdefault("use_tts", True)

    url = normalize_url(args.url, args.mode)
    parsed = urlparse(url)
    ssl_ctx = None
    if parsed.scheme == "wss":
        ssl_ctx = ssl._create_unverified_context() if args.insecure else ssl.create_default_context()

    stream_path = out_dir / "api_stream.jsonl"
    text_parts: list[str] = []
    audio_parts: list[bytes] = []
    audio_count = 0
    started = time.perf_counter()
    session_id = None

    ws = await websockets.connect(
        url,
        ssl=ssl_ctx,
        open_timeout=args.open_timeout,
        max_size=args.max_message_mb * 1024 * 1024,
    )
    try:
        with stream_path.open("w", encoding="utf-8") as stream:
            async def record(direction: str, frame: dict[str, Any]) -> None:
                stream.write(json.dumps({"ts": time.time(), "dir": direction, "frame": frame}, ensure_ascii=False) + "\n")
                stream.flush()

            send_done = False
            last_server_msg = time.perf_counter()
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=args.event_timeout_s)
                except asyncio.TimeoutError:
                    if send_done:
                        break
                    raise
                msg = json.loads(raw)
                last_server_msg = time.perf_counter()
                await record("down", msg)

                msg_type = msg.get("type")
                if msg_type in ("session.queue_done", "queue_done"):
                    frame = {"type": "session.init", "payload": payload}
                    await ws.send(json.dumps(frame, ensure_ascii=False))
                    await record("up", frame)
                elif msg_type == "session.created":
                    session_id = msg.get("session_id")
                    break

            for idx, event in enumerate(units):
                frame = event.get("frame") if isinstance(event.get("frame"), dict) else {}
                old_input = frame.get("input") if isinstance(frame.get("input"), dict) else {}
                audio = load_wav_float32(blob_path(session_dir, old_input.get("audio")))
                frames = [b64_file(blob_path(session_dir, item)) for item in (old_input.get("video_frames") or []) if item]
                new_input = {
                    "input_id": str(old_input.get("input_id") or f"legacy_{idx:04d}"),
                    "audio": b64_float32(audio),
                    "video_frames": frames,
                    "force_listen": bool(old_input.get("force_listen", False)),
                    "max_slice_nums": int(old_input.get("max_slice_nums", payload.get("max_slice_nums", 1)) or 1),
                }
                up = {"type": "input.append", "input": new_input}
                await ws.send(json.dumps(up))
                await record("up", {"type": "input.append", "input": {
                    "input_id": new_input["input_id"],
                    "audio_samples": int(audio.size),
                    "video_frames": len(frames),
                    "force_listen": new_input["force_listen"],
                    "max_slice_nums": new_input["max_slice_nums"],
                }})

                deadline = time.perf_counter() + args.unit_timeout_s
                while True:
                    timeout = max(0.1, deadline - time.perf_counter())
                    if timeout <= 0:
                        raise TimeoutError(f"timeout waiting output for unit {idx}")
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
                    msg = json.loads(raw)
                    await record("down", msg)
                    last_server_msg = time.perf_counter()
                    if msg.get("type") == "response.output.delta":
                        if msg.get("kind") == "text":
                            text_parts.append(msg.get("text") or "")
                        elif msg.get("kind") == "audio":
                            data = response_audio_bytes(msg, session_dir)
                            if data:
                                audio_parts.append(data)
                                audio_count += 1
                        # One response event is emitted per input unit in current full-duplex API.
                        if msg.get("kind") in {"listen", "text", "audio"}:
                            break
                if args.pace_s > 0:
                    elapsed_target = (idx + 1) * args.pace_s
                    sleep_s = started + elapsed_target - time.perf_counter()
                    if sleep_s > 0:
                        await asyncio.sleep(sleep_s)

            send_done = True
            while time.perf_counter() - last_server_msg < args.drain_s:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=args.drain_s)
                except asyncio.TimeoutError:
                    break
                msg = json.loads(raw)
                await record("down", msg)
                last_server_msg = time.perf_counter()
                if msg.get("type") == "response.output.delta":
                    if msg.get("kind") == "text":
                        text_parts.append(msg.get("text") or "")
                    elif msg.get("kind") == "audio":
                        data = response_audio_bytes(msg, session_dir)
                        if data:
                            audio_parts.append(data)
                            audio_count += 1
    finally:
        await ws.close()

    if audio_parts:
        # Full-duplex API returns 24kHz float32 PCM. Some older turn-based
        # paths returned WAV bytes, so keep both forms readable here.
        import io

        params0 = None
        raw_chunks: list[bytes] = []
        for data in audio_parts:
            if data.startswith(b"RIFF"):
                with wave.open(io.BytesIO(data), "rb") as wf:
                    params = wf.getparams()
                    raw = wf.readframes(wf.getnframes())
                if params0 is None:
                    params0 = params
                if (params.nchannels, params.sampwidth, params.framerate, params.comptype) != (
                    params0.nchannels,
                    params0.sampwidth,
                    params0.framerate,
                    params0.comptype,
                ):
                    raise ValueError(f"response audio params mismatch: {params} != {params0}")
                raw_chunks.append(raw)
            else:
                if len(data) % 4 != 0:
                    raise ValueError(f"float32 audio bytes length is not divisible by 4: {len(data)}")
                if params0 is None:
                    params0 = wave._wave_params(1, 4, 24000, 0, "NONE", "not compressed")
                raw_chunks.append(data)
        with wave.open(str(out_dir / "response_audio_concat.wav"), "wb") as wf:
            wf.setnchannels(params0.nchannels)
            wf.setsampwidth(params0.sampwidth)
            wf.setframerate(params0.framerate)
            wf.writeframes(b"".join(raw_chunks))

    summary = {
        "url": url,
        "source_session_dir": str(session_dir),
        "api_session_id": session_id,
        "units_sent": len(units),
        "output_audio_chunks": audio_count,
        "text": "".join(text_parts),
        "stream": "api_stream.jsonl",
        "response_audio": "response_audio_concat.wav" if audio_parts else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--url", default="https://82.157.64.212:8040")
    parser.add_argument("--mode", default="video", choices=("video", "audio"))
    parser.add_argument("--max-units", type=int, default=0)
    parser.add_argument("--pace-s", type=float, default=0.0)
    parser.add_argument("--drain-s", type=float, default=3.0)
    parser.add_argument("--unit-timeout-s", type=float, default=180.0)
    parser.add_argument("--event-timeout-s", type=float, default=180.0)
    parser.add_argument("--open-timeout", type=float, default=30.0)
    parser.add_argument("--max-message-mb", type=int, default=32)
    parser.add_argument("--insecure", action="store_true")
    return parser.parse_args()


def main() -> int:
    summary = asyncio.run(run(parse_args()))
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
