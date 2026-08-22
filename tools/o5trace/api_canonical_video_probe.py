#!/usr/bin/env python3
"""Realtime API video probe with the same media/unit contract as canonical offline probes."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import ssl
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import websockets

from thin_duplex_video_probe import (
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    audio_meta,
    extract_video,
    split_audio,
)


def now() -> float:
    return time.perf_counter()


def log(message: str) -> None:
    print(f"[api-probe] {message}", flush=True)


async def recv_json(ws: websockets.ClientConnection, *, timeout_s: float) -> dict[str, Any]:
    try:
        raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
    except asyncio.TimeoutError as exc:
        raise TimeoutError(f"timed out waiting for websocket event after {timeout_s:.1f}s") from exc
    msg = json.loads(raw)
    if not isinstance(msg, dict):
        raise RuntimeError("websocket event must be a JSON object")
    return msg


def normalize_realtime_url(url: str, mode: str = "video") -> str:
    parsed = urlparse(url)
    if parsed.scheme in ("ws", "wss") and parsed.path:
        return url
    if parsed.scheme not in ("http", "https", "ws", "wss"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
    scheme = "wss" if parsed.scheme in ("https", "wss") else "ws"
    host = parsed.netloc or parsed.path
    return f"{scheme}://{host.rstrip('/')}/v1/realtime?mode={mode}"


def b64_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def b64_float32(samples: np.ndarray) -> str:
    return base64.b64encode(samples.astype("<f4", copy=False).tobytes()).decode("ascii")


def decode_audio_b64(audio_b64: str) -> np.ndarray:
    raw = base64.b64decode(audio_b64)
    return np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=False)


def write_wav(path: Path, waveform: np.ndarray) -> None:
    try:
        import soundfile as sf

        sf.write(path, np.asarray(waveform, dtype=np.float32), OUTPUT_SAMPLE_RATE)
        return
    except Exception:
        pass

    import wave

    pcm = np.clip(waveform, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2", copy=False)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(OUTPUT_SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())


def duplex_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {
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
        "chunk_ms": args.chunk_ms,
        "sample_rate": INPUT_SAMPLE_RATE,
        "strategy_hd": args.strategy_hd,
        "strategy_hd_max_slice_nums": args.strategy_hd_max_slice_nums,
    }
    if args.listen_top_k is not None:
        config["listen_top_k"] = args.listen_top_k
    return config


async def wait_for_queue(ws: websockets.ClientConnection, *, timeout_s: float) -> None:
    while True:
        msg = await recv_json(ws, timeout_s=timeout_s)
        typ = msg.get("type")
        log(f"queue event type={typ}")
        if typ in {"session.queue_done", "queue_done"}:
            return
        if typ == "error":
            raise RuntimeError(json.dumps(msg, ensure_ascii=False))


async def init_session(ws: websockets.ClientConnection, args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "system_prompt": args.system_prompt,
        "config": duplex_config(args),
        "seed": args.seed,
    }
    if args.prompt_wav:
        payload["ref_audio_path"] = args.prompt_wav
    log("send session.init")
    await ws.send(json.dumps({"type": "session.init", "payload": payload}, ensure_ascii=False))
    while True:
        msg = await recv_json(ws, timeout_s=args.event_timeout_s)
        typ = msg.get("type")
        log(f"init event type={typ}")
        if typ == "session.created":
            return msg
        if typ == "error":
            raise RuntimeError(json.dumps(msg, ensure_ascii=False))
        if typ == "session.closed":
            raise RuntimeError(json.dumps(msg, ensure_ascii=False))


async def drain_after_audio(
    ws: websockets.ClientConnection,
    *,
    input_id: str,
    timeout_s: float,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    while timeout_s > 0:
        t0 = now()
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return events
        timeout_s -= now() - t0
        msg = json.loads(raw)
        events.append(msg)
        if msg.get("type") == "response.output.delta" and msg.get("input_id") == input_id and msg.get("kind") == "listen":
            return events
        if msg.get("type") in {"error", "session.closed"}:
            return events
    return events


async def close_session(
    ws: websockets.ClientConnection,
    *,
    timeout_s: float,
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    await ws.send(json.dumps({"type": "session.close", "reason": "probe_done"}))
    while True:
        msg = await recv_json(ws, timeout_s=timeout_s)
        events.append(msg)
        typ = msg.get("type")
        if typ == "session.closed":
            return events
        if typ == "error":
            raise RuntimeError(json.dumps(msg, ensure_ascii=False))


async def collect_unit(
    ws: websockets.ClientConnection,
    *,
    input_id: str,
    listen_silence_samples: int,
    post_audio_drain_s: float,
    event_timeout_s: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    text_parts: list[str] = []
    audio_parts: list[np.ndarray] = []
    saw_listen = False
    metrics: dict[str, Any] = {}
    events: list[dict[str, Any]] = []

    while True:
        msg = await recv_json(ws, timeout_s=event_timeout_s)
        events.append(msg)
        typ = msg.get("type")
        if typ == "error":
            raise RuntimeError(json.dumps(msg, ensure_ascii=False))
        if typ == "session.closed":
            raise RuntimeError(json.dumps(msg, ensure_ascii=False))
        if typ != "response.output.delta" or msg.get("input_id") != input_id:
            log(f"unit={input_id} skip event type={typ} input_id={msg.get('input_id')}")
            continue

        metrics.update(msg.get("metrics") or {})
        kind = msg.get("kind")
        log(f"unit={input_id} event kind={kind}")
        if kind == "text":
            text_parts.append(str(msg.get("text") or ""))
            extra = await drain_after_audio(ws, input_id=input_id, timeout_s=post_audio_drain_s)
            events.extend(extra)
            for item in extra:
                if item.get("type") != "response.output.delta" or item.get("input_id") != input_id:
                    continue
                metrics.update(item.get("metrics") or {})
                extra_kind = item.get("kind")
                log(f"unit={input_id} followup kind={extra_kind}")
                if extra_kind == "text":
                    text_parts.append(str(item.get("text") or ""))
                elif extra_kind == "audio":
                    audio_b64 = item.get("audio")
                    if audio_b64:
                        audio_parts.append(decode_audio_b64(audio_b64))
                elif extra_kind == "listen":
                    saw_listen = True
            break
        if kind == "audio":
            audio_b64 = msg.get("audio")
            if audio_b64:
                audio_parts.append(decode_audio_b64(audio_b64))
            extra = await drain_after_audio(ws, input_id=input_id, timeout_s=post_audio_drain_s)
            events.extend(extra)
            for item in extra:
                if item.get("type") == "response.output.delta" and item.get("input_id") == input_id:
                    metrics.update(item.get("metrics") or {})
                    if item.get("kind") == "listen":
                        saw_listen = True
            break
        if kind == "listen":
            saw_listen = True
            break

    audio = np.concatenate(audio_parts) if audio_parts else None
    text = "".join(text_parts)
    is_listen = bool(saw_listen and not text and audio is None)
    if is_listen and listen_silence_samples > 0:
        audio = np.zeros(listen_silence_samples, dtype=np.float32)
    unit = {
        "input_id": input_id,
        "is_listen": is_listen,
        "text": text,
        "end_of_turn": bool(saw_listen and not is_listen),
        "n_tokens": metrics.get("n_tokens"),
        "n_tts_tokens": metrics.get("n_tts_tokens"),
        "audio": audio_meta(audio),
        "metrics": metrics,
    }
    log(
        "unit=%s done is_listen=%s text_len=%d audio_samples=%d"
        % (input_id, unit["is_listen"], len(text), int(unit["audio"].get("samples", 0)))
    )
    return unit, events


async def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    url = normalize_realtime_url(args.url, "video")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames, audio = extract_video(Path(args.video), out_dir / "media", args.chunk_ms)
    audio_units = int(np.ceil(len(audio) / float(int(INPUT_SAMPLE_RATE * args.chunk_ms / 1000))))
    num_units = max(len(frames), audio_units)
    if args.max_units > 0:
        num_units = min(num_units, args.max_units)
    chunks = split_audio(audio, num_units, args.chunk_ms)
    log(f"media ready frames={len(frames)} audio_units={audio_units} run_units={num_units}")

    frame_b64 = [b64_file(path) for path in frames]
    ssl_ctx = ssl._create_unverified_context() if args.insecure else ssl.create_default_context()
    units: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    session_created: dict[str, Any] | None = None

    log(f"connect {url}")
    ws = await websockets.connect(
        url,
        ssl=ssl_ctx,
        open_timeout=args.open_timeout,
        max_size=args.max_message_mb * 1024 * 1024,
    )
    try:
        await wait_for_queue(ws, timeout_s=args.event_timeout_s)
        session_created = await init_session(ws, args)
        log(f"session.created id={session_created.get('session_id')}")
        events.append(session_created)

        for idx in range(num_units):
            input_id = f"unit_{idx:03d}"
            frame = frame_b64[min(idx, len(frame_b64) - 1)] if frame_b64 else None
            payload = {
                "type": "input.append",
                "input": {
                    "input_id": input_id,
                    "audio": b64_float32(chunks[idx]),
                    "force_listen": False,
                    "max_slice_nums": 1,
                },
            }
            if frame is not None:
                payload["input"]["video_frames"] = [frame]
            log(f"unit={input_id} send input.append audio_samples={len(chunks[idx])} frames={1 if frame is not None else 0}")
            await ws.send(json.dumps(payload, ensure_ascii=False))
            unit, unit_events = await collect_unit(
                ws,
                input_id=input_id,
                listen_silence_samples=int(OUTPUT_SAMPLE_RATE * args.chunk_ms / 1000),
                post_audio_drain_s=args.post_audio_drain_ms / 1000.0,
                event_timeout_s=args.event_timeout_s,
            )
            unit["unit_id"] = idx
            units.append(unit)
            events.extend(unit_events)
            audio_meta_item = unit.get("audio") or {}
            if audio_meta_item.get("samples", 0) > 0:
                audio_arrays = [
                    decode_audio_b64(item["audio"])
                    for item in unit_events
                    if item.get("type") == "response.output.delta"
                    and item.get("input_id") == input_id
                    and item.get("kind") == "audio"
                    and item.get("audio")
                ]
                if audio_arrays:
                    write_wav(out_dir / f"unit_{idx:03d}.wav", np.concatenate(audio_arrays))

        events.extend(await close_session(ws, timeout_s=args.event_timeout_s))
    finally:
        await ws.close()

    canonical_units = [
        {
            "unit_id": item["unit_id"],
            "prefill_success": None,
            "is_listen": item["is_listen"],
            "text": item["text"],
            "end_of_turn": item["end_of_turn"],
            "n_tokens": item["n_tokens"],
            "n_tts_tokens": item["n_tts_tokens"],
            "audio": item["audio"],
        }
        for item in units
    ]
    summary = {
        "url": url,
        "video": args.video,
        "prompt_wav": args.prompt_wav,
        "system_prompt": args.system_prompt,
        "duplex_config": duplex_config(args),
        "session_created": session_created,
        "units": canonical_units,
    }
    (out_dir / "canonical_units.json").write_text(
        json.dumps(canonical_units, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.include_events:
        (out_dir / "events.json").write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "out_dir": str(out_dir),
        "session_id": session_created.get("session_id") if session_created else None,
        "units": len(units),
        "text": "".join(item["text"] for item in units),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--prompt-wav", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--system-prompt", default="Streaming Omni Conversation.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-ms", type=int, default=1000)
    parser.add_argument("--max-units", type=int, default=8)
    parser.add_argument("--force-listen-count", type=int, default=0)
    parser.add_argument("--max-new-speak-tokens-per-chunk", type=int, default=20)
    parser.add_argument("--decode-mode", default="greedy")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--listen-prob-scale", type=float, default=1.0)
    parser.add_argument("--listen-top-k", type=int, default=None)
    parser.add_argument("--text-repetition-penalty", type=float, default=1.05)
    parser.add_argument("--text-repetition-window-size", type=int, default=512)
    parser.add_argument("--length-penalty", type=float, default=1.1)
    parser.add_argument("--post-audio-drain-ms", type=float, default=200.0)
    parser.add_argument(
        "--strategy-hd",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable lag-one strategy-HD slice scheduling in the API session",
    )
    parser.add_argument("--strategy-hd-max-slice-nums", type=int, default=4)
    parser.add_argument("--event-timeout-s", type=float, default=600.0)
    parser.add_argument("--open-timeout", type=float, default=15.0)
    parser.add_argument("--max-message-mb", type=int, default=128)
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--include-events", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = asyncio.run(run_probe(args))
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
