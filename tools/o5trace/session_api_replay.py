#!/usr/bin/env python3
"""Replay a newly recorded realtime session through a live realtime API.

This is the live-API counterpart of `session_canonical_replay.py`. It consumes
the exact `payload_trace` blobs recorded by the session recorder:

- audio: `payload_trace.audio.blob_f32`
- video: `payload_trace.video_frames[].blob_jpg`

Older sessions that only have `@blob/*.wav` inputs should use
`legacy_session_api_replay.py` instead.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import math
import ssl
import sys
import time
import wave
from array import array
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import websockets


INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


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
    if not isinstance(rel, str) or not rel.startswith("@blob/"):
        raise ValueError(f"expected @blob pointer, got: {rel!r}")
    path = (session_dir / rel[1:]).resolve()
    if not str(path).startswith(str(session_dir.resolve())):
        raise ValueError(f"blob path escapes session dir: {rel}")
    return path


def read_blob_bytes(session_dir: Path, rel: str, expected_sha: Optional[str]) -> bytes:
    raw = blob_path(session_dir, rel).read_bytes()
    if expected_sha and sha_bytes(raw) != expected_sha:
        raise ValueError(f"sha256 mismatch for {rel}")
    return raw


def session_init_payload(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in events:
        frame = event.get("frame")
        if isinstance(frame, dict) and frame.get("type") == "session.init":
            payload = frame.get("payload")
            if isinstance(payload, dict):
                return dict(payload)
    raise ValueError("recorded session has no session.init payload")


def session_replay_manifest(events: list[dict[str, Any]]) -> dict[str, Any]:
    for event in events:
        frame = event.get("frame")
        if isinstance(frame, dict) and frame.get("type") == "session.created":
            manifest = frame.get("replay_manifest")
            if isinstance(manifest, dict):
                return manifest
    return {}


def input_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for event in events:
        if event.get("dir") != "up":
            continue
        frame = event.get("frame")
        if isinstance(frame, dict) and frame.get("type") == "input.append":
            out.append(event)
    if not out:
        raise ValueError("recorded session has no upstream input.append events")
    return out


def output_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for event in events:
        if event.get("dir") != "down":
            continue
        frame = event.get("frame")
        if isinstance(frame, dict) and frame.get("type") == "response.output.delta":
            out.append(event)
    return out


def normalize_url(url: str, mode: str) -> str:
    if url.startswith("ws://") or url.startswith("wss://"):
        return url
    scheme = "wss" if url.startswith("https://") else "ws"
    return f"{scheme}://{url.split('://', 1)[-1].rstrip('/')}/v1/realtime?mode={mode}"


def event_input_id(event: dict[str, Any], idx: int) -> Optional[str]:
    frame = event.get("frame") if isinstance(event.get("frame"), dict) else {}
    inp = frame.get("input") if isinstance(frame.get("input"), dict) else {}
    value = inp.get("input_id")
    if value is None:
        return None
    return str(value)


def event_force_listen(event: dict[str, Any]) -> Optional[bool]:
    frame = event.get("frame") if isinstance(event.get("frame"), dict) else {}
    inp = frame.get("input") if isinstance(frame.get("input"), dict) else {}
    if "force_listen" in inp:
        return bool(inp.get("force_listen"))
    trace = event.get("payload_trace") if isinstance(event.get("payload_trace"), dict) else {}
    if "force_listen" in trace:
        return bool(trace.get("force_listen"))
    return None


def event_max_slice_nums(event: dict[str, Any]) -> Optional[int]:
    frame = event.get("frame") if isinstance(event.get("frame"), dict) else {}
    inp = frame.get("input") if isinstance(frame.get("input"), dict) else {}
    if "max_slice_nums" in inp:
        return int(inp.get("max_slice_nums") or 1)
    trace = event.get("payload_trace") if isinstance(event.get("payload_trace"), dict) else {}
    if "max_slice_nums" in trace:
        return int(trace.get("max_slice_nums") or 1)
    return None


def build_input_payload(session_dir: Path, event: dict[str, Any], idx: int) -> dict[str, Any]:
    trace = event.get("payload_trace") if isinstance(event.get("payload_trace"), dict) else {}
    audio_meta = trace.get("audio") if isinstance(trace.get("audio"), dict) else None
    if not audio_meta or not audio_meta.get("blob_f32"):
        raise ValueError(
            f"input event seq={event.get('seq')} has no payload_trace.audio.blob_f32; "
            "use legacy_session_api_replay.py for old sessions"
        )
    if int(audio_meta.get("sample_rate", INPUT_SAMPLE_RATE)) != INPUT_SAMPLE_RATE:
        raise ValueError(f"input event seq={event.get('seq')} sample rate is not {INPUT_SAMPLE_RATE}")

    audio_raw = read_blob_bytes(session_dir, str(audio_meta["blob_f32"]), audio_meta.get("sha256"))
    if len(audio_raw) % 4 != 0:
        raise ValueError(f"input event seq={event.get('seq')} f32 audio byte length is not divisible by 4")

    frames_b64: list[str] = []
    frame_meta = trace.get("video_frames")
    if frame_meta:
        if not isinstance(frame_meta, list):
            raise ValueError(f"input event seq={event.get('seq')} malformed video frame trace")
        for item in frame_meta:
            if not item:
                continue
            if not isinstance(item, dict) or not item.get("blob_jpg"):
                raise ValueError(f"input event seq={event.get('seq')} malformed video frame trace")
            jpg_raw = read_blob_bytes(session_dir, str(item["blob_jpg"]), item.get("sha256"))
            frames_b64.append(base64.b64encode(jpg_raw).decode("ascii"))

    payload: dict[str, Any] = {
        "audio": base64.b64encode(audio_raw).decode("ascii"),
        "video_frames": frames_b64,
    }
    input_id = event_input_id(event, idx)
    if input_id is not None:
        payload["input_id"] = input_id
    force_listen = event_force_listen(event)
    if force_listen is not None:
        payload["force_listen"] = force_listen
    max_slice_nums = event_max_slice_nums(event)
    if max_slice_nums is not None:
        payload["max_slice_nums"] = max_slice_nums
    return payload


def message_audio_bytes(msg: dict[str, Any]) -> Optional[bytes]:
    value = msg.get("audio")
    if not isinstance(value, str) or not value:
        return None
    if value.startswith("@blob/"):
        # A live websocket response normally carries base64 audio. A blob
        # pointer here would be relative to the remote service session and is
        # not readable from this client process.
        return None
    return base64.b64decode(value)


def response_kind_counts(events: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"listen": 0, "text": 0, "audio": 0, "other": 0}
    for event in events:
        frame = event.get("frame") if isinstance(event.get("frame"), dict) else {}
        kind = frame.get("kind")
        if kind in counts:
            counts[kind] += 1
        else:
            counts["other"] += 1
    return counts


def manifest_value(manifest: dict[str, Any], path: str) -> Any:
    cur: Any = manifest
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def compare_manifests(source: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    paths = [
        "resolved.seed",
        "resolved.llm_seed",
        "resolved.seed_source",
        "resolved.deterministic_replay",
        "resolved.duplex_config",
        "runtime.server_config.model_path",
        "runtime.server_config.pt_path",
        "runtime.env.O5_DEPLOY_MODE",
        "runtime.env.O5_BACKBONE_DIR",
        "runtime.env.O5_LLM_CACHE",
        "runtime.env.O5_TTS_GRAPH",
        "runtime.env.O5_LLM_GRAPH",
        "runtime.env.O5_EXPERTS_IMPLEMENTATION",
        "runtime.env.O5_TTS_ARGMAX",
        "runtime.env.O5_DETERMINISTIC_REPLAY",
        "runtime.env.O5_SESSION_SEED",
    ]
    checks = []
    mismatches = []
    for path in paths:
        src = manifest_value(source, path)
        dst = manifest_value(replay, path)
        ok = src == dst
        item = {"path": path, "source": src, "replay": dst, "match": ok}
        checks.append(item)
        if not ok:
            mismatches.append(item)
    return {"checks": checks, "mismatches": mismatches}


def float32_to_pcm16(data: bytes) -> bytes:
    if len(data) % 4 != 0:
        raise ValueError(f"float32 audio byte length is not divisible by 4: {len(data)}")
    values = array("f")
    values.frombytes(data)
    if sys.byteorder != "little":
        values.byteswap()

    pcm = array("h")
    for value in values:
        if not math.isfinite(value):
            value = 0.0
        value = max(-1.0, min(1.0, value))
        pcm.append(round(value * 32767.0))
    if sys.byteorder != "little":
        pcm.byteswap()
    return pcm.tobytes()


def write_audio_concat(out_dir: Path, audio_parts: list[bytes]) -> Optional[str]:
    if not audio_parts:
        return None
    pcm_chunks: list[bytes] = []
    for data in audio_parts:
        if data.startswith(b"RIFF"):
            import io

            with wave.open(io.BytesIO(data), "rb") as wf:
                params = wf.getparams()
                raw = wf.readframes(wf.getnframes())
            if (params.nchannels, params.framerate) != (1, OUTPUT_SAMPLE_RATE):
                raise ValueError(f"response audio params mismatch: {params}")
            if params.sampwidth == 2:
                pcm_chunks.append(raw)
            else:
                raise ValueError(f"unsupported response WAV sample width: {params.sampwidth}")
        else:
            # The realtime API carries float32 PCM bytes, while the output
            # WAV is deliberately normalized to broadly supported PCM16.
            pcm_chunks.append(float32_to_pcm16(data))

    path = out_dir / "response_audio_concat.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(OUTPUT_SAMPLE_RATE)
        wf.writeframes(b"".join(pcm_chunks))
    return path.name


async def recv_loop(
    ws: websockets.ClientConnection,
    *,
    stream,
    event_timeout_s: float,
    stop_event: asyncio.Event,
    text_parts: list[str],
    audio_parts: list[bytes],
    state: dict[str, Any],
) -> None:
    while not stop_event.is_set():
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=event_timeout_s)
        except asyncio.TimeoutError:
            state["timeout"] = True
            return
        except websockets.ConnectionClosed:
            state["closed"] = True
            return
        msg = json.loads(raw)
        state["last_msg_at"] = time.perf_counter()
        stream.write(json.dumps({"ts": time.time(), "dir": "down", "frame": msg}, ensure_ascii=False) + "\n")
        stream.flush()
        if msg.get("type") == "session.created":
            state["api_session_id"] = msg.get("session_id")
            manifest = msg.get("replay_manifest")
            if isinstance(manifest, dict):
                state["replay_manifest"] = manifest
        elif msg.get("type") == "response.output.delta":
            if msg.get("kind") == "text":
                text_parts.append(msg.get("text") or "")
            elif msg.get("kind") == "audio":
                data = message_audio_bytes(msg)
                if data:
                    audio_parts.append(data)
        elif msg.get("type") in {"error", "session.closed"}:
            state["terminal"] = msg
            return


async def wait_for_event(
    ws: websockets.ClientConnection,
    *,
    stream,
    event_timeout_s: float,
    wanted: set[str],
) -> dict[str, Any]:
    while True:
        raw = await asyncio.wait_for(ws.recv(), timeout=event_timeout_s)
        msg = json.loads(raw)
        stream.write(json.dumps({"ts": time.time(), "dir": "down", "frame": msg}, ensure_ascii=False) + "\n")
        stream.flush()
        msg_type = msg.get("type")
        if msg_type in wanted:
            return msg
        if msg_type in {"error", "session.closed"}:
            raise RuntimeError(json.dumps(msg, ensure_ascii=False))


def recorded_delay(first_ts: float, event: dict[str, Any]) -> float:
    ts = event.get("ts")
    if not isinstance(ts, (int, float)):
        return 0.0
    return max(0.0, float(ts) - first_ts)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    session_dir = Path(args.session_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    events = read_jsonl(session_dir / "stream.jsonl")
    init_payload = session_init_payload(events)
    source_manifest = session_replay_manifest(events)
    units = input_events(events)
    if args.max_units > 0:
        units = units[: args.max_units]
    source_outputs = output_events(events)

    url = normalize_url(args.url, args.mode)
    parsed = urlparse(url)
    ssl_ctx = None
    if parsed.scheme == "wss":
        ssl_ctx = ssl._create_unverified_context() if args.insecure else ssl.create_default_context()

    text_parts: list[str] = []
    audio_parts: list[bytes] = []
    state: dict[str, Any] = {}
    stream_path = out_dir / "api_stream.jsonl"

    ws = await websockets.connect(
        url,
        ssl=ssl_ctx,
        open_timeout=args.open_timeout,
        max_size=args.max_message_mb * 1024 * 1024,
    )
    try:
        with stream_path.open("w", encoding="utf-8") as stream:
            await wait_for_event(ws, stream=stream, event_timeout_s=args.event_timeout_s, wanted={"session.queue_done", "queue_done"})

            init_frame = {"type": "session.init", "payload": init_payload}
            await ws.send(json.dumps(init_frame, ensure_ascii=False))
            stream.write(json.dumps({"ts": time.time(), "dir": "up", "frame": init_frame}, ensure_ascii=False) + "\n")
            stream.flush()

            created = await wait_for_event(ws, stream=stream, event_timeout_s=args.event_timeout_s, wanted={"session.created"})
            state["api_session_id"] = created.get("session_id")
            if isinstance(created.get("replay_manifest"), dict):
                state["replay_manifest"] = created["replay_manifest"]

            stop_event = asyncio.Event()
            recv_task = asyncio.create_task(
                recv_loop(
                    ws,
                    stream=stream,
                    event_timeout_s=args.event_timeout_s,
                    stop_event=stop_event,
                    text_parts=text_parts,
                    audio_parts=audio_parts,
                    state=state,
                )
            )

            started = time.perf_counter()
            first_input_ts = float(units[0].get("ts") or 0.0)
            sent = 0
            for idx, event in enumerate(units):
                if args.timing == "recorded":
                    target = recorded_delay(first_input_ts, event) / max(args.pace_scale, 1e-6)
                    sleep_s = started + target - time.perf_counter()
                    if sleep_s > 0:
                        await asyncio.sleep(sleep_s)

                payload = build_input_payload(session_dir, event, idx)
                frame = {"type": "input.append", "input": payload}
                await ws.send(json.dumps(frame, ensure_ascii=False))
                sent += 1
                record_frame = {
                    "type": "input.append",
                    "input": {
                        "input_id": payload.get("input_id"),
                        "audio_bytes": len(base64.b64decode(payload["audio"])),
                        "audio_samples": len(base64.b64decode(payload["audio"])) // 4,
                        "video_frames": len(payload.get("video_frames") or []),
                        "force_listen": payload.get("force_listen"),
                        "max_slice_nums": payload.get("max_slice_nums"),
                    },
                    "source_seq": event.get("seq"),
                    "source_ts": event.get("ts"),
                }
                stream.write(json.dumps({"ts": time.time(), "dir": "up", "frame": record_frame}, ensure_ascii=False) + "\n")
                stream.flush()

                if args.timing == "step":
                    before = state.get("last_msg_at")
                    deadline = time.perf_counter() + args.unit_timeout_s
                    while True:
                        if state.get("terminal"):
                            raise RuntimeError(json.dumps(state["terminal"], ensure_ascii=False))
                        if state.get("last_msg_at") is not None and state.get("last_msg_at") != before:
                            break
                        if time.perf_counter() >= deadline:
                            raise TimeoutError(f"timeout waiting output after unit {idx}")
                        await asyncio.sleep(0.02)

            last_input_at = time.perf_counter()
            while time.perf_counter() - last_input_at < args.drain_s:
                if state.get("terminal") or state.get("closed"):
                    break
                await asyncio.sleep(0.1)
            stop_event.set()
            await asyncio.wait({recv_task}, timeout=1.0)
    finally:
        await ws.close()

    audio_name = write_audio_concat(out_dir, audio_parts)
    replay_manifest = state.get("replay_manifest") if isinstance(state.get("replay_manifest"), dict) else {}
    manifest_compare = compare_manifests(source_manifest, replay_manifest) if source_manifest and replay_manifest else {}
    (out_dir / "manifest_compare.json").write_text(
        json.dumps(manifest_compare, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.strict_manifest and manifest_compare.get("mismatches"):
        raise RuntimeError(f"manifest mismatch: {out_dir / 'manifest_compare.json'}")

    summary = {
        "url": url,
        "mode": args.mode,
        "timing": args.timing,
        "source_session_dir": str(session_dir),
        "api_session_id": state.get("api_session_id"),
        "units_sent": len(units),
        "source_output_counts": response_kind_counts(source_outputs),
        "replay_text_delta_count": len(text_parts),
        "replay_audio_chunks": len(audio_parts),
        "replay_text": "".join(text_parts),
        "stream": stream_path.name,
        "response_audio": audio_name,
        "manifest_compare": "manifest_compare.json",
        "manifest_mismatches": len(manifest_compare.get("mismatches", [])) if manifest_compare else None,
        "receiver_state": {k: v for k, v in state.items() if k not in {"replay_manifest", "terminal"}},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--url", default="https://82.157.64.212:8010")
    parser.add_argument("--mode", default="video", choices=("video", "audio"))
    parser.add_argument("--max-units", type=int, default=0)
    parser.add_argument("--timing", choices=("recorded", "step"), default="recorded")
    parser.add_argument("--pace-scale", type=float, default=1.0, help="Scale recorded input delays; 2.0 sends twice as fast.")
    parser.add_argument("--drain-s", type=float, default=8.0)
    parser.add_argument("--unit-timeout-s", type=float, default=180.0)
    parser.add_argument("--event-timeout-s", type=float, default=180.0)
    parser.add_argument("--open-timeout", type=float, default=30.0)
    parser.add_argument("--max-message-mb", type=int, default=32)
    parser.add_argument("--strict-manifest", action="store_true")
    parser.add_argument("--insecure", action="store_true")
    return parser.parse_args()


def main() -> int:
    summary = asyncio.run(run(parse_args()))
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
