#!/usr/bin/env python3
"""Run one Tauvoice FC-duplex sample through the backend WebSocket.

This is intentionally a small probe: it sends user audio in 1s chunks, executes
the toy Tauvoice tool locally when the model emits a tool call, feeds the tool
result back on the next chunk, and records all backend events.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import math
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import soundfile as sf
import websockets


DEFAULT_CASE = (
    "/user/heweiquan/dataset/O5_Dubplex_FC/overfit100/"
    "tau_function_call_sft_v13_think_cleaned_sample_2_colloquialize_tts_atom_export_relation_01029.json"
)


def ws_url(base_url: str, path: str) -> str:
    parsed = urlsplit(base_url.rstrip("/") + path)
    scheme = "ws" if parsed.scheme == "http" else "wss"
    return urlunsplit((scheme, parsed.netloc, parsed.path, parsed.query, parsed.fragment))


def extract_case(case_path: Path) -> tuple[str, List[Dict[str, Any]], Path, float]:
    data = json.loads(case_path.read_text(encoding="utf-8"))
    tools = data.get("system", {}).get("tools") or []
    system_prompt = "".join(seg.get("text", "") for seg in data.get("system", {}).get("segments", []))
    unit_sec = float(data.get("unit_policy", {}).get("unit_sec") or 1.0)
    rel_audio = data["tracks"]["user_audio"]["segments"][0]["audio"]["file_path"]
    audio_path = case_path.parent / rel_audio
    return system_prompt, tools, audio_path, unit_sec


def encode_float32_audio(samples: np.ndarray) -> str:
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    return base64.b64encode(samples.tobytes()).decode("ascii")


def convert_decimal_to_binary(arguments: Any) -> str:
    if isinstance(arguments, str):
        arguments = json.loads(arguments)
    number = int(float((arguments or {}).get("decimal_number")))
    return json.dumps(
        {
            "result": [
                {
                    "name": "convert_decimal_to_binary",
                    "arguments": {"decimal_number": number},
                    "results": {"binary_representation": bin(number)[2:]},
                }
            ]
        },
        ensure_ascii=False,
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="http://127.0.0.1:22512")
    parser.add_argument("--path", default="/backend")
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--out", default="run-logs/fc_tauvoice_probe.json")
    parser.add_argument("--extra-silence-units", type=int, default=8)
    parser.add_argument("--non-spoken-budget", type=int, default=30)
    parser.add_argument("--max-spoken-tokens", type=int, default=24)
    args = parser.parse_args()

    case_path = Path(args.case)
    system_prompt, tools, audio_path, unit_sec = extract_case(case_path)
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        raise RuntimeError(f"expected 16k audio, got {sr}: {audio_path}")
    chunk = int(sr * unit_sec)
    chunks = [audio[i : i + chunk] for i in range(0, len(audio), chunk)]
    chunks.extend([np.zeros(chunk, dtype=np.float32) for _ in range(args.extra_silence_units)])

    events: List[Dict[str, Any]] = []
    pending_tool_results: List[Dict[str, Any]] = []

    async with websockets.connect(ws_url(args.backend, args.path), max_size=128 * 1024 * 1024, ping_interval=None) as ws:
        await ws.send(json.dumps({
            "type": "session.init",
            "payload": {
                "mode": "full_duplex",
                "fc_duplex": True,
                "system_prompt": system_prompt,
                "tools": tools,
                "generate_audio": True,
                "config": {
                    "sample_rate": sr,
                    "unit_sec": unit_sec,
                    "decode_mode": "greedy",
                    "non_spoken_budget_per_unit": args.non_spoken_budget,
                    "max_spoken_tokens": args.max_spoken_tokens,
                    "non_spoken_scheduling": "quality",
                },
            },
        }, ensure_ascii=False))
        events.append(json.loads(await ws.recv()))

        async def drain_until_idle(timeout_s: float = 0.2) -> None:
            while True:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
                except asyncio.TimeoutError:
                    return
                event = json.loads(raw)
                events.append(event)
                if event.get("type") == "response.tool_call.args.raw":
                    raw_call = event.get("raw") or {}
                    result = convert_decimal_to_binary(json.loads(raw_call.get("arguments") or "{}"))
                    pending_tool_results.append({
                        "type": "tool_result",
                        "tool_call_id": event.get("tool_call_id"),
                        "content": result,
                    })

        for idx, samples in enumerate(chunks):
            payload = {
                "type": "input.append",
                "input": {
                    "type": "audio",
                    "input_id": f"unit_{idx:03d}",
                    "audio": encode_float32_audio(samples),
                    "sample_rate": sr,
                },
            }
            if pending_tool_results:
                payload["input"]["tool_responses"] = pending_tool_results
                # Backend runtime accepts explicit tool_result payloads more directly.
                for item in pending_tool_results:
                    await ws.send(json.dumps({"type": "input.append", "input": item}, ensure_ascii=False))
                pending_tool_results = []
            await ws.send(json.dumps(payload, ensure_ascii=False))
            await drain_until_idle()

        await ws.send(json.dumps({"type": "session.close", "reason": "probe_done"}))
        with contextlib.suppress(Exception):
            events.append(json.loads(await asyncio.wait_for(ws.recv(), timeout=5)))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"case": str(case_path), "audio": str(audio_path), "events": events}, ensure_ascii=False, indent=2), encoding="utf-8")
    tool_calls = [e for e in events if e.get("type") == "response.tool_call.args.raw"]
    spoken = "".join(e.get("delta", "") for e in events if e.get("type") == "response.output.delta" and e.get("kind") == "text")
    print(json.dumps({"out": str(out_path), "events": len(events), "tool_calls": tool_calls, "spoken_text": spoken}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import contextlib

    asyncio.run(main())
