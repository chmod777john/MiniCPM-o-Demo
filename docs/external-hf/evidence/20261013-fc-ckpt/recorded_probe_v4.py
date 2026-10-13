#!/usr/bin/env python3
"""Exercise real FC WebSocket input/tool roundtrips using one recorded bench fixture.

Recorded tool responses are fixtures; this client never calls external tools.
This is an API integration probe, not a scored OmniAgentBench evaluation.
"""
from __future__ import annotations
import argparse
import asyncio
import base64
from collections import Counter
import copy
import json
from pathlib import Path
import time
import urllib.request

import numpy as np
import soundfile as sf
import websockets


def call_key(call):
    return (call["name"], json.dumps(call.get("arguments") or {}, ensure_ascii=False, sort_keys=True))


def load_fixture(root):
    params, calls, responses = None, {}, {}
    for line in (root / "events.jsonl").open():
        entry = json.loads(line)
        frame = entry["frame"]
        if entry["direction"] == "tx" and frame["type"] == "session.init":
            params = copy.deepcopy(frame["payload"])
        if entry["direction"] == "rx" and frame["type"] == "response.tool_call.done" and frame.get("call"):
            calls[frame["tool_call_id"]] = frame["call"]
        if entry["direction"] == "tx" and frame["type"] == "input.tool_result" and frame.get("tool_call_id") in calls:
            responses.setdefault(call_key(calls[frame["tool_call_id"]]), frame.get("content", frame.get("contents")))
        if entry["direction"] == "tx" and frame["type"] == "input.append":
            inp = frame.get("input", {})
            if inp.get("type") == "tool_result" and inp.get("tool_call_id") in calls:
                responses.setdefault(call_key(calls[inp["tool_call_id"]]), inp.get("content", inp.get("contents")))
    if params is None:
        raise ValueError("Fixture lacks session.init")
    return params, responses


async def run(args):
    params, responses = load_fixture(args.fixture)
    params["checkpoint_profile_id"] = args.profile_id
    params["non_spoken_scheduling"] = "quality"
    params["generate_audio"] = args.speech
    # Exact existing policy retained; scheduling completion is awaited so all
    # submitted units can be attributed and no input can be dropped.
    audio, sample_rate = sf.read(args.fixture / "input_timeline.wav", dtype="float32")
    if sample_rate != 16000 or audio.ndim != 1:
        raise ValueError("Expected mono 16kHz fixture")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    events = (args.output_dir / "events.jsonl").open("w")
    counts = Counter()
    tool_calls, supplied_results, consumed_results, units = [], [], [], []
    session_id = None
    errors = []
    synthetic_tool_errors = []
    def record(direction, message):
        frame = copy.deepcopy(message)
        if direction == "rx":
            counts[message["type"]] += 1
        # Keep PCM as a side artifact and preserve the reference in the log.
        if direction == "rx" and frame.get("type") == "response.spoken.delta" and frame.get("audio"):
            raw = base64.b64decode(frame.pop("audio"))
            name = f"speech_{counts[message['type']]:04d}.f32le"
            (args.output_dir / name).write_bytes(raw)
            frame["audio_ref"] = name
            frame["audio_samples"] = len(raw)//4
        if direction == "tx" and frame.get("type") == "input.append":
            inp = frame["input"]
            if "audio" in inp:
                inp["audio_fixture"] = str(args.fixture / "input_timeline.wav")
                inp.pop("audio")
        events.write(json.dumps({"direction": direction, "at": time.time(), "frame": frame}, ensure_ascii=False)+"\n")
        events.flush()
    try:
        async with websockets.connect(args.url, max_size=128*1024*1024, open_timeout=30, ping_timeout=None) as ws:
            async def send(frame):
                record("tx", frame)
                await ws.send(json.dumps(frame, ensure_ascii=False))
            async def receive():
                message = json.loads(await asyncio.wait_for(ws.recv(), timeout=args.timeout))
                record("rx", message)
                if message["type"] in {"error", "session.closed"}:
                    raise RuntimeError(f"Unexpected server event: {message}")
                return message
            await send({"type": "session.init", "payload": params})
            while session_id is None:
                message = await receive()
                if message["type"] == "session.created":
                    session_id = message["session_id"]
                    print("FC_API_SESSION", session_id, flush=True)
            for unit in range(args.units):
                chunk = audio[unit*16000:(unit+1)*16000]
                if len(chunk)<16000:
                    chunk=np.pad(chunk,(0,16000-len(chunk)))
                input_id = f"probe_input_{unit:06d}"
                t0 = time.perf_counter()
                await send({"type": "input.append", "input": {"input_id": input_id,
                            "sample_rate": 16000, "audio": base64.b64encode(chunk.astype("<f4").tobytes()).decode()}})
                while True:
                    message = await receive()
                    if message["type"] == "response.tool_call.done":
                        if message.get("error"):
                            errors.append(message)
                        call = message.get("call")
                        if call:
                            tool_calls.append(message)
                            key = call_key(call)
                            if key in responses:
                                content = responses[key]
                            elif call["name"] == "show_card":
                                # UI acknowledgements are content-independent. This is a
                                # local fixture acknowledgement, not an external write.
                                content = "卡片已展示给用户。"
                            elif getattr(args, "missing_fixture", "fail") == "error-result":
                                synthetic_tool_errors.append({"tool_call_id": message["tool_call_id"], "call": call})
                                content = json.dumps({"error": "tool_unavailable",
                                    "message": "API test fixture has no recorded result for this call."})
                                print("FC_API_SYNTHETIC_TOOL_ERROR", unit, call, flush=True)
                            else:
                                raise RuntimeError(f"No recorded tool-response fixture for {key}")
                            tool_id = message["tool_call_id"]
                            await send({"type": "input.tool_result", "tool_call_id": tool_id, "content": content})
                            supplied_results.append(tool_id)
                            print("FC_API_TOOL", unit, call, flush=True)
                    if message["type"] == "response.unit.started":
                        consumed_results.extend(event["tool_call_id"] for event in message.get("tool_events", [])
                                                if event["type"] == "tool_result")
                    if message["type"] == "response.unit.committed":
                        assert message["unit_index"] == unit
                        units.append({"unit": unit, "elapsed_s": time.perf_counter()-t0,
                                      "input_id": input_id, "resume": message.get("resume")})
                        print("FC_API_UNIT", unit, round(units[-1]["elapsed_s"],3), flush=True)
                        break
            # Disconnect is the ordinary cleanup path; no traffic is sent to
            # anyone else's server or tooling endpoint.
    except Exception as exc:
        errors.append({"exception": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        events.close()
        summary = {"fixture": str(args.fixture), "url": args.url, "session_id": session_id,
                   "generate_audio": args.speech, "event_counts": dict(counts), "units": units,
                   "tool_calls": tool_calls, "supplied_result_ids": supplied_results,
                   "consumed_result_ids": consumed_results, "errors": errors,
                   "missing_fixture_policy": getattr(args, "missing_fixture", "fail"),
                   "synthetic_tool_errors": synthetic_tool_errors,
                   "scope": "Real API probe with recorded fixtures and explicitly opted-in synthetic unavailable-tool errors; not a quality score or numerical speech comparison."}
        (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n")
    assert len(units) == args.units
    assert tool_calls and consumed_results, "Probe did not cover a complete tool-result roundtrip"
    assert not errors, errors
    if args.speech:
        assert counts["response.spoken.delta"] > 0, "Speech enabled but no waveform received"
    print("FC_API_PROBE_DONE", flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="ws://127.0.0.1:9129/backend")
    p.add_argument("--profile-id", default="o5-fc-sdk006-trained")
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--units", type=int, default=10)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--speech", action="store_true")
    p.add_argument("--missing-fixture", choices=("fail", "error-result"), default="fail",
                   help="Explicit functional-test fallback; error-result is not a quality evaluation.")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
