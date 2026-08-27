#!/usr/bin/env python3
"""Replay one board FC-duplex train-data sample through the backend API.

The important part is the scheduler: audio chunks, non-spoken budgets, and GT
tool-response target units are derived from the same SDK arrangement used by
offline train-data evaluation.  The model still generates tool calls freely;
when a generated call can be matched by order to the train call id, this probe
replays the train tool response at the arranged unit.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import copy
import hashlib
import json
import re
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import websockets

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.processors.unified import FcDuplexView
from core.fc_duplex.system_input import (
    FcAudioPathInput,
    FcSystemAudioInput,
    FcSystemContentInput,
    FcSystemTextInput,
)
from core.schemas.fc_duplex import FcDuplexConfig


DEFAULT_CASE = (
    "/user/weihongliang/o5_fc_assets/board_mvp_20260724/delivery_train_data/"
    "dob_midtrain_v1_20260628_animal_seed_ct01_and_04702.json"
)


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def ws_url(base_url: str, path: str) -> str:
    parsed = urlsplit(base_url.rstrip("/") + path)
    scheme = "ws" if parsed.scheme == "http" else "wss"
    return urlunsplit((scheme, parsed.netloc, parsed.path, parsed.query, parsed.fragment))


def encode_float32_audio(samples: np.ndarray) -> str:
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    return base64.b64encode(samples.tobytes()).decode("ascii")


def make_sdk_compatible_train_data(structure: Dict[str, Any]) -> Dict[str, Any]:
    compatible = copy.deepcopy(structure)
    tracks = compatible.get("tracks")
    if not isinstance(tracks, dict):
        return compatible
    for track_name in ("input_event", "ai_spoken", "ai_non_spoken"):
        track = tracks.get(track_name)
        if isinstance(track, dict):
            track.pop("allow_segment_overlap", None)
    return compatible


def extract_system_prompt(structure: Dict[str, Any]) -> str:
    return "\n".join(
        segment.get("text", "")
        for segment in structure.get("system", {}).get("segments", []) or []
        if segment.get("kind") == "text"
    )


def extract_tools(structure: Dict[str, Any], *, normalize: bool) -> List[Dict[str, Any]]:
    tools = list(structure.get("system", {}).get("tools") or [])
    if not normalize:
        return tools
    from minicpm_o5_sdk import OpenAIToolDefinition

    return [OpenAIToolDefinition.model_validate(tool).model_dump() for tool in tools]


def resolve_media_path(data_root: Path, file_path: Optional[str]) -> Optional[Path]:
    if not file_path:
        return None
    path = Path(file_path)
    return path if path.is_absolute() else data_root / path


def extract_ref_audio_path(structure: Dict[str, Any], data_root: Path) -> Optional[str]:
    for segment in structure.get("system", {}).get("segments", []) or []:
        if segment.get("kind") != "audio":
            continue
        path = resolve_media_path(data_root, (segment.get("audio") or {}).get("file_path"))
        if path is not None and path.exists():
            return str(path)
    return None


def build_v3_system_content(
    structure: Dict[str, Any],
    data_root: Path,
    tools: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Project the train-data system block into Semantic Realtime v3 wire data."""

    segments: List[FcSystemTextInput | FcSystemAudioInput] = []
    for raw_segment in structure.get("system", {}).get("segments", []) or []:
        kind = raw_segment.get("kind")
        if kind == "text":
            segments.append(FcSystemTextInput(text=str(raw_segment.get("text", ""))))
            continue
        if kind != "audio":
            raise ValueError(f"unsupported system segment kind: {kind!r}")
        file_path = (raw_segment.get("audio") or {}).get("file_path")
        resolved = resolve_media_path(data_root, file_path)
        if resolved is None or not resolved.is_file():
            raise FileNotFoundError(f"system reference audio is not readable: {resolved}")
        segments.append(
            FcSystemAudioInput(
                audio=FcAudioPathInput(file_path=str(resolved.resolve(strict=True)))
            )
        )
    return FcSystemContentInput(segments=segments, tools=tools).model_dump(mode="json")


def extract_train_tool_call_ids(structure: Dict[str, Any]) -> List[str]:
    ids: List[str] = []
    for segment in ((structure.get("tracks") or {}).get("ai_non_spoken") or {}).get("segments") or []:
        content = segment.get("content") or {}
        if content.get("kind") == "tool_call" and content.get("tool_call_id"):
            ids.append(str(content["tool_call_id"]))
    return ids


def extract_gt_tool_calls(structure: Dict[str, Any]) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for segment in ((structure.get("tracks") or {}).get("ai_non_spoken") or {}).get("segments") or []:
        content = segment.get("content") or {}
        if content.get("kind") != "tool_call":
            continue
        calls.append(
            {
                "tool_call_id": content.get("tool_call_id"),
                "name": content.get("name") or content.get("function", {}).get("name"),
                "arguments": content.get("arguments") or content.get("function", {}).get("arguments"),
            }
        )
    return calls


def tool_response_content_from_event(event: Any) -> str:
    contents = getattr(event, "contents", None) or []
    parts: List[str] = []
    for item in contents:
        if isinstance(item, dict):
            if item.get("kind") == "text":
                parts.append(str(item.get("text", "")))
        elif getattr(item, "kind", None) == "text":
            parts.append(str(getattr(item, "text", "")))
    return "".join(parts)


def build_tool_responses_by_unit(arrangement: Any) -> Dict[int, List[Dict[str, Any]]]:
    responses: Dict[int, List[Dict[str, Any]]] = {}
    input_event_track = getattr(getattr(arrangement, "tracks", None), "input_event", None)
    if input_event_track is None:
        return responses
    for segment in getattr(input_event_track, "segments", []) or []:
        event = getattr(segment, "event", None)
        if getattr(event, "kind", None) != "tool_response":
            continue
        call_id = getattr(event, "tool_call_id", None)
        timeline = getattr(segment, "timeline", None)
        if not call_id or timeline is None:
            continue
        unit_index = int(getattr(timeline, "start_unit_index"))
        responses.setdefault(unit_index, []).append(
            {"train_tool_call_id": str(call_id), "content": tool_response_content_from_event(event)}
        )
    return responses


def normalize_raw_tool_call(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, str):
        with contextlib.suppress(Exception):
            raw = json.loads(raw)
    if not isinstance(raw, dict):
        return {"name": None, "arguments": raw}
    function = raw.get("function") if isinstance(raw.get("function"), dict) else {}
    name = raw.get("name") or function.get("name")
    arguments = raw.get("arguments")
    if arguments is None:
        arguments = function.get("arguments")
    if isinstance(arguments, str):
        with contextlib.suppress(Exception):
            arguments = json.loads(arguments)
    return {"name": name, "arguments": arguments}


def unit_index_from_input_id(input_id: Optional[str]) -> Optional[int]:
    if not input_id:
        return None
    match = re.search(r"(\d+)$", str(input_id))
    return int(match.group(1)) if match else None


def prepare_case(case_path: Path, *, normalize_tools: bool) -> Dict[str, Any]:
    structure = make_sdk_compatible_train_data(read_json(case_path))
    data_root = case_path.parent
    training_data, tokenized_result = FcDuplexView._load_sdk_train_data(
        structure,
        data_root,
        tokenizer_target="o5",
    )
    arrangement = tokenized_result.arrangement
    config = FcDuplexConfig()
    unit_chunks = FcDuplexView._build_unit_audio_chunks_from_arrangement(
        training_data,
        arrangement,
        config,
    )
    budgets_listening, budgets_speaking = FcDuplexView._build_non_spoken_budget_lists_from_arrangement(arrangement)
    unit_sec = float(getattr(arrangement.unit_policy, "unit_sec", config.unit_sec))
    sample_rate = int(config.sample_rate)
    tools = extract_tools(structure, normalize=normalize_tools)
    ref_audio_path = extract_ref_audio_path(structure, data_root)
    return {
        "case": str(case_path),
        "data_root": str(data_root),
        "structure": structure,
        "system_prompt": extract_system_prompt(structure),
        "tools": tools,
        "system": build_v3_system_content(structure, data_root, tools),
        "ref_audio_path": ref_audio_path,
        "unit_sec": unit_sec,
        "sample_rate": sample_rate,
        "unit_chunks": [np.asarray(chunk, dtype=np.float32).reshape(-1) for chunk in unit_chunks],
        "budgets_listening": budgets_listening,
        "budgets_speaking": budgets_speaking,
        "tool_responses_by_unit": build_tool_responses_by_unit(arrangement),
        "train_tool_call_ids": extract_train_tool_call_ids(structure),
        "gt_tool_calls": extract_gt_tool_calls(structure),
    }


def semantic_tool_calls(calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized = []
    for call in calls:
        raw = call.get("raw") if "raw" in call else call
        item = normalize_raw_tool_call(raw)
        normalized.append({"name": item.get("name"), "arguments": item.get("arguments")})
    return normalized


async def drain_available(ws: Any, handle_event: Any, *, timeout_s: float) -> None:
    while True:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
        except asyncio.TimeoutError:
            return
        await handle_event(json.loads(raw))


async def drain_until_unit_done(ws: Any, handle_event: Any, *, unit_index: int, timeout_s: float) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while True:
        remaining = deadline - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise TimeoutError(f"timeout waiting for unit {unit_index} completion")
        raw = await asyncio.wait_for(ws.recv(), timeout=remaining)
        event = json.loads(raw)
        await handle_event(event)
        if event.get("type") in {"error", "session.closed"}:
            raise RuntimeError(
                f"backend terminated while waiting for unit {unit_index}: {event}"
            )
        if (
            event.get("type") == "response.unit.committed"
            and int(event.get("unit_index", -1)) == unit_index
        ):
            return
        debug = event.get("debug") if event.get("type") == "response.debug" else None
        if isinstance(debug, dict) and debug.get("unit_index") == unit_index:
            return


async def run_probe(args: argparse.Namespace) -> Dict[str, Any]:
    prepared = prepare_case(Path(args.case), normalize_tools=args.normalize_tools)
    events: List[Dict[str, Any]] = []
    api_calls: List[Dict[str, Any]] = []
    train_to_api: Dict[str, str] = {}
    sent_tool_results: List[Dict[str, Any]] = []

    tool_responses_by_unit: Dict[int, List[Dict[str, Any]]] = {
        int(unit): list(items)
        for unit, items in prepared["tool_responses_by_unit"].items()
    }
    delayed_gt_responses: List[Dict[str, Any]] = []

    async def send_tool_result(ws: Any, *, api_id: str, content: str, unit_index: int, source: str, train_id: Optional[str]) -> None:
        payload = {
            "type": "input.append",
            "input": {
                "type": "tool_result",
                "tool_call_id": api_id,
                "contents": content,
                "content": content,
            },
        }
        await ws.send(json.dumps(payload, ensure_ascii=False))
        sent_tool_results.append(
            {
                "source": source,
                "unit_index": unit_index,
                "api_tool_call_id": api_id,
                "train_tool_call_id": train_id,
                "content": content,
            }
        )

    async def send_due_gt_tool_results(ws: Any, unit_index: int) -> None:
        due = delayed_gt_responses + tool_responses_by_unit.pop(unit_index, [])
        delayed_gt_responses.clear()
        for item in due:
            train_id = item.get("train_tool_call_id")
            api_id = train_to_api.get(str(train_id))
            if not api_id:
                delayed_gt_responses.append(item)
                continue
            await send_tool_result(
                ws,
                api_id=api_id,
                content=str(item.get("content", "")),
                unit_index=unit_index,
                source="gt_unit",
                train_id=str(train_id),
            )

    target_url = ws_url(args.backend, args.path)
    ssl_ctx = None
    if target_url.startswith("wss://"):
        ssl_ctx = ssl._create_unverified_context() if args.insecure else ssl.create_default_context()

    session_id: Optional[str] = None
    async with websockets.connect(target_url, max_size=128 * 1024 * 1024, ping_interval=None, ssl=ssl_ctx) as ws:
        config = {
            "sample_rate": prepared["sample_rate"],
            "unit_sec": prepared["unit_sec"],
            "decode_mode": args.decode_mode,
            "max_spoken_tokens": args.max_spoken_tokens,
            "non_spoken_scheduling": "quality",
        }
        init_payload = {
            "type": "session.init",
            "payload": {
                "mode": "full_duplex",
                "fc_duplex": True,
                "protocol_version": "3",
                "tokenizer_target": "o5",
                "system": prepared["system"],
                "generate_audio": bool(args.generate_audio),
                "tts_prompt_audio": (
                    {
                        "source": "path",
                        "file_path": args.ref_audio_path or prepared["ref_audio_path"],
                    }
                    if args.generate_audio
                    and (args.ref_audio_path or prepared["ref_audio_path"])
                    else None
                ),
                "unit_policy": prepared["structure"].get("unit_policy") or {
                    "unit_sec": prepared["unit_sec"],
                    "non_spoken_budgets_while_listening": prepared["budgets_listening"],
                    "non_spoken_budgets_while_speaking": prepared["budgets_speaking"],
                },
                "config": config,
            },
        }
        await ws.send(json.dumps(init_payload, ensure_ascii=False))
        created = json.loads(await ws.recv())
        events.append(created)
        session_id = str(created.get("session_id") or "") or None

        pending_auto_results: List[Dict[str, Any]] = []

        async def handle_event(event: Dict[str, Any]) -> None:
            events.append(event)
            event_type = event.get("type")
            # Semantic v3 emits the structured call at ``done``.  Keep the
            # legacy raw event for older backends, but record a call only once
            # when a backend happens to expose both forms.
            if event_type == "response.tool_call.done":
                raw_call = event.get("call") or {}
            elif event_type == "response.tool_call.args.raw":
                raw_call = event.get("raw") or {}
            else:
                return
            normalized = normalize_raw_tool_call(raw_call)
            api_id = str(event.get("tool_call_id") or "")
            if api_id and any(
                call.get("api_tool_call_id") == api_id for call in api_calls
            ):
                return
            call_index = len(api_calls)
            train_id = (
                prepared["train_tool_call_ids"][call_index]
                if call_index < len(prepared["train_tool_call_ids"])
                else None
            )
            if train_id and api_id:
                train_to_api[str(train_id)] = api_id
            api_calls.append(
                {
                    "index": call_index,
                    "input_id": event.get("input_id"),
                    "unit_index": (
                        int(event["unit_index"])
                        if event.get("unit_index") is not None
                        else unit_index_from_input_id(event.get("input_id"))
                    ),
                    "api_tool_call_id": api_id,
                    "train_tool_call_id": train_id,
                    "raw": raw_call,
                    **normalized,
                }
            )
            if args.tool_response_schedule == "auto" and train_id:
                response_content = None
                for items in prepared["tool_responses_by_unit"].values():
                    for item in items:
                        if item.get("train_tool_call_id") == train_id:
                            response_content = item.get("content", "")
                            break
                    if response_content is not None:
                        break
                if response_content is not None:
                    call_unit = unit_index_from_input_id(event.get("input_id"))
                    target_unit = (call_unit if call_unit is not None else 0) + args.auto_delay_units
                    pending_auto_results.append(
                        {
                            "target_unit": target_unit,
                            "api_tool_call_id": api_id,
                            "train_tool_call_id": train_id,
                            "content": str(response_content),
                        }
                    )

        total_chunks = list(prepared["unit_chunks"])
        if args.extra_silence_units > 0:
            samples_per_unit = max(1, int(round(prepared["sample_rate"] * prepared["unit_sec"])))
            silence = np.zeros(samples_per_unit, dtype=np.float32)
            total_chunks.extend([silence for _ in range(args.extra_silence_units)])

        for unit_index, samples in enumerate(total_chunks):
            if args.tool_response_schedule == "gt":
                await send_due_gt_tool_results(ws, unit_index)
            else:
                remaining: List[Dict[str, Any]] = []
                for item in pending_auto_results:
                    if int(item["target_unit"]) <= unit_index:
                        await send_tool_result(
                            ws,
                            api_id=str(item["api_tool_call_id"]),
                            content=str(item.get("content", "")),
                            unit_index=unit_index,
                            source="auto_delay",
                            train_id=str(item.get("train_tool_call_id")),
                        )
                    else:
                        remaining.append(item)
                pending_auto_results = remaining

            payload = {
                "type": "input.append",
                "input": {
                    "type": "audio",
                    "input_id": f"unit_{unit_index:03d}",
                    "audio": encode_float32_audio(samples),
                    "sample_rate": prepared["sample_rate"],
                },
            }
            await ws.send(json.dumps(payload, ensure_ascii=False))
            await drain_until_unit_done(ws, handle_event, unit_index=unit_index, timeout_s=args.unit_timeout)

        final_deadline = asyncio.get_running_loop().time() + args.final_max_wait
        while asyncio.get_running_loop().time() < final_deadline:
            if args.tool_response_schedule == "gt" and delayed_gt_responses:
                await send_due_gt_tool_results(ws, len(total_chunks))
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=args.final_idle_timeout)
            except asyncio.TimeoutError:
                break
            await handle_event(json.loads(raw))

        # The backend WebSocket protocol intentionally does not accept
        # session.close. Close through its HTTP control endpoint so the
        # server can flush the model trace and perform rank-local cleanup.
        if session_id:
            close_url = args.backend.rstrip("/") + f"/sessions/{session_id}/close"
            request = urllib.request.Request(
                close_url,
                data=json.dumps({"reason": "probe_done"}).encode("utf-8"),
                headers={"content-type": "application/json"},
                method="POST",
            )
            with contextlib.suppress(urllib.error.URLError, OSError):
                await asyncio.to_thread(urllib.request.urlopen, request, timeout=10)

    spoken_text_parts: List[str] = []
    spoken_audio_parts: List[bytes] = []
    for event in events:
        if event.get("type") != "response.spoken.delta":
            continue
        for step in event.get("steps", []) or []:
            if step.get("kind") == "text":
                spoken_text_parts.append(str(step.get("text", "")))
        audio = event.get("audio")
        if audio:
            spoken_audio_parts.append(base64.b64decode(str(audio)))
    spoken_audio = b"".join(spoken_audio_parts)
    spoken_text = "".join(spoken_text_parts)
    think_text = "".join(
        str(event.get("delta") or "")
        for event in events
        if event.get("type") == "response.think.delta"
    )
    summary = {
        "case": prepared["case"],
        "events": len(events),
        "units_sent": len(prepared["unit_chunks"]) + int(args.extra_silence_units),
        "gt_tool_calls": prepared["gt_tool_calls"],
        "api_tool_calls": api_calls,
        "gt_tool_semantic": semantic_tool_calls(prepared["gt_tool_calls"]),
        "api_tool_semantic": semantic_tool_calls(api_calls),
        "tool_calls_semantic_exact": semantic_tool_calls(prepared["gt_tool_calls"]) == semantic_tool_calls(api_calls),
        "train_to_api_tool_call_ids": train_to_api,
        "sent_tool_results": sent_tool_results,
        "unsent_gt_tool_responses": delayed_gt_responses,
        "remaining_tool_responses_by_unit": tool_responses_by_unit,
        "spoken_text": spoken_text,
        "think_text": think_text,
        "spoken_audio_events": len(spoken_audio_parts),
        "spoken_audio_bytes": len(spoken_audio),
        "spoken_audio_sha256": (
            hashlib.sha256(spoken_audio).hexdigest()
            if spoken_audio
            else None
        ),
        "spoken_audio_nonzero_bytes": sum(byte != 0 for byte in spoken_audio),
        "config": {
            "backend": args.backend,
            "path": args.path,
            "decode_mode": args.decode_mode,
            "tool_response_schedule": args.tool_response_schedule,
            "auto_delay_units": args.auto_delay_units,
            "generate_audio": args.generate_audio,
            "use_case_ref_audio": args.use_case_ref_audio,
            "ref_audio_path": args.ref_audio_path or prepared["ref_audio_path"],
        },
    }
    return {"summary": summary, "events": events}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe board FC duplex backend with train-data unit scheduling")
    parser.add_argument("--backend", default="http://127.0.0.1:22514")
    parser.add_argument("--path", default="/backend")
    parser.add_argument("--insecure", action="store_true")
    parser.add_argument("--case", default=DEFAULT_CASE)
    parser.add_argument("--out", default="/user/weihongliang/fc_board_api_runs/board_same_schedule_04702.json")
    parser.add_argument("--normalize-tools", action="store_true")
    parser.add_argument("--decode-mode", default="greedy")
    parser.add_argument("--max-spoken-tokens", type=int, default=24)
    parser.add_argument("--non-spoken-budget", type=int, default=12)
    parser.add_argument("--tool-response-schedule", choices=("gt", "auto"), default="gt")
    parser.add_argument("--auto-delay-units", type=int, default=2)
    parser.add_argument("--extra-silence-units", type=int, default=0)
    parser.add_argument("--unit-timeout", type=float, default=120.0)
    parser.add_argument("--final-idle-timeout", type=float, default=20.0)
    parser.add_argument("--final-max-wait", type=float, default=180.0)
    parser.add_argument("--generate-audio", action="store_true")
    parser.add_argument("--use-case-ref-audio", action="store_true")
    parser.add_argument(
        "--ref-audio-path",
        help="Explicit TTS reference WAV; useful for cases without a system audio segment.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    if args.dry_run:
        prepared = prepare_case(Path(args.case), normalize_tools=args.normalize_tools)
        payload = {
            "case": prepared["case"],
            "unit_sec": prepared["unit_sec"],
            "sample_rate": prepared["sample_rate"],
            "units": len(prepared["unit_chunks"]),
            "budgets_listening_head": prepared["budgets_listening"][:20],
            "budgets_speaking_head": prepared["budgets_speaking"][:20],
            "tool_responses_by_unit": prepared["tool_responses_by_unit"],
            "train_tool_call_ids": prepared["train_tool_call_ids"],
            "gt_tool_calls": prepared["gt_tool_calls"],
            "ref_audio_path": prepared["ref_audio_path"],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    result = await run_probe(args)
    out_path = Path(args.out)
    write_json(out_path, result)
    print(json.dumps({"out": str(out_path), **result["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
