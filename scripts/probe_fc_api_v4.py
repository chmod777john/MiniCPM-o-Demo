"""Small real-case gateway probe with native events and model-triggered tool results."""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
from importlib.metadata import version
from pathlib import Path

import websockets
from minicpm_o5_sdk import O5DuplexTrainingData

from scripts.fc_api_eval.models import FcApiCheckpointProfile
from scripts.fc_api_eval.training_data_scenario import build_training_data_scenario


def prepare(args):
    if version("minicpm-o5-sdk") != "0.0.6":
        raise RuntimeError("Official SDK 0.0.6 is required")
    case = Path(args.case)
    training = O5DuplexTrainingData.load_structure(
        json.loads(case.read_text()), data_root=case.parent,
    )
    profile = FcApiCheckpointProfile(
        profile_id=args.profile_id, model=args.model,
        checkpoint_path=Path(args.checkpoint), training_job="gateway-v4-probe",
        sdk_version="0.0.6", tokenizer_target="o5",
    )
    scenario = build_training_data_scenario(training_data=training, profile=profile, data_root=case.parent)
    payload = scenario.session_init["payload"]
    payload.pop("evaluation", None)
    payload["generate_audio"] = args.generate_audio
    if args.tts_pad_protocol:
        payload["config"]["tts_pad_protocol"] = args.tts_pad_protocol
    payload["config"]["decoding_policy"] = args.decoding_policy
    assert payload["protocol_version"] == "4"
    return scenario


async def run(args, scenario, trace):
    url = args.base_url.rstrip("/").replace("https://", "wss://").replace("http://", "ws://")
    url += "/v1/realtime?mode=audio"
    trace["gateway_url"] = url
    pending = []
    result_by_id = {r.tool_call_id: r.content for r in scenario.tool_responses}
    matched = set()
    async with websockets.connect(url, max_size=128 * 1024 * 1024) as ws:
        async def send(event):
            trace["events"].append({"direction": "up", "event": event})
            await ws.send(json.dumps(event))
        async def receive():
            event = json.loads(await asyncio.wait_for(ws.recv(), args.timeout))
            trace["events"].append({"direction": "down", "event": event})
            if event.get("type") in {"error", "session.resume.failed"}:
                raise RuntimeError(str(event))
            return event
        while (await receive()).get("type") != "session.queue_done":
            pass
        await send(scenario.session_init)
        while True:
            event = await receive()
            if event.get("type") == "session.closed":
                raise RuntimeError(str(event))
            if event.get("type") == "session.created":
                break
        for unit in scenario.units[:args.max_units]:
            for result in pending:
                await send(result)
            pending.clear()
            await send(unit.to_frame())
            while True:
                event = await receive()
                kind = event.get("type")
                if kind == "session.closed":
                    raise RuntimeError(str(event))
                if kind == "response.tool_call.done":
                    trace["model_tool_calls"].append(event)
                    call = event.get("tool_call") or event.get("call") or event
                    name = call.get("name") or (call.get("function") or {}).get("name")
                    candidates = [c for c in scenario.expected_tool_calls
                                  if c.name == name and c.tool_call_id not in matched
                                  and c.tool_call_id in result_by_id]
                    if candidates:
                        gt = candidates[0]
                        actual_id = event.get("tool_call_id") or call.get("id")
                        if not actual_id:
                            raise RuntimeError("Model tool call has no returned ID")
                        matched.add(gt.tool_call_id)
                        pending.append({"type": "input.tool_result", "tool_call_id": actual_id,
                                        "content": result_by_id[gt.tool_call_id]})
                        trace["autonomous_tool_calls"].append({"actual_id": actual_id, "gt_id": gt.tool_call_id})
                if kind == "response.unit.committed":
                    assert event["unit_index"] == unit.unit_index, event
                    trace["committed_units"].append(unit.unit_index)
                    break
        trace["tool_results_sent"] = sum(
            item["direction"] == "up" and item["event"].get("type") == "input.tool_result"
            for item in trace["events"]
        )
        trace["pending_tool_results"] = len(pending)
        trace["downstream_event_counts"] = {}
        for item in trace["events"]:
            if item["direction"] == "down":
                kind = item["event"].get("type", "unknown")
                trace["downstream_event_counts"][kind] = trace["downstream_event_counts"].get(kind, 0) + 1
        trace["audio_return_bytes"] = sum(
            len(base64.b64decode(item["event"]["audio"]))
            for item in trace["events"] if item["direction"] == "down"
            and item["event"].get("type") == "response.spoken.delta"
            and isinstance(item["event"].get("audio"), str)
        )
        trace["acceptance"] = {
            "all_requested_units_committed": len(trace["committed_units"]) == min(args.max_units, len(scenario.units)),
            "autonomous_fc_and_result_queue": bool(trace["autonomous_tool_calls"]) and trace["tool_results_sent"] > 0,
            "audio_returned": trace["audio_return_bytes"] > 0,
        }
        await send({"type": "session.close"})
        while (await receive()).get("type") != "session.closed":
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8009")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--profile-id", default="unprofiled")
    parser.add_argument("--max-units", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--generate-audio", action="store_true")
    parser.add_argument("--tts-pad-protocol", choices=["legacy", "revised"])
    parser.add_argument("--decoding-policy", default="raise_protocol_error")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    trace = {"events": [], "committed_units": [], "autonomous_tool_calls": [], "model_tool_calls": [],
             "protocol_version": "4", "case": args.case, "sdk_version": version("minicpm-o5-sdk")}
    try:
        scenario = prepare(args)
        trace["fixture"] = {"session_init": scenario.session_init,
                            "units": [u.to_frame() for u in scenario.units[:args.max_units]]}
        trace["available_units"] = len(scenario.units)
        trace["expected_tool_calls"] = [c.model_dump() for c in scenario.expected_tool_calls]
        if not args.dry_run:
            asyncio.run(run(args, scenario, trace))
        trace["status"] = "dry_run" if args.dry_run else "completed"
    except Exception as exc:
        trace["status"] = "failed"
        trace["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(trace, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
