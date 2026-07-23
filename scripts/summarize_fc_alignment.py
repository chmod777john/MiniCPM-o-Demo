#!/usr/bin/env python3
"""Summarize FC offline-vs-live unit behavior for one probe case."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: str) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _tool_name_args(call: dict[str, Any]) -> str:
    raw = call.get("raw") or call
    name = raw.get("name") or (call.get("tool_call") or {}).get("name")
    args = raw.get("arguments") or (call.get("tool_call") or {}).get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            pass
    return f"{name}({args})"


def _offline_units(path: str) -> list[dict[str, Any]]:
    units = _load(path)
    rows: list[dict[str, Any]] = []
    for unit in units:
        spans = unit.get("closed_spans") or []
        tool_calls = []
        for span in spans:
            if span.get("type") == "tool_call":
                tool_calls.append(_tool_name_args(span))
        rows.append(
            {
                "unit": unit.get("unit"),
                "listen": bool(unit.get("is_listen")),
                "speak": bool(unit.get("is_speaking")),
                "spoken_tokens": len(unit.get("spoken_ids") or []),
                "non_spoken_tokens": len(unit.get("non_spoken_ids") or []),
                "non_spoken_terminator": unit.get("non_spoken_terminator"),
                "tool_calls": tool_calls,
            }
        )
    return rows


def _live_rows(path: str) -> tuple[list[dict[str, Any]], str, list[str]]:
    events = _load(path).get("events") or []
    by_input: dict[str, dict[str, Any]] = {}
    tool_calls: list[str] = []
    spoken_parts: list[str] = []
    for event in events:
        input_id = event.get("input_id")
        if input_id:
            row = by_input.setdefault(
                input_id,
                {"input_id": input_id, "listen": False, "speak_text": [], "tool_calls": [], "debug_steps": 0},
            )
        else:
            row = None
        if event.get("type") == "response.output.delta":
            if event.get("kind") == "listen" and row is not None:
                row["listen"] = True
            if event.get("kind") == "text":
                text = event.get("text") or event.get("delta") or ""
                spoken_parts.append(text)
                if row is not None:
                    row["speak_text"].append(text)
        elif event.get("type") == "response.tool_call.args.raw":
            desc = _tool_name_args(event)
            tool_calls.append(desc)
            if row is not None:
                row["tool_calls"].append(desc)
        elif event.get("type") == "debug.fc_non_spoken.delta" and row is not None:
            row["debug_steps"] += 1
    rows = []
    for key in sorted(by_input.keys()):
        row = by_input[key]
        rows.append(
            {
                "input_id": key,
                "listen": row["listen"],
                "speak_text": "".join(row["speak_text"]),
                "tool_calls": row["tool_calls"],
                "debug_steps": row["debug_steps"],
            }
        )
    return rows, "".join(spoken_parts), tool_calls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline-units", required=True)
    parser.add_argument("--offline-comparison", required=True)
    parser.add_argument("--live-events", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    comparison = _load(args.offline_comparison)
    offline_rows = _offline_units(args.offline_units)
    live_rows, live_spoken, live_tool_calls = _live_rows(args.live_events)
    summary = {
        "offline": {
            "success": comparison.get("success"),
            "comparison": comparison.get("comparison"),
            "pred_tool_calls": [_tool_name_args(call) for call in comparison.get("pred_tool_calls") or []],
            "pred_spoken_text": comparison.get("pred_spoken_text"),
            "tool_units": [row for row in offline_rows if row["tool_calls"]],
            "units": offline_rows,
        },
        "live": {
            "tool_calls": live_tool_calls,
            "spoken_text": live_spoken,
            "tool_units": [row for row in live_rows if row["tool_calls"]],
            "units": live_rows,
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "out": str(out),
        "offline_tool_units": summary["offline"]["tool_units"],
        "live_tool_units": summary["live"]["tool_units"],
        "offline_spoken_text": summary["offline"]["pred_spoken_text"],
        "live_spoken_text": summary["live"]["spoken_text"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
