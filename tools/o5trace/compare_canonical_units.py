#!/usr/bin/env python3
"""Compare two canonical_units.json files from O5 duplex probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


FIELDS = (
    "is_listen",
    "text",
    "end_of_turn",
    "n_tokens",
    "n_tts_tokens",
)

AUDIO_FIELDS = (
    "samples",
    "duration_s",
    "sha256",
    "max_abs",
)


def units_path(path: str) -> Path:
    p = Path(path)
    if p.is_dir():
        p = p / "canonical_units.json"
    return p


def load_units(path: str) -> list[dict[str, Any]]:
    p = units_path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"expected list in {p}")
    return data


def equal_float(a: Any, b: Any, tolerance: float) -> bool:
    try:
        return abs(float(a) - float(b)) <= tolerance
    except Exception:
        return a == b


def compare_units(
    a_units: list[dict[str, Any]],
    b_units: list[dict[str, Any]],
    *,
    float_tolerance: float,
    compare_prefix: bool,
) -> dict[str, Any]:
    limit = min(len(a_units), len(b_units))
    diffs: list[dict[str, Any]] = []

    if len(a_units) != len(b_units) and not compare_prefix:
        diffs.append({
            "unit_id": None,
            "field": "length",
            "a": len(a_units),
            "b": len(b_units),
        })

    for idx in range(limit):
        a = a_units[idx]
        b = b_units[idx]
        unit_id = a.get("unit_id", idx)
        for field in FIELDS:
            if a.get(field) != b.get(field):
                diffs.append({
                    "unit_id": unit_id,
                    "field": field,
                    "a": a.get(field),
                    "b": b.get(field),
                })
        a_audio = a.get("audio") or {}
        b_audio = b.get("audio") or {}
        for field in AUDIO_FIELDS:
            av = a_audio.get(field)
            bv = b_audio.get(field)
            if field in {"duration_s", "max_abs"}:
                equal = equal_float(av, bv, float_tolerance)
            else:
                equal = av == bv
            if not equal:
                diffs.append({
                    "unit_id": unit_id,
                    "field": f"audio.{field}",
                    "a": av,
                    "b": bv,
                })

    return {
        "equal": not diffs,
        "num_a": len(a_units),
        "num_b": len(b_units),
        "first_diff": diffs[0] if diffs else None,
        "diff_count": len(diffs),
        "diffs": diffs,
        "text_a": "".join(str(item.get("text") or "") for item in a_units),
        "text_b": "".join(str(item.get("text") or "") for item in b_units),
    }


def write_markdown(result: dict[str, Any], path: Path, *, a_label: str, b_label: str) -> None:
    lines = [
        "# Canonical Units Compare",
        "",
        f"- A: `{a_label}`",
        f"- B: `{b_label}`",
        f"- equal: `{result['equal']}`",
        f"- units: `{result['num_a']}` vs `{result['num_b']}`",
        f"- diff_count: `{result['diff_count']}`",
        "",
        "## Text",
        "",
        "A:",
        "",
        "```text",
        result["text_a"],
        "```",
        "",
        "B:",
        "",
        "```text",
        result["text_b"],
        "```",
        "",
    ]
    if result["first_diff"] is not None:
        lines.extend([
            "## First Diff",
            "",
            "```json",
            json.dumps(result["first_diff"], ensure_ascii=False, indent=2),
            "```",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a")
    parser.add_argument("b")
    parser.add_argument("--out-json", default=None)
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--float-tolerance", type=float, default=1e-7)
    parser.add_argument(
        "--compare-prefix",
        action="store_true",
        help="Ignore total length mismatch and compare only the shared prefix.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = compare_units(
        load_units(args.a),
        load_units(args.b),
        float_tolerance=args.float_tolerance,
        compare_prefix=args.compare_prefix,
    )
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.out_md:
        write_markdown(result, Path(args.out_md), a_label=args.a, b_label=args.b)
    print(json.dumps({k: v for k, v in result.items() if k != "diffs"}, ensure_ascii=False, indent=2))
    return 0 if result["equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
