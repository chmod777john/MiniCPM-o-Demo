#!/usr/bin/env python3
"""Compare aligned model events from two O5 session trace bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import wave
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.tracing import load_session_trace_events  # noqa: E402


TENSOR_FIELDS = {
    "llm.feed": ("embeds", "hidden", "logits", "layers"),
    "llm.decode": ("logits",),
    "tts.condition": ("actual_condition", "used_condition"),
    "tts.forward": ("hidden", "logits"),
    "tts.sample": ("probabilities",),
    "tts.chunk": ("new_tokens",),
    "token2wav.call": ("output_pcm", "output_waveform"),
    "vocoder.state": ("rand_noise",),
}
TOKEN_FIELDS = {
    "llm.decode": ("selected_token_id", "local_argmax_token_id"),
    "llm.accepted": ("token_ids",),
    "llm.chunk": ("token_ids", "is_listen", "end_of_turn"),
    "tts.condition": ("llm_token_ids", "end_of_turn"),
    "tts.sample": ("selected_token_ids",),
    "tts.chunk": ("new_tokens.tokens", "token_ids", "source_llm_token_ids"),
    "token2wav.call": (
        "input_token_ids",
        "input_range",
        "committed_range",
        "lookahead_range",
        "output_sample_range",
        "last_chunk",
        "state_before",
        "state_after",
    ),
    "t2w.chunk": (
        "input_token_ids",
        "input_range",
        "committed_range",
        "lookahead_range",
        "output_sample_range",
        "last_chunk",
    ),
}


def _event_key(events: Iterable[dict[str, Any]]) -> dict[tuple[str, str, int], dict[str, Any]]:
    counters: dict[tuple[str, str], int] = defaultdict(int)
    indexed = {}
    for event in events:
        kind = str(event.get("kind") or "")
        input_id = str(event.get("input_id") or "")
        prefix = (kind, input_id)
        ordinal = counters[prefix]
        counters[prefix] += 1
        indexed[(kind, input_id, ordinal)] = event
    return indexed


def _field(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict) and torch.is_tensor(value.get("_tensor")):
        return value["_tensor"]
    return None


def _tensor_metric(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    if tuple(left.shape) != tuple(right.shape):
        return {"shape_match": False, "left_shape": list(left.shape), "right_shape": list(right.shape)}
    shape = list(left.shape)
    bitwise = left.dtype == right.dtype and torch.equal(left, right)
    a = left.detach().float().reshape(-1)
    b = right.detach().float().reshape(-1)
    delta = a - b
    rms = float(torch.sqrt(torch.mean(delta.square()))) if delta.numel() else 0.0
    base_rms = float(torch.sqrt(torch.mean(a.square()))) if a.numel() else 0.0
    denominator = max(base_rms, 1e-12)
    if a.numel() and float(torch.linalg.vector_norm(a)) and float(torch.linalg.vector_norm(b)):
        cosine = float(torch.nn.functional.cosine_similarity(a, b, dim=0))
    else:
        cosine = 1.0 if torch.equal(a, b) else 0.0
    return {
        "shape_match": True,
        "shape": shape,
        "left_dtype": str(left.dtype),
        "right_dtype": str(right.dtype),
        "bitwise": bitwise,
        "max_abs": float(delta.abs().max()) if delta.numel() else 0.0,
        "rms": rms,
        "relative_rms": rms / denominator,
        "cosine": cosine,
    }


def _probability_metric(
    left: torch.Tensor,
    right: torch.Tensor,
    *,
    logits: bool,
) -> dict[str, float]:
    if tuple(left.shape) != tuple(right.shape):
        return {}
    p = left.detach().float().reshape(-1, left.shape[-1])
    q = right.detach().float().reshape(-1, right.shape[-1])
    if logits:
        p = torch.softmax(p, dim=-1)
        q = torch.softmax(q, dim=-1)
    else:
        p = p.clamp_min(1e-30)
        q = q.clamp_min(1e-30)
        p = p / p.sum(dim=-1, keepdim=True)
        q = q / q.sum(dim=-1, keepdim=True)
    p = p.clamp_min(1e-30)
    q = q.clamp_min(1e-30)
    kl = (p * (p.log() - q.log())).sum(dim=-1)
    tv = 0.5 * (p - q).abs().sum(dim=-1)
    return {
        "kl_left_right_mean": float(kl.mean()),
        "kl_left_right_max": float(kl.max()),
        "tv_mean": float(tv.mean()),
        "tv_max": float(tv.max()),
        "argmax_equal": int(p.argmax(dim=-1).eq(q.argmax(dim=-1)).sum()),
        "argmax_total": int(p.shape[0]),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("shape_match")]
    if not valid:
        return {"count": len(rows), "shape_matches": 0}
    return {
        "count": len(rows),
        "shape_matches": len(valid),
        "bitwise_count": sum(bool(row.get("bitwise")) for row in valid),
        "max_abs": max(float(row.get("max_abs", 0.0)) for row in valid),
        "relative_rms_mean": sum(float(row.get("relative_rms", 0.0)) for row in valid) / len(valid),
        "relative_rms_max": max(float(row.get("relative_rms", 0.0)) for row in valid),
        "cosine_mean": sum(float(row.get("cosine", 0.0)) for row in valid) / len(valid),
        "cosine_min": min(float(row.get("cosine", 0.0)) for row in valid),
        "tv_mean": _mean(row.get("tv_mean") for row in valid),
        "tv_max": _max(row.get("tv_max") for row in valid),
        "kl_left_right_mean": _mean(row.get("kl_left_right_mean") for row in valid),
        "kl_left_right_max": _max(row.get("kl_left_right_max") for row in valid),
        "argmax_equal": sum(int(row.get("argmax_equal", 0)) for row in valid),
        "argmax_total": sum(int(row.get("argmax_total", 0)) for row in valid),
        "argmax_reversals": sum(
            int(row.get("argmax_total", 0)) - int(row.get("argmax_equal", 0))
            for row in valid
        ),
    }


def _numbers(values: Iterable[Any]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def _mean(values: Iterable[Any]) -> float | None:
    rows = _numbers(values)
    return sum(rows) / len(rows) if rows else None


def _max(values: Iterable[Any]) -> float | None:
    rows = _numbers(values)
    return max(rows) if rows else None


def _read_pcm16(path: Path) -> tuple[torch.Tensor, dict[str, Any], str]:
    raw = Path(path).read_bytes()
    with wave.open(str(path), "rb") as handle:
        metadata = {
            "sample_rate": handle.getframerate(),
            "channels": handle.getnchannels(),
            "sample_width": handle.getsampwidth(),
            "frames": handle.getnframes(),
        }
        if metadata["sample_width"] != 2:
            raise ValueError(f"only PCM16 WAV is supported: {path}")
        pcm = handle.readframes(metadata["frames"])
    samples = torch.frombuffer(bytearray(pcm), dtype=torch.int16).clone()
    return samples, metadata, hashlib.sha256(raw).hexdigest()


def _audio_file_metric(left: Path, right: Path) -> dict[str, Any]:
    a, left_meta, left_sha = _read_pcm16(left)
    b, right_meta, right_sha = _read_pcm16(right)
    metric = _tensor_metric(a.float() / 32768.0, b.float() / 32768.0)
    return {
        **metric,
        "pcm16_bitwise": torch.equal(a, b),
        "format_match": left_meta == right_meta,
        "left_metadata": left_meta,
        "right_metadata": right_meta,
        "left_sha256": left_sha,
        "right_sha256": right_sha,
    }


def _audio_report(left_root: Path, right_root: Path) -> dict[str, Any]:
    left_root = Path(left_root)
    right_root = Path(right_root)
    report: dict[str, Any] = {}
    left_combined = left_root / "output_audio.wav"
    right_combined = right_root / "output_audio.wav"
    if left_combined.is_file() and right_combined.is_file():
        report["combined"] = _audio_file_metric(left_combined, right_combined)

    left_units = {path.name: path for path in (left_root / "audio").glob("*.wav")}
    right_units = {path.name: path for path in (right_root / "audio").glob("*.wav")}
    common = sorted(set(left_units) & set(right_units))
    rows = [
        {"file": name, **_audio_file_metric(left_units[name], right_units[name])}
        for name in common
    ]
    report["units"] = {
        "left": len(left_units),
        "right": len(right_units),
        "common": len(common),
        "left_only": sorted(set(left_units) - set(right_units)),
        "right_only": sorted(set(right_units) - set(left_units)),
        "summary": _aggregate(rows),
        "metrics": rows,
    }
    return report


def compare(left_root: Path, right_root: Path) -> dict[str, Any]:
    left_events = load_session_trace_events(left_root, load_tensors=True)
    right_events = load_session_trace_events(right_root, load_tensors=True)
    left = _event_key(left_events)
    right = _event_key(right_events)
    common = sorted(set(left) & set(right))
    tensor_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    token_rows: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "equal": 0})
    first_mismatches: list[dict[str, Any]] = []

    for key in common:
        kind, input_id, ordinal = key
        a, b = left[key], right[key]
        for field in TOKEN_FIELDS.get(kind, ()):
            av, bv = _field(a, field), _field(b, field)
            if av is None and bv is None:
                continue
            name = f"{kind}.{field}"
            token_rows[name]["count"] += 1
            if av == bv:
                token_rows[name]["equal"] += 1
            elif len(first_mismatches) < 50:
                first_mismatches.append({
                    "event": [kind, input_id, ordinal],
                    "field": field,
                    "left": av,
                    "right": bv,
                })
        for field in TENSOR_FIELDS.get(kind, ()):
            left_values = _field(a, field)
            right_values = _field(b, field)
            pairs = zip(left_values, right_values) if isinstance(left_values, list) and isinstance(right_values, list) else [(left_values, right_values)]
            for index, (av, bv) in enumerate(pairs):
                at, bt = _tensor(av), _tensor(bv)
                if at is None or bt is None:
                    continue
                name = f"{kind}.{field}" + (f"[{index}]" if isinstance(left_values, list) else "")
                metric = _tensor_metric(at, bt)
                if kind in {"llm.decode", "tts.forward", "tts.sample"} and field in {"logits", "probabilities"}:
                    metric.update(_probability_metric(at, bt, logits=field == "logits"))
                tensor_rows[name].append(metric)

    return {
        "schema": "o5.session-trace-comparison.v1",
        "left": str(Path(left_root).resolve()),
        "right": str(Path(right_root).resolve()),
        "event_counts": {
            "left": len(left),
            "right": len(right),
            "common": len(common),
            "left_only": len(set(left) - set(right)),
            "right_only": len(set(right) - set(left)),
        },
        "tokens": dict(token_rows),
        "tensors": {name: _aggregate(rows) for name, rows in sorted(tensor_rows.items())},
        "audio": _audio_report(Path(left_root), Path(right_root)),
        "first_token_mismatches": first_mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left")
    parser.add_argument("right")
    parser.add_argument("--out")
    args = parser.parse_args()
    report = compare(Path(args.left), Path(args.right))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
