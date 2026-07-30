#!/usr/bin/env python3
"""Run a recorded session canonical replay several times in parallel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def recorded_duplex_config(session_dir: Path) -> dict[str, Any]:
    events = read_jsonl(session_dir / "stream.jsonl")
    created = next(
        event for event in events if (event.get("frame") or {}).get("type") == "session.created"
    )
    return created["frame"]["replay_manifest"]["resolved"]["duplex_config"]


def expected_generate_subset(recorded: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "max_new_speak_tokens_per_chunk",
        "decode_mode",
        "temperature",
        "top_k",
        "top_p",
        "listen_prob_scale",
        "listen_top_k",
        "text_repetition_penalty",
        "text_repetition_window_size",
    )
    return {key: recorded.get(key) for key in keys}


def expected_duplex_subset(recorded: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "generate_audio",
        "max_new_speak_tokens_per_chunk",
        "text_repetition_penalty",
        "temperature",
        "top_k",
        "top_p",
        "text_repetition_window_size",
        "listen_prob_scale",
        "force_listen_count",
        "tts_temperature",
        "tts_repetition_penalty",
        "chunk_ms",
        "sample_rate",
    )
    return {key: recorded.get(key) for key in keys}


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def summarize_case(case: int, out_dir: Path, expected_generate: dict[str, Any], expected_duplex: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"case": case, "out_dir": str(out_dir)}
    exit_path = out_dir.parent / f"case_{case}.exit"
    row["exit"] = int(exit_path.read_text().strip()) if exit_path.exists() else None

    summary_path = out_dir / "summary.json"
    row["summary_exists"] = summary_path.exists()
    if not summary_path.exists():
        return row

    summary = json.loads(summary_path.read_text())
    units = summary.get("units") or []
    row["units"] = len(units)
    row["text"] = "".join(unit.get("text") or "" for unit in units)
    row["generate_kwargs"] = summary.get("generate_kwargs")
    row["duplex_kwargs"] = summary.get("duplex_kwargs")

    actual_generate = {
        key: (summary.get("generate_kwargs") or {}).get(key)
        for key in expected_generate
    }
    actual_duplex = {
        key: (summary.get("duplex_kwargs") or {}).get(key)
        for key in expected_duplex
    }
    row["generate_matches_recorded"] = actual_generate == expected_generate
    row["duplex_matches_recorded"] = actual_duplex == expected_duplex

    wav_path = out_dir / "continuous.wav"
    if wav_path.exists():
        row["continuous_sha256"] = hashlib.sha256(wav_path.read_bytes()).hexdigest()
        row["duration_s"] = wav_duration(wav_path)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--base-out-dir", required=True)
    parser.add_argument("--canonical-root", required=True)
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--token2wav-dir", required=True)
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--num-cases", type=int, default=8)
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    session_dir = Path(args.session_dir).resolve()
    base_out_dir = Path(args.base_out_dir).resolve()
    base_out_dir.mkdir(parents=True, exist_ok=True)

    recorded = recorded_duplex_config(session_dir)
    expected_generate = expected_generate_subset(recorded)
    expected_duplex = expected_duplex_subset(recorded)

    script = Path(__file__).with_name("session_canonical_replay.py")
    procs: list[tuple[int, Path, subprocess.Popen[bytes]]] = []
    for case in range(args.num_cases):
        out_dir = base_out_dir / f"case_{case}"
        log_path = base_out_dir / f"case_{case}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(case)
        cmd = [
            args.python,
            str(script),
            "--session-dir",
            str(session_dir),
            "--out-dir",
            str(out_dir),
            "--canonical-root",
            args.canonical_root,
            "--ckpt-path",
            args.ckpt_path,
            "--token2wav-dir",
            args.token2wav_dir,
            "--attn-implementation",
            args.attn_implementation,
            "--device",
            "cuda:0",
            "--trace-token2wav",
            "--overwrite",
        ]
        with log_path.open("wb") as log:
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        procs.append((case, out_dir, proc))

    failed = False
    for case, out_dir, proc in procs:
        code = proc.wait()
        (base_out_dir / f"case_{case}.exit").write_text(f"{code}\n")
        failed = failed or code != 0

    rows = [
        summarize_case(case, out_dir, expected_generate, expected_duplex)
        for case, out_dir, _ in procs
    ]
    result = {
        "base_out_dir": str(base_out_dir),
        "session_dir": str(session_dir),
        "recorded_duplex_config": recorded,
        "expected_generate_subset": expected_generate,
        "expected_duplex_subset": expected_duplex,
        "cases": rows,
        "unique_texts": sorted({row.get("text", "") for row in rows if row.get("summary_exists")}),
        "unique_audio_sha256": sorted(
            {row.get("continuous_sha256", "") for row in rows if row.get("continuous_sha256")}
        ),
        "all_exits_zero": all(row.get("exit") == 0 for row in rows),
        "all_generate_match_recorded": all(row.get("generate_matches_recorded") for row in rows),
        "all_duplex_match_recorded": all(row.get("duplex_matches_recorded") for row in rows),
    }
    (base_out_dir / "parallel_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
