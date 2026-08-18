#!/usr/bin/env python3
"""Build a short exact-input session bundle from a local video."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.o5trace.thin_duplex_video_probe import extract_video, split_audio  # noqa: E402


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--ref-audio-path", required=True)
    parser.add_argument("--max-units", type=int, default=8)
    parser.add_argument("--chunk-ms", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--session-id", default="short-video-trace")
    parser.add_argument("--system-prompt", default="Streaming Omni Conversation.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists; pass --overwrite: {out_dir}")
        shutil.rmtree(out_dir)
    blob_dir = out_dir / "blob"
    media_dir = out_dir / "_media"
    blob_dir.mkdir(parents=True)
    frames, audio = extract_video(Path(args.video).resolve(), media_dir, args.chunk_ms)
    chunk_len = int(16000 * args.chunk_ms / 1000)
    unit_count = max(len(frames), int(math.ceil(len(audio) / chunk_len)))
    if args.max_units > 0:
        unit_count = min(unit_count, args.max_units)
    chunks = split_audio(audio, unit_count, args.chunk_ms)

    config = {
        "generate_audio": True,
        "ls_mode": "explicit",
        "force_listen_count": 0,
        "max_new_speak_tokens_per_chunk": 20,
        "decode_mode": "sampling",
        "temperature": 0.7,
        "top_k": 100,
        "top_p": 0.8,
        "tts_temperature": 0.8,
        "tts_repetition_penalty": 1.05,
        "text_repetition_penalty": 1.05,
        "text_repetition_window_size": 512,
        "listen_prob_scale": 1.0,
        "n_timesteps": 10,
        "chunk_ms": args.chunk_ms,
    }
    rows = [
        {
            "seq": 0,
            "ts": time.time(),
            "dir": "up",
            "frame": {
                "type": "session.init",
                "payload": {
                    "mode": "full_duplex",
                    "system_prompt": args.system_prompt,
                    "ref_audio_path": str(Path(args.ref_audio_path).resolve()),
                    "seed": args.seed,
                    "config": config,
                },
            },
        },
        {
            "seq": 1,
            "ts": time.time(),
            "dir": "down",
            "frame": {
                "type": "session.created",
                "session_id": args.session_id,
                "replay_manifest": {
                    "resolved": {"seed": args.seed, "llm_seed": args.seed, "duplex_config": config},
                },
            },
        },
    ]
    for index, chunk in enumerate(chunks):
        audio_raw = np.ascontiguousarray(chunk, dtype="<f4").tobytes()
        audio_name = f"unit_{index:06d}.f32"
        (blob_dir / audio_name).write_bytes(audio_raw)
        frame = frames[min(index, len(frames) - 1)] if frames else None
        frame_meta = []
        frame_pointer = []
        if frame is not None:
            frame_raw = frame.read_bytes()
            frame_name = f"unit_{index:06d}.jpg"
            (blob_dir / frame_name).write_bytes(frame_raw)
            frame_pointer.append(f"@blob/{frame_name}")
            frame_meta.append({
                "blob_jpg": f"@blob/{frame_name}",
                "sha256": _sha(frame_raw),
                "nbytes": len(frame_raw),
            })
        input_id = f"unit_{index:06d}"
        rows.append({
            "seq": len(rows),
            "ts": time.time(),
            "dir": "up",
            "frame": {
                "type": "input.append",
                "input": {
                    "input_id": input_id,
                    "audio": f"@blob/{audio_name}",
                    "video_frames": frame_pointer,
                    "force_listen": False,
                    "max_slice_nums": 1,
                },
            },
            "payload_trace": {
                "audio": {
                    "blob_f32": f"@blob/{audio_name}",
                    "sha256": _sha(audio_raw),
                    "nbytes": len(audio_raw),
                    "samples": int(chunk.size),
                    "sample_rate": 16000,
                },
                "video_frames": frame_meta,
                "force_listen": False,
                "max_slice_nums": 1,
            },
        })
    with (out_dir / "stream.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (out_dir / "meta.json").write_text(json.dumps({
        "session_id": args.session_id,
        "source_video": str(Path(args.video).resolve()),
        "units": unit_count,
        "chunk_ms": args.chunk_ms,
        "seed": args.seed,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.rmtree(media_dir)
    print(json.dumps({"session_dir": str(out_dir), "units": unit_count}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
