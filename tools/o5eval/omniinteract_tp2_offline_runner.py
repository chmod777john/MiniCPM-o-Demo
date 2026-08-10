#!/usr/bin/env python3
"""Run O5 duplex offline inference and emit OmniInteract TTS-eval outputs.

This runner deliberately avoids the realtime API service. It imports the local
model code, builds the configured deployment mode, runs video/audio duplex
units, and writes the same leaf-directory contract consumed by
OmniInteract_tts_eval/score.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
TRACE_DIR = REPO_ROOT / "tools" / "o5trace"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(TRACE_DIR))

DEFAULT_PROMPT_WAV_ZH = "/user/houyueran/GIT/SoulX-Podcast_MakeTrainData/ref_audio/ref_minicpm_signature.wav"
DEFAULT_PROMPT_WAV_EN = "/user/houyueran/GIT/SoulX-Podcast_MakeTrainData/ref_audio/F409-379_clip_310-324.wav"

from thin_duplex_video_probe import (  # noqa: E402
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    configure_seed,
    extract_video,
    split_audio,
)
from tp2_duplex_video_probe import (  # noqa: E402
    _apply_safe_engine_env,
    _backend_audio_waveform,
    _backend_result_value,
    _build_backend,
    _duplex_sampling_config,
    _run_backend_worker_loop,
    _shutdown_backend,
)


_ZH_DIGIT = "零一二三四五六七八九"


def _int_to_zh(n: int) -> str:
    if n < 10:
        return _ZH_DIGIT[n]
    if n < 20:
        return "十" + (_ZH_DIGIT[n % 10] if n % 10 else "")
    if n < 100:
        t, u = divmod(n, 10)
        return _ZH_DIGIT[t] + "十" + (_ZH_DIGIT[u] if u else "")
    if n < 10000:
        high, rem = divmod(n, 100)
        s = _int_to_zh(high) + "百"
        if not rem:
            return s
        if rem < 10:
            return s + "零" + _ZH_DIGIT[rem]
        return s + _int_to_zh(rem)
    return "".join(_ZH_DIGIT[int(c)] for c in str(n))


def normalize_spoken(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        run = match.group()
        if len(run) <= 4:
            return _int_to_zh(int(run))
        return "".join(_int_to_zh(int(run[i : i + 2])) for i in range(0, len(run), 2))

    return re.sub(r"\d+", repl, text)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def parse_job_line(line: str) -> dict[str, Any]:
    cols = line.rstrip("\n").split("\t")
    if len(cols) not in {3, 4, 5}:
        raise ValueError(f"expected 3-5 TSV columns, got {len(cols)}: {line!r}")
    tag, video, seed = cols[:3]
    lang = cols[3] if len(cols) >= 4 and cols[3] else ("en" if tag.startswith("1qna") else "zh")
    stop_sec = float(cols[4]) if len(cols) >= 5 and cols[4] else None
    return {"tag": tag, "video": video, "seed": int(seed), "lang": lang, "stop_sec": stop_sec}


def read_job(args: argparse.Namespace) -> dict[str, Any]:
    jobs = read_jobs(args)
    if len(jobs) != 1:
        raise ValueError(f"expected exactly one job, got {len(jobs)}")
    return jobs[0][1]


def _parse_job_indices(spec: str) -> list[int]:
    indices: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"invalid descending job range: {part}")
            indices.extend(range(start, end + 1))
        else:
            indices.append(int(part))
    if not indices:
        raise ValueError("--job-indices did not contain any index")
    return indices


def read_jobs(args: argparse.Namespace) -> list[tuple[int | None, dict[str, Any]]]:
    if args.job_line:
        return [(None, parse_job_line(args.job_line))]
    if args.jobs_file:
        lines = [line for line in Path(args.jobs_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        if args.job_indices:
            indices = _parse_job_indices(args.job_indices)
        else:
            if args.job_count < 0:
                raise ValueError("--job-count must be >= 0")
            if args.job_stride <= 0:
                raise ValueError("--job-stride must be > 0")
            indices = []
            index = args.job_index
            while index < len(lines) and (args.job_count == 0 or len(indices) < args.job_count):
                indices.append(index)
                index += args.job_stride
        jobs: list[tuple[int | None, dict[str, Any]]] = []
        for index in indices:
            if index < 0 or index >= len(lines):
                raise IndexError(f"job index {index} outside jobs file length {len(lines)}")
            jobs.append((index, parse_job_line(lines[index])))
        return jobs
    if not (args.video and args.video_tag):
        raise ValueError("provide --job-line, --jobs-file, or both --video and --video-tag")
    return [
        (
            None,
            {
                "tag": args.video_tag,
                "video": args.video,
                "seed": args.seed,
                "lang": args.lang or ("en" if args.video_tag.startswith("1qna") else "zh"),
                "stop_sec": args.stop_sec,
            },
        )
    ]


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.asarray(audio, dtype=np.float32), sr)


def leaf_dir(out_root: Path, ckpt_name: str, tag: str, seed: int) -> Path:
    return out_root / ckpt_name / tag / f"seed_{seed:04d}"


def prompt_wav_for_job(args: argparse.Namespace, lang: str) -> str:
    path = args.prompt_wav or (args.prompt_wav_en if lang == "en" else args.prompt_wav_zh)
    return str(Path(path).expanduser().resolve())


def run_one_job(args: argparse.Namespace, backend: object, job: dict[str, Any], job_index: int | None) -> None:
    out_dir = leaf_dir(Path(args.out_root), args.ckpt_name, job["tag"], job["seed"])
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite: {out_dir}")
    if out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_chunk_dir = out_dir / "output_audio_chunks"
    audio_chunk_dir.mkdir(parents=True, exist_ok=True)

    configure_seed(job["seed"])
    prompt_wav = prompt_wav_for_job(args, job["lang"])
    frames, input_audio = extract_video(Path(job["video"]), out_dir / "input_media", args.chunk_ms)
    chunk_len = int(INPUT_SAMPLE_RATE * args.chunk_ms / 1000)
    audio_unit_count = int(math.ceil(len(input_audio) / float(chunk_len)))
    num_units = max(len(frames), audio_unit_count)
    if args.max_units > 0:
        num_units = min(num_units, args.max_units)
    audio_chunks = split_audio(input_audio, num_units, args.chunk_ms)

    backend.duplex_prepare(
        system_prompt_text=args.system_prompt,
        ref_audio_path=prompt_wav,
        prompt_wav_path=prompt_wav,
        sampling=_duplex_sampling_config(args),
        llm_seed=job["seed"],
    )

    all_text: list[str] = []
    all_audio: list[np.ndarray] = []
    turn = {"text": [], "audio": [], "tts": 0, "llm": 0, "idx": 0}
    chunk_sec = args.chunk_ms / 1000.0
    stop_unit = math.ceil(job["stop_sec"] / chunk_sec) if job["stop_sec"] is not None else None
    soft_stop_units = math.ceil(args.soft_stop_max_seconds / chunk_sec) if stop_unit is not None else None
    units_fed = 0
    stop_reason = "full_video"

    def flush_turn(log_f):
        if not turn["text"] and not turn["audio"]:
            return
        raw_text = "".join(turn["text"])
        text = normalize_spoken(raw_text) if job["lang"] == "zh" else raw_text
        audio_path = None
        audio_samples = 0
        if turn["audio"]:
            merged = np.concatenate(turn["audio"])
            audio_samples = int(len(merged))
            wav_path = audio_chunk_dir / f"turn_{turn['idx']:03d}.wav"
            write_wav(wav_path, merged, OUTPUT_SAMPLE_RATE)
            audio_path = wav_path.relative_to(out_dir).as_posix()
        record = {
            "unit_id": 1000 + int(turn["idx"]),
            "output": {
                "is_listen": False,
                "text": text,
                "text_raw": raw_text,
                "lang": job["lang"],
                "audio_path": audio_path,
                "audio_samples": audio_samples,
                "audio_sample_rate": OUTPUT_SAMPLE_RATE if audio_path else None,
                "end_of_turn": True,
                "n_tokens": int(turn["llm"]),
                "n_tts_tokens": int(turn["tts"]),
                "tts_deferred": False,
            },
        }
        log_f.write(json.dumps(jsonable(record), ensure_ascii=False) + "\n")
        log_f.flush()
        turn["idx"] = int(turn["idx"]) + 1
        turn["text"], turn["audio"], turn["tts"], turn["llm"] = [], [], 0, 0

    unit_log = out_dir / "unit_log.jsonl"
    detail_log = out_dir / "unit_log_detailed.jsonl"
    with unit_log.open("w", encoding="utf-8") as log_f, detail_log.open("w", encoding="utf-8") as det_f:
        for idx in range(num_units):
            frame = Image.open(frames[min(idx, len(frames) - 1)]).convert("RGB") if frames else None
            frame_list = [frame] if frame is not None else None
            prefill = backend.duplex_prefill(
                audio_waveform=audio_chunks[idx],
                frame_list=frame_list,
                max_slice_nums=1,
            )
            result = backend.duplex_generate()
            backend.duplex_finalize()

            text = _backend_result_value(result, "text", "") or ""
            if text:
                all_text.append(text)
            wav = _backend_audio_waveform(result)
            unit_audio_path = None
            if wav is not None and len(wav) > 0:
                wav = np.asarray(wav, dtype=np.float32)
                all_audio.append(wav)
                unit_path = audio_chunk_dir / f"unit_{idx:06d}.wav"
                write_wav(unit_path, wav, OUTPUT_SAMPLE_RATE)
                unit_audio_path = unit_path.relative_to(out_dir).as_posix()

            is_listen = bool(_backend_result_value(result, "is_listen", False))
            end_of_turn = bool(_backend_result_value(result, "end_of_turn", False))
            if is_listen:
                flush_turn(log_f)
            else:
                if text:
                    turn["text"].append(text)
                if wav is not None and len(wav) > 0:
                    turn["audio"].append(wav)
                turn["tts"] = int(turn["tts"]) + int(_backend_result_value(result, "n_tts_tokens", 0) or 0)
                turn["llm"] = int(turn["llm"]) + int(_backend_result_value(result, "n_tokens", 0) or 0)
                if end_of_turn:
                    flush_turn(log_f)

            detail = {
                "unit_id": idx,
                "input": {
                    "video_path": str(job["video"]),
                    "time_start_sec": idx * chunk_sec,
                    "frame_path": str(frames[min(idx, len(frames) - 1)]) if frames else None,
                },
                "prefill": prefill,
                "output": {
                    "is_listen": is_listen,
                    "text": text,
                    "audio_path": unit_audio_path,
                    "audio_samples": int(len(wav)) if wav is not None else 0,
                    "audio_sample_rate": OUTPUT_SAMPLE_RATE if wav is not None else None,
                    "end_of_turn": end_of_turn,
                    "n_tokens": _backend_result_value(result, "n_tokens", 0),
                    "n_tts_tokens": _backend_result_value(result, "n_tts_tokens", 0),
                    "tts_deferred": bool(_backend_result_value(result, "tts_deferred", False)),
                },
            }
            det_f.write(json.dumps(jsonable(detail), ensure_ascii=False) + "\n")
            det_f.flush()
            units_fed = idx + 1

            if stop_unit is not None and idx >= stop_unit and (is_listen or end_of_turn):
                stop_reason = "turn_boundary"
                break
            if (
                stop_unit is not None
                and soft_stop_units is not None
                and idx >= stop_unit + soft_stop_units
            ):
                stop_reason = "soft_stop_exhausted"
                break

    (out_dir / "transcript.txt").write_text("".join(all_text), encoding="utf-8")
    if all_audio:
        write_wav(out_dir / "output_audio.wav", np.concatenate(all_audio), OUTPUT_SAMPLE_RATE)
    meta = {
        "lang": job["lang"],
        "video_path": str(job["video"]),
        "seed": job["seed"],
        "num_units": num_units,
        "units_fed": units_fed,
        "input_sec_fed": units_fed * chunk_sec,
        "requested_stop_sec": job["stop_sec"],
        "soft_stop_max_sec": args.soft_stop_max_seconds if job["stop_sec"] is not None else None,
        "stop_reason": stop_reason,
        "turns": int(turn["idx"]),
        "prompt_wav": prompt_wav,
        "sampling": _duplex_sampling_config(args),
        "chunk_ms": args.chunk_ms,
        "n_timesteps_requested": args.n_timesteps,
        "model_path": args.model_path,
        "ckpt_path": args.ckpt_path,
        "backbone_dir": args.backbone_dir,
        "deployment_mode": args.deployment_mode,
    }
    (out_dir / "meta.json").write_text(json.dumps(jsonable(meta), ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "job_index": job_index,
                "tag": job["tag"],
                "out_dir": str(out_dir),
                "turns": int(turn["idx"]),
                "units_fed": units_fed,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def run_driver(args: argparse.Namespace, backend: object) -> int:
    jobs = read_jobs(args)
    print(
        json.dumps(
            {
                "out_root": args.out_root,
                "ckpt_name": args.ckpt_name,
                "num_jobs": len(jobs),
                "job_indices": [index for index, _ in jobs],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    for job_index, job in jobs:
        print(
            json.dumps(
                {"event": "job_start", "job_index": job_index, **job},
                ensure_ascii=False,
            ),
            flush=True,
        )
        run_one_job(args, backend, job, job_index)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--ckpt-path", required=True)
    parser.add_argument("--backbone-dir", required=True)
    parser.add_argument("--prompt-wav")
    parser.add_argument("--prompt-wav-zh", default=DEFAULT_PROMPT_WAV_ZH)
    parser.add_argument("--prompt-wav-en", default=DEFAULT_PROMPT_WAV_EN)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--ckpt-name", default="o5_tp2_fullaccel")
    parser.add_argument("--jobs-file")
    parser.add_argument("--job-index", type=int, default=0)
    parser.add_argument("--job-count", type=int, default=1, help="number of jobs to run from --job-index; 0 means until EOF")
    parser.add_argument("--job-stride", type=int, default=1)
    parser.add_argument("--job-indices", help="comma-separated indices/ranges, for example 0,2,10-15")
    parser.add_argument("--job-line")
    parser.add_argument("--video")
    parser.add_argument("--video-tag")
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--lang", choices=("zh", "en"))
    parser.add_argument("--stop-sec", type=float)
    parser.add_argument("--chunk-ms", type=int, default=1000)
    parser.add_argument("--max-units", type=int, default=0)
    parser.add_argument("--soft-stop-max-seconds", type=float, default=30.0)
    parser.add_argument("--system-prompt", default="Streaming Omni Conversation.")
    parser.add_argument("--deployment-mode", choices=("single_eager", "tp2"), default="tp2")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--decode-mode", default="sampling")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--tts-temperature", type=float, default=0.8)
    parser.add_argument("--tts-repetition-penalty", type=float, default=1.05)
    parser.add_argument("--listen-prob-scale", type=float, default=1.0)
    parser.add_argument("--text-repetition-penalty", type=float, default=1.05)
    parser.add_argument("--text-repetition-window-size", type=int, default=512)
    parser.add_argument("--length-penalty", type=float, default=1.1)
    parser.add_argument("--force-listen-count", type=int, default=0)
    parser.add_argument("--max-new-speak-tokens-per-chunk", type=int, default=20)
    parser.add_argument("--n-timesteps", type=int, default=10)
    parser.add_argument("--experts-implementation", choices=("eager", "batched_mm"), default="batched_mm")
    parser.add_argument("--o5-llm-cache", type=int, default=32768)
    parser.add_argument("--llm-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tts-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vocoder-graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tts-fast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lmhead", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fuse-vision-audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-vision-feed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for lang, prompt_wav in (("zh", args.prompt_wav_zh), ("en", args.prompt_wav_en)):
        if args.prompt_wav:
            prompt_wav = args.prompt_wav
        if not Path(prompt_wav).expanduser().is_file():
            parser.error(f"{lang} prompt audio not found: {prompt_wav}")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _apply_safe_engine_env(args)

    backend = _build_backend(args)
    if getattr(backend, "spmd_is_worker", False):
        _run_backend_worker_loop(backend)
    try:
        return run_driver(args, backend)
    finally:
        _shutdown_backend(backend)


if __name__ == "__main__":
    raise SystemExit(main())
