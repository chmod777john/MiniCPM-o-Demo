#!/usr/bin/env python3
"""Run static human-eval videos through the Demo TP2 backend.

The output layout matches ``add_static_case_model.py``:

    <save>/<ckpt>/<prefix>/<video>/run_1/
      result.json
      chunk_results.json
      subtitles.srt
      output_audio.wav
      joint_audio.wav
      duplex_output.mp4

This file is intentionally an outer evaluation adapter. Model loading and
duplex inference continue to use the existing ``core.deploy`` backend.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from PIL import Image
from scipy.signal import resample_poly


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACE_DIR = REPO_ROOT / "tools" / "o5trace"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(TRACE_DIR))

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
    _run_backend_worker_loop,
    _shutdown_backend,
)


DEFAULT_MODEL_PATH = "/user/weihongliang/MiniCPM-o-4_6"
DEFAULT_CHECKPOINT = (
    "/user/weihongliang/o5_weights/"
    "joint_stage2_40rank_alltry_best_fix_addNum_from_merge_equal_start_gemm_iter_3000.pt"
)
DEFAULT_BACKBONE = (
    "/user/weihongliang/o5_weights/"
    "o5_backbone_hf_joint_stage2_40rank_alltry_best_fix_addNum_from_merge_equal_start_gemm_iter_3000"
)
DEFAULT_REF_AUDIO = (
    "/user/xubokai/audio_eval_3o/"
    "BH-Ref-HT-F224-Ref06_82_U001_话题_3_348s-355s.wav"
)
DEFAULT_SYSTEM_PROMPT = "You, MiniCPM o, the helpful omni full-duplex little powerhouse~"
SPECIAL_TOKENS = (
    "<|tts_pad|>",
    "<|turn_eos|>",
    "<|chunk_eos|>",
    "<|listen|>",
    "<|speak|>",
)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def run_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, capture_output=True, text=True)


def probe_duration(video_path: Path) -> float:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
    )
    return float(result.stdout.strip())


def has_audio_stream(video_path: Path) -> bool:
    result = run_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ]
    )
    return result.stdout.strip() == "audio"


def clean_subtitle_text(text: str) -> str:
    for token in SPECIAL_TOKENS:
        text = text.replace(token, "")
    return text.strip()


def format_srt_time(seconds: float) -> str:
    milliseconds = max(0, int(round(seconds * 1000.0)))
    hours, milliseconds = divmod(milliseconds, 3_600_000)
    minutes, milliseconds = divmod(milliseconds, 60_000)
    secs, milliseconds = divmod(milliseconds, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def write_subtitles(
    path: Path,
    chunk_results: list[dict[str, Any]],
    video_duration: float,
    chunk_seconds: float,
) -> int:
    blocks: list[str] = []
    subtitle_index = 1
    for result in chunk_results:
        if result.get("is_listen", True):
            continue
        text = clean_subtitle_text(str(result.get("text") or ""))
        if not text:
            continue
        start = (int(result["chunk_idx"]) + 1) * chunk_seconds
        if start >= video_duration:
            continue
        end = min(start + chunk_seconds, video_duration)
        blocks.append(
            f"{subtitle_index}\n{format_srt_time(start)} --> {format_srt_time(end)}\n{text}"
        )
        subtitle_index += 1
    path.write_text("\n\n".join(blocks) + ("\n" if blocks else ""), encoding="utf-8")
    return len(blocks)


def build_timed_assistant_track(
    timed_audio: list[tuple[int, np.ndarray]],
    chunk_seconds: float,
    minimum_duration: float,
) -> np.ndarray:
    total_samples = int(math.ceil(minimum_duration * OUTPUT_SAMPLE_RATE))
    for chunk_index, waveform in timed_audio:
        start = int(round((chunk_index + 1) * chunk_seconds * OUTPUT_SAMPLE_RATE))
        total_samples = max(total_samples, start + len(waveform))
    track = np.zeros(total_samples, dtype=np.float32)
    for chunk_index, waveform in timed_audio:
        start = int(round((chunk_index + 1) * chunk_seconds * OUTPUT_SAMPLE_RATE))
        end = start + len(waveform)
        track[start:end] += waveform
    return np.clip(track, -1.0, 1.0)


def write_audio_outputs(
    run_dir: Path,
    input_audio: np.ndarray,
    sequential_audio: list[np.ndarray],
    timed_audio: list[tuple[int, np.ndarray]],
    chunk_seconds: float,
    video_duration: float,
) -> Path | None:
    if sequential_audio:
        sf.write(
            run_dir / "output_audio.wav",
            np.concatenate(sequential_audio).astype(np.float32, copy=False),
            OUTPUT_SAMPLE_RATE,
            subtype="PCM_16",
        )

    assistant_track = build_timed_assistant_track(timed_audio, chunk_seconds, video_duration)
    user_track = resample_poly(input_audio, OUTPUT_SAMPLE_RATE, INPUT_SAMPLE_RATE).astype(np.float32)
    total_samples = max(len(user_track), len(assistant_track))
    stereo = np.zeros((total_samples, 2), dtype=np.float32)
    stereo[: len(user_track), 0] = user_track
    stereo[: len(assistant_track), 1] = assistant_track
    sf.write(run_dir / "joint_audio.wav", stereo, OUTPUT_SAMPLE_RATE, subtype="PCM_16")

    if not timed_audio:
        return None
    assistant_path = run_dir / "assistant_timed.wav"
    sf.write(assistant_path, assistant_track, OUTPUT_SAMPLE_RATE, subtype="PCM_16")
    return assistant_path


def _subtitle_filter(srt_path: Path) -> str:
    escaped = str(srt_path).replace("\\", "\\\\").replace("'", "'\\''").replace(":", "\\:")
    return (
        f"subtitles='{escaped}':"
        "force_style='FontSize=20,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&HA6000000,BackColour=&HA6000000,"
        "BorderStyle=3,Outline=1,Shadow=0,MarginL=36,MarginR=36,"
        "MarginV=24,WrapStyle=0,Alignment=2'"
    )


def render_duplex_video(
    input_video: Path,
    output_video: Path,
    srt_path: Path,
    subtitle_count: int,
    assistant_audio: Path | None,
) -> None:
    input_has_audio = has_audio_stream(input_video)
    has_subtitles = subtitle_count > 0 and srt_path.is_file()
    command = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(input_video)]
    if assistant_audio is not None:
        command.extend(["-i", str(assistant_audio)])

    video_filter = _subtitle_filter(srt_path) if has_subtitles else None
    if assistant_audio is not None and input_has_audio:
        filters = ["[0:a][1:a]amix=inputs=2:duration=longest:dropout_transition=0[aout]"]
        if video_filter:
            filters.insert(0, f"[0:v]{video_filter}[vout]")
            command.extend(["-filter_complex", ";".join(filters), "-map", "[vout]", "-map", "[aout]"])
        else:
            command.extend(["-filter_complex", filters[0], "-map", "0:v:0", "-map", "[aout]"])
    elif assistant_audio is not None:
        if video_filter:
            command.extend(["-filter_complex", f"[0:v]{video_filter}[vout]", "-map", "[vout]", "-map", "1:a:0"])
        else:
            command.extend(["-map", "0:v:0", "-map", "1:a:0"])
    else:
        if video_filter:
            command.extend(["-vf", video_filter])
        command.extend(["-map", "0:v:0"])
        if input_has_audio:
            command.extend(["-map", "0:a:0"])

    command.extend(["-c:v", "libx264", "-preset", "medium", "-crf", "23"])
    if input_has_audio or assistant_audio is not None:
        command.extend(["-c:a", "aac"])
    command.extend(["-movflags", "+faststart", str(output_video)])
    run_command(command)


def sampling_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "generate_audio": True,
        "ls_mode": "explicit",
        "force_listen_count": args.force_listen_count,
        "max_new_speak_tokens_per_chunk": args.max_new_speak_tokens_per_chunk,
        "decode_mode": args.decode_mode,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "listen_prob_scale": args.listen_prob_scale,
        "text_repetition_penalty": args.text_repetition_penalty,
        "text_repetition_window_size": args.text_repetition_window_size,
        "length_penalty": args.length_penalty,
        "tts_temperature": args.tts_temperature,
        "tts_repetition_penalty": args.tts_repetition_penalty,
    }


def engine_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "deployment_mode": args.deployment_mode,
        "attn_implementation": args.attn_implementation,
        "experts_implementation": args.experts_implementation,
        "o5_llm_cache": args.o5_llm_cache,
        "llm_graph": args.llm_graph,
        "tts_graph": args.tts_graph,
        "vocoder_graph": args.vocoder_graph,
        "tts_fast": args.tts_fast,
        "lmhead": args.lmhead,
        "fuse_vision_audio": args.fuse_vision_audio,
        "batch_vision_feed": args.batch_vision_feed,
    }


def read_video_list(path: Path) -> list[Path]:
    videos = [Path(line.strip()).expanduser() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    missing = [str(video) for video in videos if not video.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} input videos are missing; first: {missing[0]}")
    stems = [video.stem for video in videos]
    duplicates = sorted({stem for stem in stems if stems.count(stem) > 1})
    if duplicates:
        raise ValueError(f"duplicate video stems are not compatible with the visualization: {duplicates}")
    return videos


def assigned_videos(args: argparse.Namespace, videos: list[Path]) -> list[tuple[int, Path]]:
    return [
        (index, video)
        for index, video in enumerate(videos, start=1)
        if (index - 1) % args.worker_count == args.worker_index
    ]


def output_root(args: argparse.Namespace) -> Path:
    return Path(args.save_path).expanduser().resolve() / args.ckpt_name / args.prefix_dir


def infer_one(
    args: argparse.Namespace,
    backend: Any,
    item_index: int,
    video_path: Path,
) -> dict[str, Any]:
    run_dir = output_root(args) / video_path.stem / f"run_{args.run_index}"
    required = ("result.json", "subtitles.srt", "duplex_output.mp4", "joint_audio.wav")
    if args.resume and all((run_dir / name).is_file() for name in required):
        return {"item_idx": item_index, "video_path": str(video_path), "status": "skipped", "run_dir": str(run_dir)}
    if run_dir.exists():
        if not (args.overwrite or args.resume):
            raise FileExistsError(f"output exists; pass --overwrite or --resume: {run_dir}")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    sample_seed = args.seed + item_index - 1
    configure_seed(sample_seed)
    seed_runtime = getattr(backend, "seed_runtime", None)
    if callable(seed_runtime):
        seed_runtime(sample_seed)

    frames, input_audio = extract_video(video_path, run_dir / "input_media", args.chunk_ms)
    chunk_samples = int(INPUT_SAMPLE_RATE * args.chunk_ms / 1000)
    audio_units = int(math.ceil(len(input_audio) / float(chunk_samples)))
    unit_count = max(len(frames), audio_units)
    audio_chunks = split_audio(input_audio, unit_count, args.chunk_ms)
    chunk_seconds = args.chunk_ms / 1000.0
    video_duration = probe_duration(video_path)

    sampling = sampling_config(args)
    backend.duplex_prepare(
        system_prompt_text=args.system_prompt,
        ref_audio_path=args.ref_audio,
        prompt_wav_path=args.ref_audio,
        sampling=sampling,
        llm_seed=sample_seed,
    )

    chunk_results: list[dict[str, Any]] = []
    sequential_audio: list[np.ndarray] = []
    timed_audio: list[tuple[int, np.ndarray]] = []
    generated_text: list[str] = []
    total_tokens = 0
    total_tts_tokens = 0
    started = time.monotonic()
    try:
        for chunk_index, audio_chunk in enumerate(audio_chunks):
            frame_list = None
            if frames:
                frame = Image.open(frames[min(chunk_index, len(frames) - 1)]).convert("RGB")
                frame_list = [frame]
            prefill_started = time.monotonic()
            prefill = backend.duplex_prefill(
                audio_waveform=audio_chunk,
                frame_list=frame_list,
                max_slice_nums=args.slice_nums,
            )
            prefill_seconds = time.monotonic() - prefill_started
            generate_started = time.monotonic()
            result = backend.duplex_generate()
            backend.duplex_finalize()
            generate_seconds = time.monotonic() - generate_started

            text = str(_backend_result_value(result, "text", "") or "")
            is_listen = bool(_backend_result_value(result, "is_listen", False))
            end_of_turn = bool(_backend_result_value(result, "end_of_turn", False))
            n_tokens = int(_backend_result_value(result, "n_tokens", 0) or 0)
            n_tts_tokens = int(_backend_result_value(result, "n_tts_tokens", 0) or 0)
            waveform = _backend_audio_waveform(result)
            if text:
                generated_text.append(text)
            if waveform is not None and len(waveform) > 0:
                waveform = np.asarray(waveform, dtype=np.float32)
                sequential_audio.append(waveform)
                timed_audio.append((chunk_index, waveform))
            total_tokens += n_tokens
            total_tts_tokens += n_tts_tokens
            chunk_results.append(
                {
                    "chunk_idx": chunk_index,
                    "time_start_sec": chunk_index * chunk_seconds,
                    "time_end_sec": (chunk_index + 1) * chunk_seconds,
                    "is_listen": is_listen,
                    "text": text,
                    "end_of_turn": end_of_turn,
                    "audio_length": int(len(waveform)) if waveform is not None else 0,
                    "n_tokens": n_tokens,
                    "n_tts_tokens": n_tts_tokens,
                    "prefill_success": bool(prefill.get("success")) if isinstance(prefill, dict) else bool(prefill),
                    "prefill_seconds": prefill_seconds,
                    "generate_seconds": generate_seconds,
                }
            )
    except Exception:
        try:
            backend.duplex_finalize()
        except Exception:
            pass
        raise

    elapsed = time.monotonic() - started
    chunk_results_path = run_dir / "chunk_results.json"
    chunk_results_path.write_text(
        json.dumps(jsonable(chunk_results), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    subtitle_count = write_subtitles(
        run_dir / "subtitles.srt", chunk_results, video_duration, chunk_seconds
    )
    assistant_audio = write_audio_outputs(
        run_dir,
        input_audio,
        sequential_audio,
        timed_audio,
        chunk_seconds,
        video_duration,
    )
    render_duplex_video(
        video_path,
        run_dir / "duplex_output.mp4",
        run_dir / "subtitles.srt",
        subtitle_count,
        assistant_audio,
    )
    if assistant_audio is not None:
        assistant_audio.unlink(missing_ok=True)

    payload = {
        "item_idx": item_index,
        "video_path": str(video_path),
        "generated_text": "".join(generated_text),
        "output_dir": str(run_dir),
        "output_video_path": str(run_dir / "duplex_output.mp4"),
        "joint_audio_path": str(run_dir / "joint_audio.wav"),
        "subtitles_path": str(run_dir / "subtitles.srt"),
        "generate_elapsed_time": elapsed,
        "total_n_tokens": total_tokens,
        "total_n_tts_tokens": total_tts_tokens,
        "run_idx": args.run_index,
        "worker_index": args.worker_index,
        "seed": sample_seed,
        "slice_nums": args.slice_nums,
        "decode_mode": args.decode_mode,
        "generation_config": sampling,
        "engine_config": engine_config(args),
        "model_path": args.model_path,
        "checkpoint_path": args.ckpt_path,
        "backbone_dir": args.backbone_dir,
        "ref_audio_path": args.ref_audio,
        "system_prompt": args.system_prompt,
    }
    (run_dir / "result.json").write_text(
        json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"item_idx": item_index, "video_path": str(video_path), "status": "success", "run_dir": str(run_dir)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--ckpt-path", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--backbone-dir", default=DEFAULT_BACKBONE)
    parser.add_argument("--ref-audio", default=DEFAULT_REF_AUDIO)
    parser.add_argument("--video-list", required=True)
    parser.add_argument("--save-path", required=True)
    parser.add_argument("--prefix-dir", required=True)
    parser.add_argument("--ckpt-name")
    parser.add_argument("--run-index", type=int, default=1)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--chunk-ms", type=int, default=1000)
    parser.add_argument("--slice-nums", type=int, default=1)
    parser.add_argument("--deployment-mode", choices=("single_eager", "tp2"), default="tp2")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--decode-mode", choices=("sampling", "greedy"), default="sampling")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--tts-temperature", type=float, default=0.8)
    parser.add_argument("--tts-repetition-penalty", type=float, default=1.10)
    parser.add_argument("--listen-prob-scale", type=float, default=1.0)
    parser.add_argument("--text-repetition-penalty", type=float, default=1.05)
    parser.add_argument("--text-repetition-window-size", type=int, default=512)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--force-listen-count", type=int, default=0)
    parser.add_argument("--max-new-speak-tokens-per-chunk", type=int, default=20)
    parser.add_argument("--experts-implementation", choices=("eager", "batched_mm"), default="batched_mm")
    parser.add_argument("--o5-llm-cache", type=int, default=65536)
    parser.add_argument("--llm-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tts-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vocoder-graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tts-fast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lmhead", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fuse-vision-audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-vision-feed", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.worker_count < 1 or not 0 <= args.worker_index < args.worker_count:
        parser.error("worker index must satisfy 0 <= worker-index < worker-count")
    if args.chunk_ms <= 0:
        parser.error("chunk-ms must be positive")
    if args.slice_nums < 1:
        parser.error("slice-nums must be positive")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if args.ckpt_name is None:
        args.ckpt_name = Path(args.ckpt_path).stem
    for path in (args.model_path, args.ckpt_path, args.backbone_dir, args.ref_audio, args.video_list):
        if not Path(path).expanduser().exists():
            parser.error(f"path does not exist: {path}")
    for command in ("ffmpeg", "ffprobe"):
        if shutil.which(command) is None:
            parser.error(f"required command is not available: {command}")
    return args


def main() -> int:
    args = parse_args()
    videos = read_video_list(Path(args.video_list))
    assigned = assigned_videos(args, videos)
    plan = {
        "event": "run_plan",
        "worker_index": args.worker_index,
        "worker_count": args.worker_count,
        "input_count": len(videos),
        "assigned_count": len(assigned),
        "assigned_indices": [index for index, _ in assigned],
        "output_root": str(output_root(args)),
        "checkpoint": args.ckpt_path,
        "backbone": args.backbone_dir,
        "engine": engine_config(args),
        "sampling": sampling_config(args),
    }
    print(json.dumps(plan, ensure_ascii=False), flush=True)
    if args.dry_run:
        return 0

    output_root(args).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    _apply_safe_engine_env(args)

    backend = _build_backend(args)
    if getattr(backend, "spmd_is_worker", False):
        _run_backend_worker_loop(backend)
    records: list[dict[str, Any]] = []
    try:
        for item_index, video_path in assigned:
            print(
                json.dumps(
                    {"event": "case_start", "item_idx": item_index, "video_path": str(video_path)},
                    ensure_ascii=False,
                ),
                flush=True,
            )
            record = infer_one(args, backend, item_index, video_path)
            records.append(record)
            print(json.dumps({"event": "case_done", **record}, ensure_ascii=False), flush=True)
    finally:
        _shutdown_backend(backend)

    summary_path = output_root(args) / f"worker_{args.worker_index:02d}_summary.json"
    summary_path.write_text(
        json.dumps({**plan, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"event": "run_done", "summary": str(summary_path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
