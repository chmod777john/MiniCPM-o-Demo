#!/usr/bin/env python3
"""Run OmniPro Online through the Demo TP2 backend.

The benchmark adapter and scorer stay in HumanEvalKit.  This runner owns only
Demo model construction, the TP2 lockstep loop, and per-sample serialization.
It intentionally keeps the inference and remote-judge phases separate: rank 1
must remain in the SPMD worker loop while rank 0 writes results.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import soundfile as sf
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
TRACE_DIR = REPO_ROOT / "tools" / "o5trace"
HEK_ROOT = Path(os.environ.get("HUMANEVALKIT_ROOT", "/user/weihongliang/humanevalkit-codeup"))
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(TRACE_DIR))
sys.path.insert(0, str(HEK_ROOT / "src"))

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
    _run_backend_worker_loop,
    _shutdown_backend,
)


DEFAULT_MODEL_PATH = "/user/weihongliang/MiniCPM-o-4_6"
DEFAULT_CHECKPOINT = (
    "/user/weihongliang/o5_weights/"
    "joint_stage2_40rank_lessa2o2_audio_coni_from_merge_equal_start_gemm_iter_4000.pt"
)
DEFAULT_BACKBONE = (
    "/user/weihongliang/o5_weights/"
    "o5_backbone_hf_joint_stage2_40rank_lessa2o2_audio_coni_from_merge_equal_start_gemm_iter_4000"
)
DEFAULT_DATA_ROOT = "/user/sunyinuo/data/OmniPro"
DEFAULT_REF_AUDIO = (
    "/backup/user/xubokai/humanevalkit_dev/migration_v2/humanevalkit/"
    "runs_chaoqun/HT_ref_audio.wav"
)
DEFAULT_JUDGE_URL = "https://llm-center.modelbest.co/llm/v1/chat/completions"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _sampling_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "generate_audio": bool(args.generate_audio),
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
        "n_timesteps": args.n_timesteps,
    }


def _engine_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "deployment_mode": args.deployment_mode,
        "attn_implementation": args.attn_implementation,
        "experts_implementation": args.experts_implementation,
        "llm_graph": args.llm_graph,
        "tts_graph": args.tts_graph,
        "vocoder_graph": args.vocoder_graph,
        "tts_fast": args.tts_fast,
        "lmhead": args.lmhead,
        "fuse_vision_audio": args.fuse_vision_audio,
        "batch_vision_feed": args.batch_vision_feed,
        "llm_cache_len": args.o5_llm_cache,
        "generate_audio": bool(args.generate_audio),
    }


def _video_path(sample: Any) -> str:
    for track in sample.tracks:
        if str(track.modality) != "video" or not track.segments:
            continue
        path = getattr(track.segments[0].payload, "video_path", None)
        if path:
            return str(path)
    raise ValueError(f"sample {sample.sample_id} has no video path")


def _build_backend(args: argparse.Namespace) -> Any:
    """Build the Demo backend locally so its TP2 path is the code under test."""

    from core.processors.backend_factory import create_backend

    os.environ["O5_DEPLOY_MODE"] = args.deployment_mode
    os.environ["O5_BACKBONE_DIR"] = args.backbone_dir
    backend = create_backend(
        {
            "deployment_mode": args.deployment_mode,
            "model_path": args.model_path,
            "pt_path": args.checkpoint_path,
            "backbone_dir": args.backbone_dir,
            "gpu_id": int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0"))),
            "chat_vocoder": "token2wav",
            "attn_implementation": args.attn_implementation,
            "llm_cache_len": args.o5_llm_cache,
            "duplex_config": _sampling_config(args),
        }
    )
    backend.load_model()

    model = getattr(getattr(backend, "processor", None), "model", None)
    duplex = getattr(model, "duplex", None)
    if duplex is not None:
        for name, value in _sampling_config(args).items():
            if hasattr(duplex, name):
                setattr(duplex, name, value)
    return backend


def _make_output(sample_id: str, units: list[Any], audio_chunks: list[np.ndarray], audio_path: Path) -> Any:
    from humanevalkit.schemas.outputs import DuplexOutput

    output_audio_path: str | None = None
    sample_rate: int | None = None
    if audio_chunks:
        merged = np.concatenate(audio_chunks).astype(np.float32, copy=False)
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(audio_path, merged, OUTPUT_SAMPLE_RATE)
        output_audio_path = str(audio_path)
        sample_rate = OUTPUT_SAMPLE_RATE
    return DuplexOutput(
        sample_id=sample_id,
        units=units,
        audio_path=output_audio_path,
        sample_rate=sample_rate,
    )


def _write_result(
    *,
    run_item: Any,
    output: Any,
    records: list[Any],
    skipped: list[dict[str, Any]],
    args: argparse.Namespace,
    output_root: Path,
) -> dict[str, Any]:
    result_path = run_item.sample_dir / "result.json"
    payload = {
        "sample": run_item.sample.model_dump(mode="json"),
        "request": run_item.request.model_dump(mode="json"),
        "output": output.model_dump(mode="json"),
        "metrics": [record.model_dump(mode="json") for record in records],
        "skipped_metrics": skipped,
        "runtime_metadata": {
            "demo_worktree": str(REPO_ROOT),
            "demo_git_commit": os.environ.get("DEMO_GIT_COMMIT", "unknown"),
            "data_root": args.data_root,
            "engine": _engine_config(args),
            "sampling": _sampling_config(args),
            "sample_seed": args.seed + int(run_item.raw_sample.get("_row_index", 0)),
            "inference_contract": {
                "mode": "omnipro_online",
                "chunk_seconds": args.chunk_ms / 1000.0,
                "timestamp_source": "input_chunk_index",
                "remote_judge_inference_phase": False,
            },
        },
    }
    result_path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "sample_id": run_item.sample.sample_id,
        "result_path": str(result_path.relative_to(output_root)),
        "unit_count": len(output.units),
        "speak_units": sum(1 for unit in output.units if str(unit.unit_type) == "speak"),
        "text": output.text,
        "audio_path": output.audio_path,
        "error": output.error.model_dump(mode="json") if output.error else None,
        "metric_values": {
            record.metric_name: _jsonable(record.value) for record in records
        },
    }


def run_one(
    args: argparse.Namespace,
    backend: Any,
    raw_row: dict[str, Any],
    index: int,
    deps: SimpleNamespace,
) -> tuple[Any, list[Any], list[dict[str, Any]], dict[str, Any]]:
    output_root = Path(args.output_dir).resolve()
    run_item = deps.prepare_run_item(
        benchmark_name="omnipro_online",
        raw_sample=raw_row,
        output_dir=output_root,
        model_path=args.model_path,
        checkpoint_path=args.checkpoint_path,
        device="cuda:0",
        index=index,
        chunk_seconds=args.chunk_ms / 1000.0,
        duplex_generate_audio=args.generate_audio,
    )
    request = run_item.request
    # The OmniPro builder already installs the task-specific prompt and the
    # uncapped 1fps flag.  Keep the resolved request in result.json while
    # applying CLI overrides to the Demo backend call.
    request.decode_mode = args.decode_mode
    request.generate_audio = bool(args.generate_audio)
    request.max_new_speak_tokens_per_chunk = args.max_new_speak_tokens_per_chunk
    request.chunk_plan.chunk_seconds = args.chunk_ms / 1000.0
    request.runtime_kwargs["duplex_uncapped_fps"] = True
    request.runtime_kwargs["propagate_generate_audio"] = True
    if args.generate_audio:
        request.duplex_reference_audio_path = args.ref_audio
        request.output_audio_path = str(run_item.sample_dir / "assistant.wav")
    else:
        request.duplex_reference_audio_path = None
        request.output_audio_path = None

    result_path = run_item.sample_dir / "result.json"
    if args.resume and result_path.exists():
        preview = {"sample_id": run_item.sample.sample_id, "resumed": True}
        print(json.dumps({"event": "resume_skip", **preview}, ensure_ascii=False), flush=True)
        return run_item.sample, [], [], preview

    row_index = int(raw_row.get("_row_index", index))
    sample_seed = args.seed + row_index
    configure_seed(sample_seed)
    seed_runtime = getattr(backend, "seed_runtime", None)
    if callable(seed_runtime):
        seed_runtime(sample_seed)

    video_path = Path(_video_path(run_item.sample))
    frames, input_audio = extract_video(
        video_path,
        run_item.sample_dir / "input_media",
        args.chunk_ms,
    )
    chunk_len = int(INPUT_SAMPLE_RATE * args.chunk_ms / 1000)
    audio_units = int(np.ceil(len(input_audio) / float(chunk_len)))
    unit_count = max(len(frames), audio_units)
    if args.max_input_seconds > 0:
        unit_count = min(unit_count, int(np.ceil(args.max_input_seconds / (args.chunk_ms / 1000.0))))
    if unit_count <= 0:
        raise RuntimeError(f"video produced no input units: {video_path}")
    chunks = deps.split_audio(input_audio, unit_count, args.chunk_ms)

    sampling = _sampling_config(args)
    backend.duplex_prepare(
        system_prompt_text=request.duplex_system_prompt,
        ref_audio_path=args.ref_audio if args.generate_audio else None,
        prompt_wav_path=args.ref_audio if args.generate_audio else None,
        sampling=sampling,
        llm_seed=sample_seed,
    )

    from humanevalkit.schemas.common import AudioSpan, DuplexUnitType, ErrorInfo
    from humanevalkit.schemas.outputs import DuplexOutput

    units: list[Any] = []
    output_audio: list[np.ndarray] = []
    audio_offset = 0.0
    try:
        for unit_index, chunk in enumerate(chunks):
            frame_list = []
            if frames:
                frame_list = [Image.open(frames[min(unit_index, len(frames) - 1)]).convert("RGB")]
            prefill = backend.duplex_prefill(
                audio_waveform=chunk,
                frame_list=frame_list or None,
                max_slice_nums=1,
            )
            result = backend.duplex_generate()
            backend.duplex_finalize()

            waveform = _backend_audio_waveform(result)
            audio_span = None
            if waveform is not None and len(waveform) > 0:
                waveform = np.asarray(waveform, dtype=np.float32)
                output_audio.append(waveform)
                duration = len(waveform) / float(OUTPUT_SAMPLE_RATE)
                audio_span = AudioSpan(
                    start_time_sec=audio_offset,
                    end_time_sec=audio_offset + duration,
                )
                audio_offset += duration

            is_listen = bool(_backend_result_value(result, "is_listen", True))
            units.append(
                deps.DuplexOutputUnit(
                    unit_id=f"unit_{unit_index}",
                    unit_type=DuplexUnitType.LISTEN if is_listen else DuplexUnitType.SPEAK,
                    start_time_sec=unit_index * args.chunk_ms / 1000.0,
                    end_time_sec=(unit_index + 1) * args.chunk_ms / 1000.0,
                    text=_backend_result_value(result, "text", "") or None,
                    audio_span=audio_span,
                    metadata={
                        "end_of_turn": bool(_backend_result_value(result, "end_of_turn", False)),
                        "prefill_success": bool(prefill.get("success")) if isinstance(prefill, dict) else None,
                        "n_tokens": _backend_result_value(result, "n_tokens", 0),
                        "n_tts_tokens": _backend_result_value(result, "n_tts_tokens", 0),
                        "input_time_sec": unit_index * args.chunk_ms / 1000.0,
                    },
                )
            )
    except Exception as exc:
        try:
            backend.duplex_finalize()
        except Exception:
            pass
        output = DuplexOutput(
            sample_id=run_item.sample.sample_id,
            error=ErrorInfo(message=str(exc), code=type(exc).__name__, retriable=False),
            units=units,
        )
    else:
        output = _make_output(
            run_item.sample.sample_id,
            units,
            output_audio,
            run_item.sample_dir / "assistant.wav",
        )

    metric_names = deps.get_registered_subbenchmark(
        "omnipro_online", run_item.sample.subbenchmark_name
    ).metrics
    records, skipped = deps.compute_metrics_for_sample(
        run_item.sample,
        output,
        metric_names,
        metric_context=deps.metric_context,
    )
    preview = _write_result(
        run_item=run_item,
        output=output,
        records=records,
        skipped=skipped,
        args=args,
        output_root=output_root,
    )
    print(json.dumps({"event": "sample_done", **preview}, ensure_ascii=False), flush=True)
    return run_item.sample, records, skipped, preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--backbone-dir", default=DEFAULT_BACKBONE)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--ref-audio", default=DEFAULT_REF_AUDIO)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--task", dest="duplex_task", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-input-seconds", type=float, default=0.0)
    parser.add_argument("--generate-audio", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--chunk-ms", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--decode-mode", choices=("sampling", "greedy"), default="sampling")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--tts-temperature", type=float, default=0.8)
    parser.add_argument("--tts-repetition-penalty", type=float, default=1.05)
    parser.add_argument("--listen-prob-scale", type=float, default=1.0)
    parser.add_argument("--text-repetition-penalty", type=float, default=1.05)
    parser.add_argument("--text-repetition-window-size", type=int, default=512)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--force-listen-count", type=int, default=0)
    parser.add_argument("--max-new-speak-tokens-per-chunk", type=int, default=1024)
    parser.add_argument("--n-timesteps", type=int, default=5)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--deployment-mode", choices=("tp2", "tp2_llm"), default="tp2")
    parser.add_argument("--experts-implementation", choices=("eager", "batched_mm"), default="batched_mm")
    parser.add_argument("--o5-llm-cache", type=int, default=65536)
    parser.add_argument("--llm-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tts-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vocoder-graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tts-fast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lmhead", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fuse-vision-audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-vision-feed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--enable-remote-judge", action="store_true")
    parser.add_argument("--judge-api-key", default=os.environ.get("JUDGE_API_KEY"))
    parser.add_argument("--judge-api-url", default=DEFAULT_JUDGE_URL)
    parser.add_argument("--judge-model", default="gemini-3-flash-preview")
    parser.add_argument("--judge-max-retry", type=int, default=2)
    parser.add_argument("--judge-sleep-between-retry", type=int, default=5)
    parser.add_argument("--judge-max-tokens", type=int, default=1024)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-top-p", type=float, default=1.0)
    parser.add_argument("--judge-seed", type=int, default=0)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if args.chunk_ms <= 0:
        parser.error("--chunk-ms must be positive")
    for name, path in (
        ("model-path", args.model_path),
        ("checkpoint-path", args.checkpoint_path),
        ("backbone-dir", args.backbone_dir),
        ("data-root", args.data_root),
    ):
        if not Path(path).exists():
            parser.error(f"{name} does not exist: {path}")
    if args.generate_audio and not Path(args.ref_audio).exists():
        parser.error(f"ref-audio does not exist: {args.ref_audio}")
    return args


def main() -> int:
    args = parse_args()
    os.environ["HEK_OMNIPRO_ROOT"] = args.data_root
    os.environ["O5_LLM_CACHE"] = str(args.o5_llm_cache)
    os.environ["DEMO_GIT_COMMIT"] = os.popen(f"git -C {REPO_ROOT} rev-parse HEAD").read().strip()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    args.ckpt_path = args.checkpoint_path
    _apply_safe_engine_env(args)

    from humanevalkit.benchmarks.pipeline import load_benchmark_rows, prepare_run_item
    from humanevalkit.benchmarks.registry import get_registered_subbenchmark
    from humanevalkit.metrics import aggregate_metric_records, compute_metrics_for_sample, summarize_skipped_metrics
    from humanevalkit.schemas.outputs import DuplexOutputUnit

    deps = SimpleNamespace(
        load_benchmark_rows=load_benchmark_rows,
        prepare_run_item=prepare_run_item,
        get_registered_subbenchmark=get_registered_subbenchmark,
        compute_metrics_for_sample=compute_metrics_for_sample,
        DuplexOutputUnit=DuplexOutputUnit,
        split_audio=split_audio,
        metric_context={
            "enable_remote_judge": bool(args.enable_remote_judge and args.judge_api_key),
            "judge_model": args.judge_model,
            "judge_api_url": args.judge_api_url,
            "judge_api_key": args.judge_api_key,
            "judge_max_retry": args.judge_max_retry,
            "judge_sleep_between_retry": args.judge_sleep_between_retry,
            "judge_max_tokens": args.judge_max_tokens,
            "judge_temperature": args.judge_temperature,
            "judge_top_p": args.judge_top_p,
            "judge_seed": args.judge_seed,
        },
    )

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    shard_rows = deps.load_benchmark_rows(
        benchmark_name="omnipro_online",
        start_index=args.start_index,
        limit=args.limit,
        duplex_task=args.duplex_task,
    )
    if args.num_shards > 1:
        shard_rows = shard_rows[args.shard_index :: args.num_shards]
    print(
        json.dumps(
            {
                "event": "run_start",
                "benchmark": "omnipro_online",
                "sample_count": len(shard_rows),
                "shard": f"{args.shard_index}/{args.num_shards}",
                "output_dir": str(output_root),
                "demo_commit": os.environ["DEMO_GIT_COMMIT"],
                "engine": _engine_config(args),
                "sampling": _sampling_config(args),
                "remote_judge": deps.metric_context["enable_remote_judge"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if not shard_rows:
        raise RuntimeError("OmniPro selection returned zero samples")

    backend = _build_backend(args)
    if getattr(backend, "spmd_is_worker", False):
        _run_backend_worker_loop(backend)

    processed: list[Any] = []
    sample_records: list[Any] = []
    skipped_metrics: list[dict[str, Any]] = []
    previews: list[dict[str, Any]] = []
    try:
        for local_index, raw_row in enumerate(shard_rows):
            row_index = int(raw_row.get("_row_index", args.start_index + local_index))
            sample, records, skipped, preview = run_one(args, backend, raw_row, row_index, deps)
            processed.append(sample)
            sample_records.extend(records)
            skipped_metrics.extend(skipped)
            previews.append(preview)

        aggregate_records, metrics_summary = aggregate_metric_records(sample_records)
        report = {
            "benchmark_family": "omni_proact",
            "benchmark_name": "omnipro_online",
            "processed_sample_count": len(processed),
            "error_sample_count": sum(1 for item in previews if item.get("error")),
            "samples": previews,
            "sample_metric_records": [record.model_dump(mode="json") for record in sample_records],
            "aggregate_metric_records": [record.model_dump(mode="json") for record in aggregate_records],
            "metrics_summary": metrics_summary,
            "skipped_metrics": summarize_skipped_metrics(skipped_metrics),
            "run_config": {
                "model_path": args.model_path,
                "checkpoint_path": args.checkpoint_path,
                "backbone_dir": args.backbone_dir,
                "data_root": args.data_root,
                "demo_commit": os.environ["DEMO_GIT_COMMIT"],
                "engine": _engine_config(args),
                "sampling": _sampling_config(args),
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
                "start_index": args.start_index,
                "limit": args.limit,
                "remote_judge": deps.metric_context["enable_remote_judge"],
            },
        }
        (output_root / "run_report.json").write_text(
            json.dumps(_jsonable(report), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {"event": "run_done", "samples": len(processed), "metrics": _jsonable(metrics_summary)},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return 0
    finally:
        _shutdown_backend(backend)


if __name__ == "__main__":
    raise SystemExit(main())
