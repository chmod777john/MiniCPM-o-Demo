#!/usr/bin/env python3
"""Run a small ODB split through the Demo TP2 backend.

The benchmark loader, output schema, metrics, and remote judge remain in
humanevalkit.  This file only replaces the model inference layer with the
Demo's ``core.deploy`` backend and serializes its chunk results as
``DuplexOutput``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
HEK_ROOT = Path(os.environ.get("HUMANEVALKIT_ROOT", "/user/weihongliang/humanevalkit-codeup"))
TRACE_DIR = REPO_ROOT / "tools" / "o5trace"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(TRACE_DIR))
sys.path.insert(0, str(HEK_ROOT / "src"))

from humanevalkit.benchmarks.pipeline import prepare_run_item  # noqa: E402
from humanevalkit.benchmarks.registry import get_registered_subbenchmark  # noqa: E402
from humanevalkit.benchmarks.omni_duplex.chaoqun_omn_bench.common import (  # noqa: E402
    ChaoqunOmnBenchPaths,
    load_chaoqun_omn_bench_rows,
)
from humanevalkit.metrics import (  # noqa: E402
    aggregate_metric_records,
    build_grouped_evaluation_summary,
    build_sample_metric_map,
    compute_metrics_for_sample,
    summarize_skipped_metrics,
)
from humanevalkit.schemas.common import AudioSpan, DuplexUnitType, ErrorInfo  # noqa: E402
from humanevalkit.schemas.outputs import DuplexOutput, DuplexOutputUnit  # noqa: E402
from tp2_duplex_video_probe import (  # noqa: E402
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
    _apply_safe_engine_env,
    _apply_backend_duplex_runtime_config,
    _backend_audio_waveform,
    _backend_result_value,
    _build_backend,
    _run_backend_worker_loop,
    _shutdown_backend,
)
from thin_duplex_video_probe import configure_seed, extract_video, split_audio  # noqa: E402


DEFAULT_MODEL_PATH = "/user/weihongliang/MiniCPM-o-4_6"
DEFAULT_CHECKPOINT = (
    "/user/weihongliang/o5_weights/"
    "joint_stage2_40rank_lessa2o2_audio_coni_from_merge_equal_start_gemm_iter_4000.pt"
)
DEFAULT_BACKBONE = (
    "/user/weihongliang/o5_weights/"
    "o5_backbone_hf_joint_stage2_40rank_lessa2o2_audio_coni_from_merge_equal_start_gemm_iter_4000"
)
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
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _sampling_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "generate_audio": args.generate_audio,
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


def _video_path(sample: Any) -> str:
    for track in sample.tracks:
        if str(track.modality) != "video" or not track.segments:
            continue
        payload = track.segments[0].payload
        path = getattr(payload, "video_path", None)
        if path:
            return str(path)
    raise ValueError(f"sample {sample.sample_id} has no video path")


def _make_output(
    sample_id: str,
    units: list[DuplexOutputUnit],
    audio_chunks: list[np.ndarray],
    audio_path: Path,
) -> DuplexOutput:
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
    output: DuplexOutput,
    metric_names: list[str],
    metric_context: dict[str, Any],
    output_root: Path,
) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]]:
    records, skipped = compute_metrics_for_sample(
        run_item.sample,
        output,
        metric_names,
        metric_context=metric_context,
    )
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
            "o5_llm_cache": int(os.environ.get("O5_LLM_CACHE", "0") or 0),
            "engine": _jsonable(metric_context.get("engine", {})),
            "sampling": _jsonable(metric_context.get("sampling", {})),
        },
    }
    result_path.write_text(json.dumps(_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    preview = {
        "sample_id": run_item.sample.sample_id,
        "benchmark_family": run_item.sample.benchmark_family,
        "benchmark_name": run_item.sample.benchmark_name,
        "subbenchmark_name": run_item.sample.subbenchmark_name,
        "benchmark_path": run_item.sample.benchmark_path,
        "result_path": str(result_path.relative_to(output_root)),
        "metric_values": build_sample_metric_map(records).get(run_item.sample.sample_id, {}),
        "output_preview": {
            "error": output.error.model_dump(mode="json") if output.error else None,
            "text": output.text,
            "audio_path": output.audio_path,
            "unit_count": len(output.units),
        },
    }
    return records, skipped, preview


def _infer_one(
    args: argparse.Namespace,
    backend: Any,
    raw_row: dict[str, Any],
    index: int,
) -> tuple[Any, DuplexOutput | None, list[str], dict[str, Any] | None]:
    output_root = Path(args.output_dir).resolve()
    run_item = prepare_run_item(
        benchmark_name="chaoqun_omn_bench",
        raw_sample=raw_row,
        output_dir=output_root,
        model_path=args.model_path,
        checkpoint_path=args.checkpoint_path,
        device="cuda:0",
        index=index,
        chunk_seconds=args.chunk_ms / 1000.0,
        system_prompt=args.system_prompt,
        duplex_generate_audio=args.generate_audio,
        duplex_ref_audio=args.ref_audio,
    )
    result_path = run_item.sample_dir / "result.json"
    if args.resume and result_path.exists():
        print(json.dumps({"event": "resume_skip", "sample_id": run_item.sample.sample_id}, ensure_ascii=False), flush=True)
        return run_item, None, [], {"sample_id": run_item.sample.sample_id, "resumed": True}

    sample_seed = args.seed + index
    configure_seed(sample_seed)
    seed_runtime = getattr(backend, "seed_runtime", None)
    if callable(seed_runtime):
        seed_runtime(sample_seed)

    video_path = Path(_video_path(run_item.sample))
    frames, input_audio = extract_video(video_path, run_item.sample_dir / "input_media", args.chunk_ms)
    chunk_len = int(INPUT_SAMPLE_RATE * args.chunk_ms / 1000)
    audio_units = int(np.ceil(len(input_audio) / float(chunk_len)))
    unit_count = max(len(frames), audio_units)
    chunks = split_audio(input_audio, unit_count, args.chunk_ms)
    ref_audio = args.ref_audio
    sampling = _sampling_config(args)
    backend.duplex_prepare(
        system_prompt_text=args.system_prompt,
        ref_audio_path=ref_audio,
        prompt_wav_path=ref_audio,
        sampling=sampling,
        llm_seed=sample_seed,
    )

    units: list[DuplexOutputUnit] = []
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
            audio_span: AudioSpan | None = None
            if waveform is not None and len(waveform) > 0:
                waveform = np.asarray(waveform, dtype=np.float32)
                output_audio.append(waveform)
                duration = len(waveform) / float(OUTPUT_SAMPLE_RATE)
                audio_span = AudioSpan(
                    start_time_sec=audio_offset,
                    end_time_sec=audio_offset + duration,
                )
                audio_offset += duration

            is_listen = bool(_backend_result_value(result, "is_listen", False))
            units.append(
                DuplexOutputUnit(
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

    metric_names = get_registered_subbenchmark(
        "chaoqun_omn_bench", run_item.sample.subbenchmark_name
    ).metrics
    return run_item, output, metric_names, None


def run_one(
    args: argparse.Namespace,
    backend: Any,
    raw_row: dict[str, Any],
    index: int,
) -> tuple[Any, list[Any], list[dict[str, Any]], dict[str, Any]]:
    output_root = Path(args.output_dir).resolve()
    run_item, output, metric_names, resumed_preview = _infer_one(args, backend, raw_row, index)
    if output is None:
        return run_item.sample, [], [], resumed_preview or {"sample_id": run_item.sample.sample_id}
    records, skipped, preview = _write_result(
        run_item=run_item,
        output=output,
        metric_names=metric_names,
        metric_context=args.metric_context,
        output_root=output_root,
    )
    print(json.dumps({"event": "sample_done", **preview}, ensure_ascii=False), flush=True)
    return run_item.sample, records, skipped, preview


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--checkpoint-path", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--backbone-dir", default=DEFAULT_BACKBONE)
    parser.add_argument("--ref-audio", default=DEFAULT_REF_AUDIO)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-root", default="/user/hechaoqun/final_data-v0")
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--task-name")
    parser.add_argument(
        "--event-only",
        action="store_true",
        help="keep only the three event-oriented ODB subsets (纠错/事件提醒/事件发生后提醒)",
    )
    parser.add_argument("--sww-only", action="store_true")
    parser.add_argument("--per-category", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--generate-audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--chunk-ms", type=int, default=1000)
    parser.add_argument("--system-prompt", default="Streaming Omni Conversation.")
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
    parser.add_argument("--length-penalty", type=float, default=1.1)
    parser.add_argument("--force-listen-count", type=int, default=0)
    parser.add_argument("--max-new-speak-tokens-per-chunk", type=int, default=20)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--deployment-mode", choices=("tp2",), default="tp2")
    parser.add_argument("--experts-implementation", choices=("eager", "batched_mm"), default="batched_mm")
    parser.add_argument("--o5-llm-cache", type=int, default=65536)
    parser.add_argument("--llm-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tts-graph", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vocoder-graph", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--tts-fast", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lmhead", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fuse-vision-audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-vision-feed", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--judge-api-key", default=os.environ.get("HEK_JUDGE_KEY") or os.environ.get("JUDGE_API_KEY"))
    parser.add_argument("--judge-api-url", default=DEFAULT_JUDGE_URL)
    parser.add_argument("--judge-model", default="GEMINI_8daxh7")
    parser.add_argument("--judge-max-retry", type=int, default=2)
    parser.add_argument("--judge-sleep-between-retry", type=int, default=5)
    parser.add_argument("--judge-max-tokens", type=int, default=1200)
    parser.add_argument("--judge-temperature", type=float, default=0.0)
    parser.add_argument("--judge-top-p", type=float, default=1.0)
    parser.add_argument("--judge-seed", type=int, default=0)
    parser.add_argument(
        "--judge-concurrency",
        type=int,
        default=0,
        help="rank0 background Judge workers per TP2 shard; 0 keeps synchronous judging",
    )
    parser.add_argument(
        "--judge-inflight-limit",
        type=int,
        default=0,
        help="maximum pending Judge samples per shard; 0 defaults to max(2 * concurrency, 4)",
    )
    parser.add_argument("--disable-judge", action="store_true")
    args = parser.parse_args()
    if args.judge_concurrency < 0:
        parser.error("--judge-concurrency must be non-negative")
    if args.judge_inflight_limit < 0:
        parser.error("--judge-inflight-limit must be non-negative")
    for path_arg in (args.model_path, args.checkpoint_path, args.backbone_dir, args.ref_audio):
        if not Path(path_arg).exists():
            parser.error(f"path does not exist: {path_arg}")
    if not Path(args.data_root).is_dir():
        parser.error(f"ODB data root does not exist: {args.data_root}")
    return args


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    os.environ["HEK_CHAOQUN_DATA_ROOT"] = args.data_root
    os.environ["O5_EXPERTS_IMPLEMENTATION"] = args.experts_implementation
    os.environ["O5_LLM_CACHE"] = str(args.o5_llm_cache)
    os.environ["DEMO_GIT_COMMIT"] = os.popen(f"git -C {REPO_ROOT} rev-parse HEAD").read().strip()
    # tp2_duplex_video_probe owns the deployment flag wiring.  Set every flag
    # before backend.load_model(), because core.deploy reads them during build.
    args.ckpt_path = args.checkpoint_path
    _apply_safe_engine_env(args)

    engine_config = {
        "deployment_mode": args.deployment_mode,
        "experts_implementation": args.experts_implementation,
        "llm_graph": args.llm_graph,
        "tts_graph": args.tts_graph,
        "vocoder_graph": args.vocoder_graph,
        "tts_fast": args.tts_fast,
        "lmhead": args.lmhead,
        "fuse_vision_audio": args.fuse_vision_audio,
        "batch_vision_feed": args.batch_vision_feed,
        "attn_implementation": args.attn_implementation,
        "o5_llm_cache": args.o5_llm_cache,
    }
    sampling_config = _sampling_config(args)

    args.metric_context = {
        "enable_remote_judge": bool(args.judge_api_key) and not args.disable_judge,
        "judge_model": args.judge_model,
        "judge_api_url": args.judge_api_url,
        "judge_api_key": args.judge_api_key,
        "judge_max_retry": args.judge_max_retry,
        "judge_sleep_between_retry": args.judge_sleep_between_retry,
        "judge_max_tokens": args.judge_max_tokens,
        "judge_temperature": args.judge_temperature,
        "judge_top_p": args.judge_top_p,
        "judge_seed": args.judge_seed,
        "engine": engine_config,
        "sampling": sampling_config,
    }

    paths = ChaoqunOmnBenchPaths()
    raw_rows = load_chaoqun_omn_bench_rows(
        paths,
        task_name=args.task_name,
        limit=None,
        sww_only=args.sww_only,
        per_category=args.per_category,
    )
    if args.event_only:
        event_subbenchmarks = {
            "event_correction",
            "event_reminder",
            "event_post_occurrence_reminder",
        }
        raw_rows = [row for row in raw_rows if row.get("subbenchmark_name") in event_subbenchmarks]
    raw_rows = raw_rows[args.start_index : args.start_index + args.limit]
    print(json.dumps({
        "event": "run_start",
        "sample_count": len(raw_rows),
        "output_dir": str(output_root),
        "demo_commit": os.environ["DEMO_GIT_COMMIT"],
        "judge_enabled": args.metric_context["enable_remote_judge"],
        "judge_key_present": bool(args.judge_api_key),
        "judge_concurrency": args.judge_concurrency,
        "judge_inflight_limit": (
            args.judge_inflight_limit
            if args.judge_inflight_limit > 0
            else max(2 * args.judge_concurrency, 4)
        ),
    }, ensure_ascii=False), flush=True)
    if not raw_rows:
        raise RuntimeError("ODB selection returned zero samples")

    backend = _build_backend(args)
    if getattr(backend, "spmd_is_worker", False):
        _run_backend_worker_loop(backend)

    processed_samples: list[Any] = []
    sample_records: list[Any] = []
    skipped_metrics: list[dict[str, Any]] = []
    sample_summaries: list[dict[str, Any]] = []
    judge_executor: ThreadPoolExecutor | None = None
    pending_judges: dict[Future, str] = {}
    judge_inflight_limit = (
        args.judge_inflight_limit
        if args.judge_inflight_limit > 0
        else max(2 * args.judge_concurrency, 4)
    )

    def collect_judges(done: set[Future]) -> None:
        for future in done:
            sample_id = pending_judges.pop(future)
            try:
                records, skipped, preview = future.result()
            except Exception as exc:
                raise RuntimeError(f"ODB Judge failed for sample {sample_id}") from exc
            sample_records.extend(records)
            skipped_metrics.extend(skipped)
            sample_summaries.append(preview)
            print(json.dumps({"event": "sample_done", **preview}, ensure_ascii=False), flush=True)

    def drain_judges(*, wait_for_all: bool) -> None:
        while pending_judges:
            if wait_for_all:
                done, _ = wait(pending_judges, return_when=FIRST_COMPLETED)
            else:
                if len(pending_judges) < judge_inflight_limit:
                    return
                done, _ = wait(pending_judges, return_when=FIRST_COMPLETED)
            collect_judges(done)

    def submit_judge(run_item: Any, output: DuplexOutput, metric_names: list[str]) -> None:
        if judge_executor is None:
            records, skipped, preview = _write_result(
                run_item=run_item,
                output=output,
                metric_names=metric_names,
                metric_context=args.metric_context,
                output_root=output_root,
            )
            sample_records.extend(records)
            skipped_metrics.extend(skipped)
            sample_summaries.append(preview)
            print(json.dumps({"event": "sample_done", **preview}, ensure_ascii=False), flush=True)
            return
        drain_judges(wait_for_all=False)
        future = judge_executor.submit(
            _write_result,
            run_item=run_item,
            output=output,
            metric_names=metric_names,
            metric_context=args.metric_context,
            output_root=output_root,
        )
        pending_judges[future] = run_item.sample.sample_id

    try:
        if args.judge_concurrency > 0 and args.metric_context["enable_remote_judge"]:
            judge_executor = ThreadPoolExecutor(
                max_workers=args.judge_concurrency,
                thread_name_prefix="judge",
            )
        for index, raw_row in enumerate(raw_rows, start=args.start_index):
            try:
                run_item, output, metric_names, resumed_preview = _infer_one(args, backend, raw_row, index)
            except Exception as exc:
                raise RuntimeError(f"ODB sample index {index} failed: {raw_row.get('sample_name')}") from exc
            processed_samples.append(run_item.sample)
            if output is None:
                sample_summaries.append(resumed_preview or {"sample_id": run_item.sample.sample_id})
                continue
            submit_judge(run_item, output, metric_names)

        drain_judges(wait_for_all=True)

        aggregate_records, metrics_summary = aggregate_metric_records(sample_records)
        run_report = {
            "benchmark_family": "omni_duplex",
            "benchmark_name": "chaoqun_omn_bench",
            "processed_sample_count": len(processed_samples),
            "error_sample_count": sum(1 for item in sample_summaries if item.get("output_preview", {}).get("error")),
            "samples": sample_summaries,
            "sample_metric_records": [record.model_dump(mode="json") for record in sample_records],
            "aggregate_metric_records": [record.model_dump(mode="json") for record in aggregate_records],
            "metrics_summary": metrics_summary,
            "grouped_evaluation_summary": build_grouped_evaluation_summary(processed_samples, sample_records),
            "skipped_metrics": summarize_skipped_metrics(skipped_metrics),
            "run_config": {
                "model_path": args.model_path,
                "checkpoint_path": args.checkpoint_path,
                "backbone_dir": args.backbone_dir,
                "attn_implementation": args.attn_implementation,
                "deployment_mode": args.deployment_mode,
                "experts_implementation": args.experts_implementation,
                "o5_llm_cache": args.o5_llm_cache,
                "llm_graph": args.llm_graph,
                "tts_graph": args.tts_graph,
                "tts_fast": args.tts_fast,
                "lmhead": args.lmhead,
                "fuse_vision_audio": args.fuse_vision_audio,
                "batch_vision_feed": args.batch_vision_feed,
                "vocoder_graph": args.vocoder_graph,
                "generate_audio": args.generate_audio,
                "event_only": args.event_only,
                "chunk_ms": args.chunk_ms,
                "seed": args.seed,
                "sampling": sampling_config,
                "judge_enabled": args.metric_context["enable_remote_judge"],
                "judge_model": args.judge_model,
                "judge_api_url": args.judge_api_url,
                "judge_concurrency": args.judge_concurrency,
                "judge_inflight_limit": judge_inflight_limit if judge_executor else None,
                "judge_pipeline": "rank0_background" if judge_executor else "synchronous",
            },
        }
        (output_root / "run_report.json").write_text(
            json.dumps(_jsonable(run_report), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps({"event": "run_done", "samples": len(processed_samples), "metrics": metrics_summary}, ensure_ascii=False), flush=True)
        return 0
    finally:
        if judge_executor is not None:
            judge_executor.shutdown(wait=True)
        _shutdown_backend(backend)


if __name__ == "__main__":
    raise SystemExit(main())
