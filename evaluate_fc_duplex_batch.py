#!/usr/bin/env python3
"""Batch-evaluate O5 FC duplex inference against training-data token streams."""

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.fc_duplex.system_input import FcAudioPathInput
from core.processors import FcDuplexView, UnifiedProcessor
from core.schemas.duplex import DuplexConfig
from core.schemas.fc_duplex import FcDuplexConfig, FcDuplexTrainDataRequest


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_BASE_MODEL = os.environ.get(
    "O5_FC_EVAL_MODEL_PATH",
    "/user/weihongliang/MiniCPM-o-4_6",
)
DEFAULT_PT_PATH = os.environ.get("O5_FC_EVAL_PT_PATH") or os.environ.get("PT_PATH") or ""
DEFAULT_DATA_DIR = os.environ.get(
    "O5_FC_EVAL_DATA_DIR",
    "/user/weihongliang/o5_fc_assets/board_mvp_20260724/delivery_train_data",
)
DEFAULT_OUTPUT_DIR = os.environ.get(
    "O5_FC_EVAL_OUTPUT_DIR",
    "/user/weihongliang/fc_board_offline_runs/20260725_iter500",
)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def make_sdk_compatible_train_data(structure: Dict[str, Any]) -> Dict[str, Any]:
    """Drop training-data metadata rejected by the current minicpm_o5_sdk schema."""

    compatible = copy.deepcopy(structure)
    tracks = compatible.get("tracks")
    if not isinstance(tracks, dict):
        return compatible

    for track_name in ("input_event", "ai_spoken", "ai_non_spoken"):
        track = tracks.get(track_name)
        if isinstance(track, dict):
            track.pop("allow_segment_overlap", None)
    return compatible


def strip_cli_options(argv: List[str], options: set[str]) -> List[str]:
    """Remove options that are replaced when launching worker processes."""

    stripped: List[str] = []
    index = 0
    while index < len(argv):
        item = argv[index]
        if item in options:
            index += 2
            continue
        if any(item.startswith(f"{option}=") for option in options):
            index += 1
            continue
        stripped.append(item)
        index += 1
    return stripped


def select_cuda_devices(gpu_num: int) -> List[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [device.strip() for device in visible.split(",") if device.strip()]
    else:
        devices = [str(index) for index in range(gpu_num)]
    if len(devices) < gpu_num:
        raise ValueError(
            f"--gpu-num={gpu_num} but only {len(devices)} CUDA devices are visible: "
            f"{visible!r}"
        )
    return devices[:gpu_num]


def merge_group_summaries(summaries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    summaries = [summary for summary in summaries if summary]
    if not summaries:
        return None

    list_keys = {
        "missing_tool_call_samples",
        "extra_tool_call_samples",
        "failed_samples",
        "token_diff_samples",
        "semantic_diff_samples",
    }
    bool_all_keys = {"tool_call_ids_exact_applicable"}
    merged: Dict[str, Any] = {"group": summaries[0].get("group")}

    keys = set().union(*(summary.keys() for summary in summaries))
    for key in sorted(keys):
        if key == "group":
            continue
        values = [summary.get(key) for summary in summaries]
        if key in list_keys:
            merged[key] = [
                item
                for value in values
                if isinstance(value, list)
                for item in value
            ]
        elif key in bool_all_keys:
            merged[key] = all(bool(value) for value in values)
        elif all(isinstance(value, int) and not isinstance(value, bool) for value in values):
            merged[key] = sum(values)
        else:
            merged[key] = values[0]
    return merged


def merge_parallel_results(output_dir: Path, gpu_num: int, args: argparse.Namespace) -> None:
    worker_summaries = []
    for worker_index in range(gpu_num):
        summary_path = output_dir / f"worker_{worker_index}" / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing worker summary: {summary_path}")
        worker_summaries.append(read_json(summary_path))

    original_summary = merge_group_summaries(
        [summary.get("original") for summary in worker_summaries]
    )
    modified_summary = merge_group_summaries(
        [
            summary.get("modified")
            for summary in worker_summaries
            if summary.get("modified") is not None
        ]
    )

    write_json(
        output_dir / "summary.json",
        {
            "original": original_summary,
            "modified": modified_summary,
            "workers": worker_summaries,
        },
    )
    write_json(output_dir / "run_config.json", vars(args))


def launch_parallel(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    devices = select_cuda_devices(args.gpu_num)
    script_path = Path(__file__).resolve()
    base_argv = strip_cli_options(
        sys.argv[1:],
        {"--gpu-num", "--num-workers", "--worker-index", "--output-dir"},
    )

    processes = []
    log_files = []
    print(f"[parallel] launching {args.gpu_num} workers", flush=True)
    for worker_index, device in enumerate(devices):
        worker_dir = output_dir / f"worker_{worker_index}"
        log_path = output_dir / f"worker_{worker_index}.log"
        cmd = [
            sys.executable,
            str(script_path),
            *base_argv,
            "--gpu-num",
            "1",
            "--num-workers",
            str(args.gpu_num),
            "--worker-index",
            str(worker_index),
            "--output-dir",
            str(worker_dir),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        log_file = log_path.open("w", encoding="utf-8")
        log_files.append(log_file)
        print(
            f"[parallel] worker {worker_index}/{args.gpu_num} "
            f"CUDA_VISIBLE_DEVICES={device} log={log_path}",
            flush=True,
        )
        processes.append(
            subprocess.Popen(
                cmd,
                cwd=str(script_path.parent),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        )

    failed = []
    try:
        for worker_index, process in enumerate(processes):
            exit_code = process.wait()
            if exit_code != 0:
                failed.append((worker_index, exit_code))
    finally:
        for log_file in log_files:
            log_file.close()

    if failed:
        raise RuntimeError(f"Parallel workers failed: {failed}")

    merge_parallel_results(output_dir, args.gpu_num, args)
    print(f"[parallel done] results written to {output_dir}", flush=True)


def run_one(
    fc: FcDuplexView,
    data_path: Path,
    output_dir: Path,
    budget: Optional[int],
    extra_response_units: int,
    decode_mode: str,
    generate_audio: bool,
    tts_prompt_audio_path: Optional[str],
    use_train_tool_call_ids: bool,
    inject_train_tool_responses: bool,
    tool_response_schedule: str,
) -> Dict[str, Any]:
    sample_dir = output_dir / data_path.stem
    sample_dir.mkdir(parents=True, exist_ok=True)

    config_kwargs: Dict[str, Any] = {
        "decode_mode": decode_mode,
        "extra_response_units": extra_response_units,
    }
    if budget is not None:
        config_kwargs["non_spoken_budget_per_unit"] = budget

    compatible_data_path = sample_dir / data_path.name
    write_json(
        compatible_data_path,
        make_sdk_compatible_train_data(read_json(data_path)),
    )

    # A None request-level override deliberately preserves the SDK arrangement's
    # per-unit listening/speaking budgets inside FcDuplexView.
    result = fc.offline_inference_from_train_data(
        FcDuplexTrainDataRequest(
            train_data_path=str(compatible_data_path),
            data_root=str(data_path.parent),
            config=FcDuplexConfig(**config_kwargs),
            non_spoken_budget_per_unit=budget,
            generate_audio=generate_audio,
            tts_prompt_audio=(
                FcAudioPathInput(file_path=tts_prompt_audio_path)
                if tts_prompt_audio_path
                else None
            ),
            output_artifact_dir=str(sample_dir),
            use_train_tool_call_ids=use_train_tool_call_ids,
            inject_train_tool_responses=inject_train_tool_responses,
            tool_response_schedule=tool_response_schedule,
        )
    )
    dumped = result.model_dump()
    if not dumped.get("success") and dumped.get("pred_output_render"):
        (sample_dir / "pred_prefix_token_stream.txt").write_text(
            dumped["pred_output_render"],
            encoding="utf-8",
        )
        write_json(sample_dir / "pred_prefix_token_ids.json", dumped.get("pred_output_ids") or [])
    write_json(sample_dir / "comparison.json", dumped)
    return dumped


def make_mutated_inputs(data_paths: List[Path], output_dir: Path) -> List[Path]:
    mutated_dir = output_dir / "modified_inputs"
    mutated_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []
    for index, source in enumerate(data_paths[:2], start=1):
        structure = read_json(source)
        structure["data_id"] = f"{structure.get('data_id', source.stem)}:modified_{index}"
        for segment in structure.get("system", {}).get("segments", []):
            if segment.get("kind") == "text":
                segment["text"] = (
                    segment.get("text", "")
                    + f"\n\n轻微改动测试 {index}：保持原规则不变，但表达更偏保守，只有非常明确的可见物体才展示。"
                )
                break
        if structure.get("system", {}).get("tools"):
            function = structure["system"]["tools"][0].get("function", {})
            function["description"] = (
                (function.get("description") or "") + f"（轻微改动测试 {index}）"
            )
        destination = mutated_dir / f"{source.stem}_modified_{index}.json"
        write_json(destination, structure)

        media_source = source.parent.parent / "media" / source.stem
        media_destination = destination.parent.parent / "media" / destination.stem
        if media_destination.exists():
            shutil.rmtree(media_destination)
        shutil.copytree(media_source, media_destination)
        outputs.append(destination)
    return outputs


def summarize(
    comparisons: List[Dict[str, Any]],
    output_dir: Path,
    group: str,
    evaluate_tool_call_ids: bool,
) -> Dict[str, Any]:
    def matched(item: Dict[str, Any], key: str) -> bool:
        return bool((item.get("comparison") or {}).get(key))

    def tool_call_names(item: Dict[str, Any], key: str) -> List[Any]:
        return [
            (call.get("arguments") or {}).get("name")
            for call in (item.get(key) or [])
        ]

    def has_missing_tool_calls(item: Dict[str, Any]) -> bool:
        predicted = tool_call_names(item, "pred_tool_calls")
        return any(name not in predicted for name in tool_call_names(item, "gt_tool_calls"))

    def has_extra_tool_calls(item: Dict[str, Any]) -> bool:
        expected = tool_call_names(item, "gt_tool_calls")
        return any(name not in expected for name in tool_call_names(item, "pred_tool_calls"))

    def semantic_diff(item: Dict[str, Any]) -> bool:
        comparison = item.get("comparison") or {}
        fields = ["spoken_text_exact", "think_text_exact", "tool_calls_semantic_exact"]
        if evaluate_tool_call_ids:
            fields.append("tool_call_ids_exact")
        return not all(bool(comparison.get(field)) for field in fields)

    summary = {
        "group": group,
        "total": len(comparisons),
        "success": sum(1 for item in comparisons if item.get("success")),
        "token_ids_exact": sum(1 for item in comparisons if matched(item, "token_ids_exact")),
        "rendered_token_stream_exact": sum(
            1 for item in comparisons if matched(item, "rendered_token_stream_exact")
        ),
        "spoken_text_exact": sum(1 for item in comparisons if matched(item, "spoken_text_exact")),
        "think_text_exact": sum(1 for item in comparisons if matched(item, "think_text_exact")),
        "tool_calls_semantic_exact": sum(
            1 for item in comparisons if matched(item, "tool_calls_semantic_exact")
        ),
        "tool_call_ids_exact": sum(
            1 for item in comparisons if matched(item, "tool_call_ids_exact")
        ),
        "tool_call_ids_exact_applicable": evaluate_tool_call_ids,
        "total_gt_tool_calls": sum(len(item.get("gt_tool_calls") or []) for item in comparisons),
        "total_pred_tool_calls": sum(
            len(item.get("pred_tool_calls") or []) for item in comparisons
        ),
        "missing_tool_call_samples": [
            item["sample_id"]
            for item in comparisons
            if item.get("success") and has_missing_tool_calls(item)
        ],
        "extra_tool_call_samples": [
            item["sample_id"]
            for item in comparisons
            if item.get("success") and has_extra_tool_calls(item)
        ],
        "failed_samples": [
            item["sample_id"] for item in comparisons if not item.get("success")
        ],
        "token_diff_samples": [
            item["sample_id"]
            for item in comparisons
            if not (
                matched(item, "token_ids_exact")
                and matched(item, "rendered_token_stream_exact")
            )
        ],
        "semantic_diff_samples": [
            item["sample_id"] for item in comparisons if semantic_diff(item)
        ],
    }
    write_json(output_dir / f"{group}_summary.json", summary)
    return summary


def evaluate_group(
    fc: FcDuplexView,
    data_paths: List[Path],
    output_dir: Path,
    group: str,
    args: argparse.Namespace,
    generate_audio: bool,
    use_train_tool_call_ids: bool,
    inject_train_tool_responses: bool,
    tool_response_schedule: str,
) -> Dict[str, Any]:
    results = []
    group_dir = output_dir / group
    for index, path in enumerate(data_paths, start=1):
        print(f"[{group} {index:03d}/{len(data_paths):03d}] {path.name}", flush=True)
        started = time.perf_counter()
        try:
            comparison = run_one(
                fc=fc,
                data_path=path,
                output_dir=group_dir,
                budget=args.budget,
                extra_response_units=args.extra_response_units,
                decode_mode=args.decode_mode,
                generate_audio=generate_audio,
                tts_prompt_audio_path=args.tts_prompt_path,
                use_train_tool_call_ids=use_train_tool_call_ids,
                inject_train_tool_responses=inject_train_tool_responses,
                tool_response_schedule=tool_response_schedule,
            )
        except Exception as exception:
            comparison = {
                "sample_id": path.stem,
                "data_path": str(path),
                "success": False,
                "error": repr(exception),
                "comparison": {},
            }
            write_json(group_dir / path.stem / "comparison.json", comparison)
            print(f"  ERROR: {exception}", flush=True)
        print(f"  elapsed: {time.perf_counter() - started:.2f}s", flush=True)
        results.append(comparison)

    summary = summarize(
        results,
        output_dir,
        group,
        evaluate_tool_call_ids=use_train_tool_call_ids,
    )
    print(f"[{group} summary] {summary}", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch O5 FC duplex train/infer consistency evaluation"
    )
    parser.add_argument("--model-path", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--pt-path", default=DEFAULT_PT_PATH)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Debug override only; omit to preserve SDK arrangement per-unit budgets.",
    )
    parser.add_argument("--extra-response-units", type=int, default=0)
    parser.add_argument("--decode-mode", default="greedy")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-mutated", action="store_true")
    parser.add_argument(
        "--gpu-num",
        type=int,
        default=1,
        help="Number of GPUs to use for data-parallel evaluation.",
    )
    parser.add_argument("--num-workers", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--worker-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--free-tool-call-ids",
        action="store_true",
        help="Generate tool call ids freely; tool_call_ids_exact then does not apply.",
    )
    parser.add_argument(
        "--no-train-tool-responses",
        action="store_true",
        help="Do not inject ground-truth tool responses during offline inference.",
    )
    parser.add_argument(
        "--tool-response-schedule",
        choices=("gt", "auto"),
        default="gt",
        help=(
            "Schedule train-data tool responses by GT arrangement unit indices "
            "('gt') or by runtime auto-delay after predicted tool calls ('auto')."
        ),
    )
    parser.add_argument("--tts-prompt-path", default=None)
    args = parser.parse_args()
    if not args.pt_path:
        parser.error("--pt-path is required, or set O5_FC_EVAL_PT_PATH/PT_PATH")
    if args.gpu_num < 1:
        parser.error("--gpu-num must be >= 1")
    if args.num_workers < 1:
        parser.error("--num-workers must be >= 1")
    if not 0 <= args.worker_index < args.num_workers:
        parser.error("--worker-index must satisfy 0 <= worker_index < num_workers")
    return args


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.gpu_num > 1:
        launch_parallel(args)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_paths = sorted(Path(args.data_dir).glob("*.json"))
    if args.limit is not None:
        data_paths = data_paths[: args.limit]
    if args.num_workers > 1:
        data_paths = data_paths[args.worker_index :: args.num_workers]
        print(
            f"[worker] index={args.worker_index}/{args.num_workers} "
            f"samples={len(data_paths)} output={output_dir}",
            flush=True,
        )

    generate_audio = bool(args.tts_prompt_path)
    use_train_tool_call_ids = not args.free_tool_call_ids
    inject_train_tool_responses = not args.no_train_tool_responses
    print(f"[load] model={args.model_path}")
    print(f"[load] pt={args.pt_path}")
    print(
        f"[mode] tool_call_ids={'train-fixed' if use_train_tool_call_ids else 'free-generated'} "
        f"tool_responses={'train-injected' if inject_train_tool_responses else 'disabled'} "
        f"tool_response_schedule={args.tool_response_schedule}",
        flush=True,
    )
    processor = UnifiedProcessor(
        model_path=args.model_path,
        pt_path=args.pt_path,
        device=args.device,
        compile=False,
        attn_implementation=args.attn_implementation,
        preload_both_tts=generate_audio,
        duplex_config=DuplexConfig(generate_audio=generate_audio),
    )
    fc: FcDuplexView = processor.fc_duplex

    original_summary = evaluate_group(
        fc,
        data_paths,
        output_dir,
        "original",
        args,
        generate_audio,
        use_train_tool_call_ids,
        inject_train_tool_responses,
        args.tool_response_schedule,
    )
    modified_summary = None
    if not args.skip_mutated:
        modified_summary = evaluate_group(
            fc,
            make_mutated_inputs(data_paths, output_dir),
            output_dir,
            "modified",
            args,
            generate_audio,
            use_train_tool_call_ids,
            inject_train_tool_responses,
            args.tool_response_schedule,
        )

    write_json(output_dir / "run_config.json", vars(args))
    write_json(
        output_dir / "summary.json",
        {"original": original_summary, "modified": modified_summary},
    )
    print(f"[done] results written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
