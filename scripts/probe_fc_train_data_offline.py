#!/usr/bin/env python3
"""Run one FC duplex train-data sample through offline inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.processors import UnifiedProcessor
from core.schemas.duplex import DuplexConfig
from core.schemas.fc_duplex import FcDuplexConfig, FcDuplexTrainDataRequest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--pt-path", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--decode-mode", default="greedy")
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--extra-response-units", type=int, default=0)
    parser.add_argument("--tool-response-schedule", choices=("gt", "auto"), default="auto")
    parser.add_argument("--free-tool-call-ids", action="store_true")
    parser.add_argument("--no-train-tool-responses", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = UnifiedProcessor(
        model_path=args.model_path,
        pt_path=args.pt_path,
        device=args.device,
        compile=False,
        attn_implementation=args.attn_implementation,
        preload_both_tts=False,
        duplex_config=DuplexConfig(generate_audio=False),
    )
    fc = processor.set_fc_duplex_mode()

    config_kwargs = {
        "decode_mode": args.decode_mode,
        "extra_response_units": args.extra_response_units,
    }
    if args.budget is not None:
        config_kwargs["non_spoken_budget_per_unit"] = args.budget

    result = fc.offline_inference_from_train_data(
        FcDuplexTrainDataRequest(
            train_data_path=args.case,
            config=FcDuplexConfig(**config_kwargs),
            non_spoken_budget_per_unit=args.budget,
            generate_audio=False,
            output_artifact_dir=str(out_dir),
            use_train_tool_call_ids=not args.free_tool_call_ids,
            inject_train_tool_responses=not args.no_train_tool_responses,
            tool_response_schedule=args.tool_response_schedule,
        )
    )
    payload = result.model_dump()
    (out_dir / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "out_dir": str(out_dir),
        "success": payload.get("success"),
        "token_ids_exact": payload.get("token_ids_exact"),
        "rendered_token_stream_exact": payload.get("rendered_token_stream_exact"),
        "spoken_text_exact": payload.get("spoken_text_exact"),
        "think_text_exact": payload.get("think_text_exact"),
        "tool_calls_semantic_exact": payload.get("tool_calls_semantic_exact"),
        "tool_call_ids_exact": payload.get("tool_call_ids_exact"),
        "gt_tool_calls": payload.get("gt_tool_calls"),
        "pred_tool_calls": payload.get("pred_tool_calls"),
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
