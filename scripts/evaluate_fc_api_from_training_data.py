"""从 SDK 0.0.5 TrainingData 调用 Semantic API v2 并落盘 token round-trip 报告。

本 CLI 只运行 CPU 侧数据构造、WebSocket 调用和 token/parser 审计，不加载 checkpoint
或 GPU 模型。Overfit/Full preset 读取训练侧正式 Profile JSON；checkpoint 路径可由参数
覆盖，并只作为报告元信息保存。
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

from minicpm_o5_sdk import O5DuplexTrainingData, O5UnitPolicy

from scripts.fc_api_eval import (
    FcApiCheckpointProfile,
    FcApiSemanticV2Client,
    FcApiTrainingDataEvaluationResult,
    FcApiTrainingDataEvaluator,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
PRESET_PROFILE_PATHS = {
    "overfit": REPO_ROOT
    / "omni_agent_research/minicpm_o5_training/checkpoint_profiles/profiles/"
    "o45_fc_board_overfit100_sdk005_step100.json",
    "full": REPO_ROOT
    / "omni_agent_research/minicpm_o5_training/checkpoint_profiles/profiles/"
    "o45_fc_board_full4850_sdk005_step400.json",
}
DEFAULT_EXTERNAL_DEMO_ROOT = REPO_ROOT


def build_argument_parser() -> argparse.ArgumentParser:
    """构造 CLI 参数解析器。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_data", type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument(
        "--preset",
        choices=tuple(PRESET_PROFILE_PATHS),
        default="full",
    )
    parser.add_argument("--profile-json", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--checkpoint-profile-id")
    parser.add_argument("--model")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument(
        "--external-demo-root",
        type=Path,
        default=DEFAULT_EXTERNAL_DEMO_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "omni_agent_research/minicpm_o5_demo/"
            "runs/api_training_data_eval"
        ),
    )
    parser.add_argument("--insecure", action="store_true")
    return parser


def load_profile(
    *,
    profile_path: Path,
    checkpoint_override: Path | None,
    profile_id_override: str | None,
    model: str | None,
) -> FcApiCheckpointProfile:
    """读取训练侧正式 Profile 并投影为 evaluator 元信息。

    参数:
        profile_path: 训练侧 checkpoint Profile JSON。
        checkpoint_override: 可选 checkpoint 路径覆盖。
        profile_id_override: 可选 API endpoint Profile ID 覆盖。
        model: 可选 API model 身份覆盖；默认使用 Profile 的绝对 base model 路径。

    返回:
        通用 evaluator 使用的强类型 Profile。
    """

    structure = json.loads(profile_path.read_text(encoding="utf-8"))
    identity = _require_object(structure, "identity")
    protocol = _require_object(structure, "protocol")
    deployment = _require_object(structure, "deployment")
    unit_policy = O5UnitPolicy.model_validate(
        _require_object(protocol, "unit_policy")
    )
    raw_checkpoint = checkpoint_override or Path(
        str(identity["checkpoint_path"])
    )
    checkpoint_path = (
        raw_checkpoint
        if raw_checkpoint.is_absolute()
        else (REPO_ROOT / raw_checkpoint).resolve()
    )
    raw_model_path = Path(str(deployment["base_model_path"]))
    model_path = (
        raw_model_path
        if raw_model_path.is_absolute()
        else (REPO_ROOT / raw_model_path).resolve()
    )
    tokenizer_target = cast(
        Literal["o45_fc", "o5"],
        str(protocol["tokenizer_target"]),
    )
    non_spoken_scheduling = cast(
        Literal["quality", "latency"],
        str(deployment["non_spoken_scheduling"]),
    )
    return FcApiCheckpointProfile(
        profile_id=(
            profile_id_override
            or str(structure["profile_id"])
        ),
        model=model or str(model_path),
        checkpoint_path=checkpoint_path,
        training_job=str(identity["training_job"]),
        sdk_version=str(protocol["sdk_version"]),
        tokenizer_target=tokenizer_target,
        tool_serializer_name=str(protocol["tool_serializer"]),
        non_spoken_scheduling=non_spoken_scheduling,
        expected_unit_policy=unit_policy,
    )


def infer_data_root(training_data_path: Path) -> Path:
    """根据 TrainingData JSON 位置推断媒体根目录。

    参数:
        training_data_path: 单条 TrainingData JSON 路径。

    返回:
        ``load_structure(data_root=...)`` 使用的目录。
    """

    if training_data_path.parent.name in {
        "training_data",
        "delivery_train_data",
    }:
        return training_data_path.parent
    return training_data_path.parent


async def run(
    args: argparse.Namespace,
) -> tuple[Path, FcApiTrainingDataEvaluationResult]:
    """执行一次评测并写入 ignored runs 目录。

    参数:
        args: CLI 解析后的参数。

    返回:
        写出的 ``result.json`` 路径与结构化评测结果。
    """

    profile_path = (
        args.profile_json
        if args.profile_json is not None
        else PRESET_PROFILE_PATHS[args.preset]
    )
    profile = load_profile(
        profile_path=profile_path,
        checkpoint_override=args.checkpoint_path,
        profile_id_override=args.checkpoint_profile_id,
        model=args.model,
    )
    data_root = args.data_root or infer_data_root(args.training_data)
    structure = json.loads(
        args.training_data.read_text(encoding="utf-8")
    )
    training_data = O5DuplexTrainingData.load_structure(
        structure,
        data_root=data_root,
    )
    client = FcApiSemanticV2Client(
        base_url=args.base_url,
        insecure=args.insecure,
    )
    evaluator = FcApiTrainingDataEvaluator(
        client=client,
        profile=profile,
        data_root=data_root,
        external_demo_root=args.external_demo_root,
    )
    result = await evaluator.evaluate(training_data)
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "__"
        + args.training_data.stem
        + "__"
        + profile.profile_id
    )
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    result_path = run_dir / "result.json"
    result_path.write_text(
        result.model_dump_json(indent=2),
        encoding="utf-8",
    )
    request_meta = {
        "training_data": str(args.training_data.resolve()),
        "data_root": str(data_root.resolve()),
        "base_url": args.base_url,
        "profile_source": str(profile_path.resolve()),
        "profile": profile.model_dump(mode="json"),
    }
    (run_dir / "request.json").write_text(
        json.dumps(request_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result_path, result


def _require_object(
    parent: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    """读取必需 JSON object 字段。"""

    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"Profile {key} 必须是 JSON object")
    return value


def main() -> None:
    """CLI 入口。"""

    args = build_argument_parser().parse_args()
    result_path, result = asyncio.run(run(args))
    print(result_path)
    reconstruction = result.reconstruction
    if (
        not result.execution_ok
        or reconstruction is None
        or reconstruction.full_exact is not True
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
