"""并行调用多个 Semantic API endpoint 批量评测 TrainingData case folder。

每个 ``--base-url`` 对应一个 async worker，worker 内顺序处理任务；单样本终态原子写入
``<output>/samples/<file-stem>/result.json``，重复运行会跳过已有终态。退出码只由
execution/full-token exact 决定，parser round-trip 保持独立统计。
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from scripts.evaluate_fc_api_from_training_data import (
    DEFAULT_EXTERNAL_DEMO_ROOT,
    PRESET_PROFILE_PATHS,
    load_profile,
)
from scripts.fc_api_eval.batch_runner import (
    FcApiBatchConfig,
    FcApiBatchRunner,
    FcApiBatchSummary,
    discover_training_data_tasks,
    parse_base_urls,
)
from scripts.fc_api_eval.evaluator import (
    FcApiTrainingDataEvaluator,
)
from scripts.fc_api_eval.semantic_v2_client import (
    FcApiSemanticV2Client,
)


def build_argument_parser() -> argparse.ArgumentParser:
    """构造 batch CLI 参数解析器。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_folder", type=Path)
    parser.add_argument(
        "--base-url",
        action="append",
        required=True,
        help="可重复传入，也可在一个参数内使用逗号分隔",
    )
    parser.add_argument(
        "--preset",
        choices=tuple(PRESET_PROFILE_PATHS),
        default="overfit",
    )
    parser.add_argument("--profile-json", type=Path)
    parser.add_argument("--checkpoint-path", type=Path)
    parser.add_argument("--checkpoint-profile-id")
    parser.add_argument("--model")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--retry", type=int, default=2)
    parser.add_argument(
        "--external-demo-root",
        type=Path,
        default=DEFAULT_EXTERNAL_DEMO_ROOT,
    )
    parser.add_argument("--insecure", action="store_true")
    return parser


async def run(args: argparse.Namespace) -> tuple[Path, FcApiBatchSummary]:
    """执行 batch CLI。

    参数:
        args: ``build_argument_parser`` 解析后的参数。

    返回:
        ``summary.json`` 路径与强类型汇总。
    """

    if args.retry < 0:
        raise ValueError("retry 必须大于等于 0")
    base_urls = parse_base_urls(args.base_url)
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
    tasks = discover_training_data_tasks(
        args.case_folder,
        limit=args.limit,
    )
    config = FcApiBatchConfig(
        case_folder=args.case_folder,
        base_urls=base_urls,
        output_dir=args.output_dir,
        retry=args.retry,
    )

    def evaluator_factory(
        endpoint: str,
        data_root: Path,
    ) -> FcApiTrainingDataEvaluator:
        """为一个 endpoint 创建可跨样本复用的 client/evaluator。"""

        return FcApiTrainingDataEvaluator(
            client=FcApiSemanticV2Client(
                base_url=endpoint,
                insecure=args.insecure,
            ),
            profile=profile,
            data_root=data_root,
            external_demo_root=args.external_demo_root,
        )

    runner = FcApiBatchRunner(
        config=config,
        evaluator_factory=evaluator_factory,
    )
    summary = await runner.run(tasks)
    return args.output_dir / "summary.json", summary


def main() -> None:
    """CLI 入口，并按 execution/full exact 门禁返回退出码。"""

    args = build_argument_parser().parse_args()
    summary_path, summary = asyncio.run(run(args))
    print(summary_path)
    if not summary.exit_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
