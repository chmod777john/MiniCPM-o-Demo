"""Overfit/Full TrainingData 多 Semantic API endpoint 批量评测协调器。

本模块只在 CPU devbox 上负责确定性发现 case、按 endpoint 启动常驻 async worker、
顺序调用现有 ``FcApiTrainingDataEvaluator``、原子落盘单样本终态并生成汇总。它不
复制单样本 evaluator 的 token/parser 逻辑，也不通过 subprocess 调用 CLI。
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

from minicpm_o5_sdk import O5DuplexTrainingData
from pydantic import BaseModel, ConfigDict, Field

from .models import (
    FcApiCheckpointProfile,
    FcApiTrainingDataEvaluationResult,
    JsonObject,
)


class FcApiBatchTask(BaseModel):
    """一个由文件名稳定标识的批量任务。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    source_path: Path


class FcApiBatchTaskError(BaseModel):
    """批量协调层的一条结构化错误。"""

    model_config = ConfigDict(extra="forbid")

    error_type: str
    message: str
    attempt: int = Field(ge=1)
    detail: JsonObject | None = None


class FcApiBatchSampleResult(BaseModel):
    """单样本原子终态；成功和失败都可被断点续跑识别。"""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    source_path: Path
    endpoint: str
    status: Literal["succeeded", "failed"]
    attempts: int = Field(ge=1)
    elapsed_sec: float = Field(ge=0.0)
    evaluation: FcApiTrainingDataEvaluationResult | None = None
    error: FcApiBatchTaskError | None = None


class FcApiBatchSummary(BaseModel):
    """整个 batch 的稳定聚合指标。"""

    model_config = ConfigDict(extra="forbid")

    total: int = Field(ge=0)
    succeeded: int = Field(ge=0)
    execution_ok: int = Field(ge=0)
    full_exact: int = Field(ge=0)
    unit_all_exact: int = Field(ge=0)
    parser_validate: int = Field(ge=0)
    parser_roundtrip: int = Field(ge=0)
    failed: int = Field(ge=0)
    failed_task_ids: list[str] = Field(default_factory=list)
    non_exact: list[str] = Field(default_factory=list)
    endpoint_processed: dict[str, int] = Field(default_factory=dict)
    skipped: int = Field(ge=0)
    elapsed_sec: float = Field(ge=0.0)

    @property
    def exit_ok(self) -> bool:
        """返回 batch 是否满足模型/API exact 退出门禁。

        parser round-trip 仅作为独立统计，不参与退出码判断。
        """

        return (
            self.total > 0
            and self.execution_ok == self.total
            and self.full_exact == self.total
        )


class FcApiBatchConfig(BaseModel):
    """批量协调器的固定配置。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    case_folder: Path
    base_urls: list[str] = Field(min_length=1)
    output_dir: Path
    retry: int = Field(default=2, ge=0)


class FcApiBatchEvaluatorProtocol(Protocol):
    """批量 worker 依赖的最小 evaluator 接口。"""

    async def evaluate(
        self,
        training_data: O5DuplexTrainingData,
    ) -> FcApiTrainingDataEvaluationResult:
        """执行一条 TrainingData 评测。"""


FcApiBatchEvaluatorFactory = Callable[
    [str, Path],
    FcApiBatchEvaluatorProtocol,
]
FcApiTrainingDataLoader = Callable[[Path], O5DuplexTrainingData]


class FcApiBatchRunner:
    """一个 endpoint 一个 worker、worker 内严格串行的 batch runner。"""

    def __init__(
        self,
        *,
        config: FcApiBatchConfig,
        evaluator_factory: FcApiBatchEvaluatorFactory,
        training_data_loader: FcApiTrainingDataLoader = (
            lambda path: load_training_data(path)
        ),
    ) -> None:
        """初始化 runner。

        参数:
            config: case、endpoint、输出和重试配置。
            evaluator_factory: 根据 endpoint 和数据根目录创建 evaluator。
            training_data_loader: 单样本 SDK TrainingData 加载函数，测试可注入。
        """

        self.config = config
        self.evaluator_factory = evaluator_factory
        self.training_data_loader = training_data_loader

    async def run(
        self,
        tasks: Sequence[FcApiBatchTask],
    ) -> FcApiBatchSummary:
        """执行任务并原子写入样本终态与 ``summary.json``。

        参数:
            tasks: 已按 ``task_id`` 排序且无重复 ID 的任务。

        返回:
            包含 exact、parser、失败、endpoint 和耗时统计的汇总。
        """

        started_at = time.monotonic()
        samples_dir = self.config.output_dir / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)
        result_by_task_id: dict[str, FcApiBatchSampleResult] = {}
        pending: list[FcApiBatchTask] = []
        skipped = 0

        for task in tasks:
            resumed = read_terminal_sample_result(
                sample_result_path(self.config.output_dir, task.task_id)
            )
            if resumed is None:
                pending.append(task)
                continue
            result_by_task_id[task.task_id] = resumed
            skipped += 1

        queue: asyncio.Queue[FcApiBatchTask] = asyncio.Queue()
        for task in pending:
            queue.put_nowait(task)

        workers = [
            asyncio.create_task(
                self._worker(
                    endpoint=endpoint,
                    queue=queue,
                    result_by_task_id=result_by_task_id,
                )
            )
            for endpoint in self.config.base_urls
        ]
        await asyncio.gather(*workers)

        ordered_results = [
            result_by_task_id[task.task_id]
            for task in tasks
            if task.task_id in result_by_task_id
        ]
        summary = summarize_batch_results(
            ordered_results,
            skipped=skipped,
            elapsed_sec=time.monotonic() - started_at,
        )
        atomic_write_model_json(
            self.config.output_dir / "summary.json",
            summary,
        )
        return summary

    async def _worker(
        self,
        *,
        endpoint: str,
        queue: asyncio.Queue[FcApiBatchTask],
        result_by_task_id: dict[str, FcApiBatchSampleResult],
    ) -> None:
        """在一个 endpoint 上顺序消费共享任务队列。"""

        evaluator = self.evaluator_factory(
            endpoint,
            self.config.case_folder,
        )
        while True:
            try:
                task = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                result = await self._evaluate_with_retry(
                    task=task,
                    endpoint=endpoint,
                    evaluator=evaluator,
                )
            except Exception as exc:
                result = _unexpected_worker_failure(
                    task=task,
                    endpoint=endpoint,
                    exc=exc,
                )

            result_by_task_id[task.task_id] = result
            try:
                atomic_write_model_json(
                    sample_result_path(
                        self.config.output_dir,
                        task.task_id,
                    ),
                    result,
                )
            except Exception as exc:
                # 单个输出目录异常不能中止其余 endpoint/任务；内存 summary 仍保留失败。
                result_by_task_id[task.task_id] = result.model_copy(
                    update={
                        "status": "failed",
                        "error": FcApiBatchTaskError(
                            error_type="atomic_output_error",
                            message=f"{type(exc).__name__}: {exc}",
                            attempt=result.attempts,
                        ),
                    }
                )
            finally:
                queue.task_done()

    async def _evaluate_with_retry(
        self,
        *,
        task: FcApiBatchTask,
        endpoint: str,
        evaluator: FcApiBatchEvaluatorProtocol,
    ) -> FcApiBatchSampleResult:
        """在同一 endpoint 上执行单任务，结构性失败最多重试指定次数。"""

        started_at = time.monotonic()
        max_attempts = self.config.retry + 1
        last_evaluation: FcApiTrainingDataEvaluationResult | None = None
        last_error: FcApiBatchTaskError | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                training_data = self.training_data_loader(task.source_path)
                evaluation = await evaluator.evaluate(training_data)
                last_evaluation = evaluation
                if evaluation.execution_ok:
                    return FcApiBatchSampleResult(
                        task_id=task.task_id,
                        source_path=task.source_path,
                        endpoint=endpoint,
                        status="succeeded",
                        attempts=attempt,
                        elapsed_sec=_rounded_elapsed(started_at),
                        evaluation=evaluation,
                    )
                last_error = FcApiBatchTaskError(
                    error_type="evaluation_failed",
                    message="evaluator 返回结构性失败",
                    attempt=attempt,
                    detail={
                        "errors": [
                            error.model_dump(mode="json")
                            for error in evaluation.errors
                        ],
                        "reconstruction_errors": (
                            []
                            if evaluation.reconstruction is None
                            else [
                                error.model_dump(mode="json")
                                for error in evaluation.reconstruction.errors
                            ]
                        ),
                    },
                )
            except Exception as exc:
                last_error = FcApiBatchTaskError(
                    error_type=type(exc).__name__,
                    message=str(exc),
                    attempt=attempt,
                )

        return FcApiBatchSampleResult(
            task_id=task.task_id,
            source_path=task.source_path,
            endpoint=endpoint,
            status="failed",
            attempts=max_attempts,
            elapsed_sec=_rounded_elapsed(started_at),
            evaluation=last_evaluation,
            error=last_error,
        )


def discover_training_data_tasks(
    case_folder: Path,
    *,
    limit: int | None = None,
) -> list[FcApiBatchTask]:
    """稳定发现 case folder 顶层 TrainingData JSON。

    参数:
        case_folder: 包含 TrainingData JSON 的目录。
        limit: 排序和过滤后最多保留的任务数；``None`` 表示不限。

    返回:
        按文件名排序、以文件 stem 作为稳定 ``task_id`` 的任务。

    异常:
        ValueError: limit 为负数、目录不存在或出现重复 task_id。
    """

    if limit is not None and limit < 0:
        raise ValueError("limit 必须大于等于 0")
    if not case_folder.is_dir():
        raise ValueError(f"case folder 不存在或不是目录: {case_folder}")

    tasks: list[FcApiBatchTask] = []
    for path in sorted(case_folder.glob("*.json"), key=lambda item: item.name):
        if path.stem.startswith("_meta"):
            continue
        structure = _read_json_object(path)
        if structure is None or not _looks_like_training_data(structure):
            continue
        tasks.append(
            FcApiBatchTask(
                task_id=path.stem,
                source_path=path,
            )
        )
        if limit is not None and len(tasks) >= limit:
            break

    task_ids = [task.task_id for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("case folder 中存在重复文件 stem")
    return tasks


def load_training_data(path: Path) -> O5DuplexTrainingData:
    """通过 SDK public API 加载一条 TrainingData。

    参数:
        path: TrainingData JSON 路径。

    返回:
        已绑定相对媒体根目录的 SDK TrainingData。
    """

    structure = json.loads(path.read_text(encoding="utf-8"))
    return O5DuplexTrainingData.load_structure(
        structure,
        data_root=path.parent,
    )


def sample_result_path(output_dir: Path, task_id: str) -> Path:
    """返回单任务约定的终态结果路径。"""

    return output_dir / "samples" / task_id / "result.json"


def read_terminal_sample_result(
    path: Path,
) -> FcApiBatchSampleResult | None:
    """读取可恢复的原子终态；不存在或损坏时返回 ``None``。"""

    if not path.is_file():
        return None
    try:
        return FcApiBatchSampleResult.model_validate_json(
            path.read_text(encoding="utf-8")
        )
    except Exception:
        return None


def summarize_batch_results(
    results: Sequence[FcApiBatchSampleResult],
    *,
    skipped: int,
    elapsed_sec: float,
) -> FcApiBatchSummary:
    """从单样本终态聚合 batch 指标，所有列表和 endpoint key 稳定排序。"""

    ordered = sorted(results, key=lambda item: item.task_id)
    endpoint_processed: dict[str, int] = {}
    for item in ordered:
        endpoint_processed[item.endpoint] = (
            endpoint_processed.get(item.endpoint, 0) + 1
        )

    execution_ok = sum(
        item.evaluation is not None and item.evaluation.execution_ok
        for item in ordered
    )
    full_exact = sum(_is_full_exact(item) for item in ordered)
    unit_all_exact = sum(_is_unit_all_exact(item) for item in ordered)
    parser_validate = sum(_parser_validate_ok(item) for item in ordered)
    parser_roundtrip = sum(_parser_roundtrip_ok(item) for item in ordered)
    failed_task_ids = sorted(
        item.task_id for item in ordered if item.status == "failed"
    )
    non_exact = sorted(
        item.task_id
        for item in ordered
        if item.evaluation is not None
        and item.evaluation.execution_ok
        and not _is_full_exact(item)
    )
    succeeded = sum(item.status == "succeeded" for item in ordered)
    return FcApiBatchSummary(
        total=len(ordered),
        succeeded=succeeded,
        execution_ok=execution_ok,
        full_exact=full_exact,
        unit_all_exact=unit_all_exact,
        parser_validate=parser_validate,
        parser_roundtrip=parser_roundtrip,
        failed=len(failed_task_ids),
        failed_task_ids=failed_task_ids,
        non_exact=non_exact,
        endpoint_processed={
            endpoint: endpoint_processed[endpoint]
            for endpoint in sorted(endpoint_processed)
        },
        skipped=skipped,
        elapsed_sec=round(elapsed_sec, 6),
    )


def atomic_write_model_json(path: Path, model: BaseModel) -> None:
    """以同目录临时文件 + ``os.replace`` 原子写入稳定排序 JSON。

    参数:
        path: 最终 JSON 路径。
        model: 要序列化的 Pydantic 模型。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )
    payload = json.dumps(
        model.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    try:
        with temporary_path.open("x", encoding="utf-8") as file:
            file.write(payload)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def parse_base_urls(values: Sequence[str]) -> list[str]:
    """解析可重复或逗号分隔的 ``--base-url``，并稳定去重。"""

    urls: list[str] = []
    seen: set[str] = set()
    for value in values:
        for candidate in value.split(","):
            normalized = candidate.strip().rstrip("/")
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            urls.append(normalized)
    if not urls:
        raise ValueError("至少需要一个非空 base URL")
    return urls


def _read_json_object(path: Path) -> dict[str, Any] | None:
    """读取 JSON object；非 JSON 或非 object 返回 ``None``。"""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _looks_like_training_data(structure: dict[str, Any]) -> bool:
    """用 TrainingData 顶层固定字段过滤元数据和无关 JSON。"""

    return isinstance(structure.get("tracks"), dict) and isinstance(
        structure.get("unit_policy"),
        dict,
    )


def _is_full_exact(result: FcApiBatchSampleResult) -> bool:
    """返回单样本完整 token 是否 exact。"""

    reconstruction = (
        None if result.evaluation is None else result.evaluation.reconstruction
    )
    return reconstruction is not None and reconstruction.full_exact is True


def _is_unit_all_exact(result: FcApiBatchSampleResult) -> bool:
    """返回单样本所有已报告 Unit 是否 exact。"""

    reconstruction = (
        None if result.evaluation is None else result.evaluation.reconstruction
    )
    return (
        reconstruction is not None
        and bool(reconstruction.unit_exact)
        and all(reconstruction.unit_exact.values())
    )


def _parser_validate_ok(result: FcApiBatchSampleResult) -> bool:
    """返回单样本 parser validate 是否通过。"""

    reconstruction = (
        None if result.evaluation is None else result.evaluation.reconstruction
    )
    return (
        reconstruction is not None
        and reconstruction.parser_audit.validate_ok
    )


def _parser_roundtrip_ok(result: FcApiBatchSampleResult) -> bool:
    """返回单样本 parser round-trip 是否通过。"""

    reconstruction = (
        None if result.evaluation is None else result.evaluation.reconstruction
    )
    return (
        reconstruction is not None
        and reconstruction.parser_audit.roundtrip_ok
    )


def _unexpected_worker_failure(
    *,
    task: FcApiBatchTask,
    endpoint: str,
    exc: Exception,
) -> FcApiBatchSampleResult:
    """把 worker 未预期异常降为单任务结构化失败。"""

    return FcApiBatchSampleResult(
        task_id=task.task_id,
        source_path=task.source_path,
        endpoint=endpoint,
        status="failed",
        attempts=1,
        elapsed_sec=0.0,
        error=FcApiBatchTaskError(
            error_type=type(exc).__name__,
            message=str(exc),
            attempt=1,
        ),
    )


def _rounded_elapsed(started_at: float) -> float:
    """返回微秒精度的非负耗时秒数。"""

    return round(max(0.0, time.monotonic() - started_at), 6)
