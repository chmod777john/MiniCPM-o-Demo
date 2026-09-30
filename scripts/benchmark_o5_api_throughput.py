"""用一个固定 TrainingData 样例测量 O5 API v3 副本池吞吐。

每个副本先执行一次并发 warmup；正式阶段按轮次运行，每轮并发 Session 数等于副本数，
确保所有 Worker 各承载一个 Session 且不在 Worker 内复用并发。结果记录总墙钟吞吐、
单 Session 延迟分布、失败和首条完整协议 history。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from minicpm_o5_sdk import O5DuplexTrainingData
from pydantic import BaseModel, ConfigDict, Field

from examples.fc_duplex.infer_from_training_data import (
    ApiHistoryEntry,
    ApiInferenceClient,
    ApiInferenceScenario,
    build_api_inference_scenario,
    load_training_data_structure,
)


class SessionMeasurement(BaseModel):
    """一次 Session 的耗时与协议完整性摘要。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    round_index: int = Field(ge=0)
    slot_index: int = Field(ge=0)
    elapsed_sec: float = Field(ge=0)
    committed_unit_count: int = Field(ge=0)
    event_count: int = Field(ge=0)
    protocol_sha256: str
    error: str | None = None


class ThroughputBenchmarkResult(BaseModel):
    """一个部署拓扑的完整吞吐结果。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "o5_api_throughput_benchmark.v1"
    base_url: str
    data_id: str | None
    replica_count: int = Field(ge=1)
    repetitions: int = Field(ge=1)
    warmup_success_count: int = Field(ge=0)
    total_session_count: int = Field(ge=1)
    success_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    measured_wall_sec: float = Field(ge=0)
    sessions_per_sec: float = Field(ge=0)
    audio_sec_per_wall_sec: float = Field(ge=0)
    mean_session_elapsed_sec: float | None = None
    median_session_elapsed_sec: float | None = None
    p95_session_elapsed_sec: float | None = None
    measurements: tuple[SessionMeasurement, ...]


def _canonical_protocol_hash(history: list[ApiHistoryEntry]) -> str:
    """计算去除 Session 随机身份后的协议事件哈希。"""

    ignored_keys = {
        "session_id",
        "created_at",
        "timestamp",
        "queue_position",
        "estimated_wait",
        "metrics",
    }

    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: normalize(item)
                for key, item in sorted(value.items())
                if key not in ignored_keys
            }
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    protocol = [
        {
            "direction": entry.direction,
            "event": normalize(entry.event),
        }
        for entry in history
    ]
    payload = json.dumps(
        protocol,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


async def _run_one(
    *,
    client: ApiInferenceClient,
    scenario: ApiInferenceScenario,
    round_index: int,
    slot_index: int,
) -> tuple[SessionMeasurement, list[ApiHistoryEntry] | None]:
    started = time.perf_counter()
    try:
        session = await client.infer(scenario)
    except Exception as exc:  # noqa: BLE001
        return (
            SessionMeasurement(
                round_index=round_index,
                slot_index=slot_index,
                elapsed_sec=time.perf_counter() - started,
                committed_unit_count=0,
                event_count=0,
                protocol_sha256="",
                error=f"{type(exc).__name__}: {exc}",
            ),
            None,
        )
    elapsed = time.perf_counter() - started
    return (
        SessionMeasurement(
            round_index=round_index,
            slot_index=slot_index,
            elapsed_sec=elapsed,
            committed_unit_count=len(session.committed_unit_indices),
            event_count=len(session.history),
            protocol_sha256=_canonical_protocol_hash(session.history),
        ),
        session.history,
    )


async def _run_round(
    *,
    client: ApiInferenceClient,
    scenario: ApiInferenceScenario,
    round_index: int,
    replica_count: int,
) -> list[tuple[SessionMeasurement, list[ApiHistoryEntry] | None]]:
    return list(
        await asyncio.gather(
            *[
                _run_one(
                    client=client,
                    scenario=scenario,
                    round_index=round_index,
                    slot_index=slot_index,
                )
                for slot_index in range(replica_count)
            ]
        )
    )


def _percentile_95(values: list[float]) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[94]


async def benchmark(
    *,
    base_url: str,
    scenario: ApiInferenceScenario,
    replica_count: int,
    repetitions: int,
    insecure: bool,
) -> tuple[ThroughputBenchmarkResult, list[ApiHistoryEntry] | None]:
    """执行 warmup 和固定轮次吞吐测试。"""

    client = ApiInferenceClient(
        base_url=base_url,
        insecure=insecure,
        queue_timeout_sec=1200.0,
        event_timeout_sec=1200.0,
    )
    warmup = await _run_round(
        client=client,
        scenario=scenario,
        round_index=0,
        replica_count=replica_count,
    )
    warmup_success_count = sum(
        measurement.error is None for measurement, _history in warmup
    )
    if warmup_success_count != replica_count:
        failures = [
            measurement.error
            for measurement, _history in warmup
            if measurement.error is not None
        ]
        raise RuntimeError(f"warmup 未全部通过: {failures}")

    measurements: list[SessionMeasurement] = []
    first_history: list[ApiHistoryEntry] | None = None
    measured_started = time.perf_counter()
    for repetition in range(repetitions):
        round_results = await _run_round(
            client=client,
            scenario=scenario,
            round_index=repetition + 1,
            replica_count=replica_count,
        )
        for measurement, history in round_results:
            measurements.append(measurement)
            if first_history is None and history is not None:
                first_history = history
    measured_wall_sec = time.perf_counter() - measured_started

    successful_elapsed = [
        item.elapsed_sec for item in measurements if item.error is None
    ]
    success_count = len(successful_elapsed)
    total_session_count = replica_count * repetitions
    source_audio_sec = (
        scenario.user_audio_samples.size / 16_000 * success_count
    )
    result = ThroughputBenchmarkResult(
        base_url=base_url,
        data_id=scenario.data_id,
        replica_count=replica_count,
        repetitions=repetitions,
        warmup_success_count=warmup_success_count,
        total_session_count=total_session_count,
        success_count=success_count,
        failure_count=total_session_count - success_count,
        measured_wall_sec=measured_wall_sec,
        sessions_per_sec=(
            success_count / measured_wall_sec if measured_wall_sec else 0.0
        ),
        audio_sec_per_wall_sec=(
            source_audio_sec / measured_wall_sec if measured_wall_sec else 0.0
        ),
        mean_session_elapsed_sec=(
            statistics.fmean(successful_elapsed)
            if successful_elapsed
            else None
        ),
        median_session_elapsed_sec=(
            statistics.median(successful_elapsed)
            if successful_elapsed
            else None
        ),
        p95_session_elapsed_sec=(
            _percentile_95(successful_elapsed)
            if successful_elapsed
            else None
        ),
        measurements=tuple(measurements),
    )
    return result, first_history


def _write_history(
    path: Path,
    history: list[ApiHistoryEntry] | None,
) -> None:
    if history is None:
        return
    path.write_text(
        "".join(
            json.dumps(
                entry.model_dump(mode="json"),
                ensure_ascii=False,
            )
            + "\n"
            for entry in history
        ),
        encoding="utf-8",
    )


def build_argument_parser() -> argparse.ArgumentParser:
    """构造吞吐 benchmark 命令行。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("training_data", type=Path)
    parser.add_argument("--line-index", type=int, default=0)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--replica-count", type=int, required=True)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--insecure", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """加载固定样例、执行测试并写入结果。"""

    args = build_argument_parser().parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
        training_data_path = args.training_data.expanduser().resolve()
        data_root = (
            args.data_root.expanduser().resolve()
            if args.data_root is not None
            else training_data_path.parent
        )
        structure = load_training_data_structure(
            training_data_path,
            line_index=args.line_index,
        )
        training_data = O5DuplexTrainingData.load_structure(
            structure,
            data_root=data_root,
        )
        scenario = build_api_inference_scenario(
            training_data=training_data,
            data_root=data_root,
        )
        result, first_history = asyncio.run(
            benchmark(
                base_url=args.base_url,
                scenario=scenario,
                replica_count=args.replica_count,
                repetitions=args.repetitions,
                insecure=args.insecure,
            )
        )
        (output_dir / "summary.json").write_text(
            result.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )
        _write_history(output_dir / "first_history.jsonl", first_history)
        print(
            "[O5_THROUGHPUT_RESULT] "
            f"success={result.success_count}/{result.total_session_count} "
            f"sessions_per_sec={result.sessions_per_sec:.6f} "
            f"mean_session_sec={result.mean_session_elapsed_sec:.3f}",
            flush=True,
        )
    except Exception as exc:
        error = {"error": f"{type(exc).__name__}: {exc}"}
        (output_dir / "error.json").write_text(
            json.dumps(error, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"[o5-throughput-benchmark] error: {error['error']}",
            file=__import__("sys").stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
