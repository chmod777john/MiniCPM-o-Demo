#!/usr/bin/env python3
"""Rebuild an ODB run report from existing per-sample result.json files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from humanevalkit.metrics import aggregate_metric_records, build_sample_metric_map
from humanevalkit.schemas.metrics import MetricRecord


def _preview(payload: dict[str, Any], result_path: Path, root: Path) -> dict[str, Any]:
    sample = payload.get("sample") or {}
    output = payload.get("output") or {}
    records = [MetricRecord(**raw) for raw in payload.get("metrics") or []]
    error = output.get("error")
    return {
        "sample_id": sample.get("sample_id"),
        "benchmark_family": sample.get("benchmark_family"),
        "benchmark_name": sample.get("benchmark_name"),
        "subbenchmark_name": sample.get("subbenchmark_name"),
        "benchmark_path": sample.get("benchmark_path"),
        "result_path": str(result_path.relative_to(root)),
        "metric_values": build_sample_metric_map(records).get(sample.get("sample_id"), {}),
        "output_preview": {
            "error": error,
            "text": output.get("text"),
            "audio_path": output.get("audio_path"),
            "unit_count": len(output.get("units") or []),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    paths = sorted(root.rglob("result.json"))
    if not paths:
        raise SystemExit(f"no result.json under {root}")

    records: list[MetricRecord] = []
    samples: list[dict[str, Any]] = []
    error_count = 0
    base: dict[str, Any] = {}
    skipped: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not base:
            sample = payload.get("sample") or {}
            runtime = payload.get("runtime_metadata") or {}
            base = {
                "benchmark_family": sample.get("benchmark_family"),
                "benchmark_name": sample.get("benchmark_name"),
                "run_config": runtime.get("engine", {}),
                "sampling": runtime.get("sampling", {}),
            }
        records.extend(MetricRecord(**raw) for raw in payload.get("metrics") or [])
        if (payload.get("output") or {}).get("error"):
            error_count += 1
        samples.append(_preview(payload, path, root))
        skipped.extend(payload.get("skipped_metrics") or [])

    aggregate_records, metrics_summary = aggregate_metric_records(records)
    report = {
        **base,
        "processed_sample_count": len(paths),
        "error_sample_count": error_count,
        "samples": samples,
        "sample_metric_records": [record.model_dump(mode="json") for record in records],
        "aggregate_metric_records": [record.model_dump(mode="json") for record in aggregate_records],
        "metrics_summary": metrics_summary,
        "skipped_metrics": skipped,
        "report_source": "merge_odb_result_files.py",
    }
    report_path = root / "run_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(report_path), "samples": len(paths), "metrics": metrics_summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
