#!/usr/bin/env python3
"""Summarize Nsight Systems SQLite exports for decode profiling."""

import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def pct(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int((len(ordered) - 1) * q)
    return ordered[idx]


def summarize_us(values):
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "sum_ms": sum(values) / 1000.0,
        "mean_us": statistics.mean(values),
        "p50_us": statistics.median(values),
        "p90_us": pct(values, 0.90),
        "p99_us": pct(values, 0.99),
        "max_us": max(values),
    }


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: analyze_nsys_sqlite.py path/to/report.sqlite")
    sqlite_path = Path(sys.argv[1])
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()

    result = {
        "sqlite": str(sqlite_path),
        "counts": {},
        "kernel_span_ms": None,
        "kernel_total_ms": None,
        "stream_gaps": [],
        "top_kernels": [],
        "top_cuda_runtime": [],
        "launch_correlation": {},
        "memcpy": [],
        "memset": {},
    }

    for table in (
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_RUNTIME",
        "CUPTI_ACTIVITY_KIND_MEMCPY",
        "CUPTI_ACTIVITY_KIND_MEMSET",
        "NVTX_EVENTS",
    ):
        result["counts"][table] = cur.execute(f"select count(*) from {table}").fetchone()[0]

    kernel_rows = list(cur.execute("select start,end,streamId from CUPTI_ACTIVITY_KIND_KERNEL order by start"))
    if kernel_rows:
        start = min(row[0] for row in kernel_rows)
        end = max(row[1] for row in kernel_rows)
        result["kernel_span_ms"] = (end - start) / 1e6
        result["kernel_total_ms"] = sum(row[1] - row[0] for row in kernel_rows) / 1e6

        by_stream = defaultdict(list)
        for s, e, stream in kernel_rows:
            by_stream[stream].append((s, e))
        for stream, intervals in sorted(by_stream.items(), key=lambda item: -sum(e - s for s, e in item[1])):
            gaps = [
                intervals[i][0] - intervals[i - 1][1]
                for i in range(1, len(intervals))
                if intervals[i][0] > intervals[i - 1][1]
            ]
            result["stream_gaps"].append(
                {
                    "stream": stream,
                    "kernels": len(intervals),
                    "kernel_total_ms": sum(e - s for s, e in intervals) / 1e6,
                    "span_ms": (intervals[-1][1] - intervals[0][0]) / 1e6,
                    "gap_total_ms": sum(gaps) / 1e6,
                    "gap_count": len(gaps),
                    "gap_p50_us": statistics.median(gaps) / 1e3 if gaps else 0.0,
                    "gap_p90_us": pct(gaps, 0.90) / 1e3 if gaps else 0.0,
                    "gap_max_ms": max(gaps) / 1e6 if gaps else 0.0,
                }
            )

    kernel_query = """
        select coalesce(s.value, printf('id:%d', k.demangledName)) name,
               count(*) calls,
               sum(k.end-k.start)/1e6 total_ms,
               avg(k.end-k.start)/1e3 avg_us,
               max(k.end-k.start)/1e3 max_us
        from CUPTI_ACTIVITY_KIND_KERNEL k
        left join StringIds s on s.id = k.demangledName
        group by k.demangledName
        order by sum(k.end-k.start) desc
        limit 30
    """
    for name, calls, total_ms, avg_us, max_us in cur.execute(kernel_query):
        result["top_kernels"].append(
            {"name": name, "calls": calls, "total_ms": total_ms, "avg_us": avg_us, "max_us": max_us}
        )

    runtime_query = """
        select coalesce(s.value, printf('id:%d', r.nameId)) name,
               count(*) calls,
               sum(r.end-r.start)/1e6 total_ms,
               avg(r.end-r.start)/1e3 avg_us,
               max(r.end-r.start)/1e3 max_us
        from CUPTI_ACTIVITY_KIND_RUNTIME r
        left join StringIds s on s.id = r.nameId
        group by r.nameId
        order by sum(r.end-r.start) desc
        limit 30
    """
    for name, calls, total_ms, avg_us, max_us in cur.execute(runtime_query):
        result["top_cuda_runtime"].append(
            {"name": name, "calls": calls, "total_ms": total_ms, "avg_us": avg_us, "max_us": max_us}
        )

    launch_rows = list(
        cur.execute(
            """
            select r.start,r.end,k.start,k.end
            from CUPTI_ACTIVITY_KIND_RUNTIME r
            join CUPTI_ACTIVITY_KIND_KERNEL k on k.correlationId = r.correlationId
            left join StringIds s on s.id = r.nameId
            where s.value like '%Launch%' or s.value like '%Kernel%'
            """
        )
    )
    if launch_rows:
        result["launch_correlation"] = {
            "runtime": summarize_us([(re - rs) / 1e3 for rs, re, _ks, _ke in launch_rows]),
            "api_to_kernel_gap": summarize_us([(ks - re) / 1e3 for _rs, re, ks, _ke in launch_rows]),
            "kernel": summarize_us([(ke - ks) / 1e3 for _rs, _re, ks, ke in launch_rows]),
        }

    for copy_kind, calls, bytes_total, total_ms, avg_us in cur.execute(
        "select copyKind,count(*),sum(bytes),sum(end-start)/1e6,avg(end-start)/1e3 from CUPTI_ACTIVITY_KIND_MEMCPY group by copyKind"
    ):
        result["memcpy"].append(
            {"copy_kind": copy_kind, "calls": calls, "bytes": bytes_total, "total_ms": total_ms, "avg_us": avg_us}
        )

    row = cur.execute("select count(*),sum(end-start)/1e6,avg(end-start)/1e3,sum(bytes) from CUPTI_ACTIVITY_KIND_MEMSET").fetchone()
    if row:
        result["memset"] = {"calls": row[0], "total_ms": row[1], "avg_us": row[2], "bytes": row[3]}

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
