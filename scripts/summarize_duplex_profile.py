#!/usr/bin/env python3
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _mean(values):
    return statistics.mean(values) if values else 0.0


def _p50(values):
    return statistics.median(values) if values else 0.0


def _p90(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(0.9 * (len(ordered) - 1))))
    return ordered[idx]


def main():
    parser = argparse.ArgumentParser(description="Summarize duplex profile_events from timings.jsonl")
    parser.add_argument("timings", type=Path, help="Path to timings.jsonl")
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.timings.read_text().splitlines() if line.strip()]
    by_event = defaultdict(list)
    by_event_listen = defaultdict(list)
    by_event_speak = defaultdict(list)

    for row in rows:
        is_listen = bool(row.get("is_listen"))
        events = []
        prefill = row.get("prefill")
        if isinstance(prefill, dict):
            events.extend(prefill.get("profile_events") or [])
        events.extend(row.get("profile_events") or [])
        for event in events:
            name = event.get("name")
            dur = event.get("dur_s")
            if name is None or dur is None:
                continue
            by_event[name].append(float(dur))
            (by_event_listen if is_listen else by_event_speak)[name].append(float(dur))

    print(f"rows {len(rows)}")
    print(f"listen {sum(1 for r in rows if r.get('is_listen'))}")
    print(f"speak {sum(1 for r in rows if not r.get('is_listen'))}")
    print()
    print("event,count,mean_ms,p50_ms,p90_ms,max_ms,listen_mean_ms,speak_mean_ms")
    for name in sorted(by_event):
        vals = by_event[name]
        listen_vals = by_event_listen.get(name, [])
        speak_vals = by_event_speak.get(name, [])
        print(
            f"{name},{len(vals)},"
            f"{_mean(vals) * 1000:.3f},{_p50(vals) * 1000:.3f},"
            f"{_p90(vals) * 1000:.3f},{max(vals) * 1000:.3f},"
            f"{_mean(listen_vals) * 1000:.3f},{_mean(speak_vals) * 1000:.3f}"
        )


if __name__ == "__main__":
    main()
