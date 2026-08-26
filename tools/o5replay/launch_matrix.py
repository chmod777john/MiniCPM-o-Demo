#!/usr/bin/env python3
"""Submit record -> parallel replay jobs to cctl using one immutable trace bundle."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TERMINAL_SUCCESS = {"Succeeded"}
TERMINAL_FAILURE = {"Failed", "OOMKilled", "ImagePullBackOff", "Invalid", "Killed", "Stopped"}


@dataclass(frozen=True)
class JobSpec:
    name: str
    target: str
    gpu: int
    output: Path


def _run(command: list[str], *, capture: bool = True) -> str:
    print("+ " + shlex.join(command), flush=True)
    completed = subprocess.run(command, check=True, text=True, capture_output=capture)
    if completed.stderr:
        print(completed.stderr.rstrip(), flush=True)
    return completed.stdout.strip() if completed.stdout else ""


def _pool_availability() -> dict[str, int]:
    rows = json.loads(_run(["cctl", "pool", "list", "--own", "-o", "json"]))
    available = {}
    for row in rows:
        name = row.get("name")
        quotas = (row.get("edges") or {}).get("resource_quotas") or []
        a100 = next((item for item in quotas if str(item.get("name")).lower() == "a100"), None)
        if name and a100:
            available[str(name)] = int(a100.get("capacity", 0)) - int(a100.get("used", 0))
    return available


def choose_pool(requested_gpus: int, explicit: str | None) -> str:
    if explicit:
        return explicit
    available = _pool_availability()
    for pool in ("agent-dev", "agent-train"):
        if available.get(pool, 0) >= requested_gpus:
            print(f"selected pool={pool} available_a100={available[pool]}", flush=True)
            return pool
    raise RuntimeError(
        f"neither agent-dev nor agent-train has quota for {requested_gpus} A100s: {available}"
    )


def _runner_command(
    args: argparse.Namespace,
    spec: JobSpec,
    *,
    source: Path,
    reference: Path | None,
) -> str:
    executable = Path(args.venv) / ("bin/torchrun" if spec.target == "demo-tp2" else "bin/python")
    command = [str(executable)]
    if spec.target == "demo-tp2":
        command.extend(["--standalone", "--nproc_per_node=2"])
    command.extend([
        str(Path(args.repo) / "tools/o5replay/run_session.py"),
        "--session-dir", str(source),
        "--out-dir", str(spec.output),
        "--target", spec.target,
        "--capture-mode", "replay",
        "--max-units", str(args.max_units),
        "--overwrite",
    ])
    if reference is not None:
        command.extend(["--reference-session", str(reference), "--forcing", args.forcing])
    if args.capture_layers:
        command.append("--capture-layers")
    command.extend(args.runner_arg)
    return "cd " + shlex.quote(str(Path(args.repo).resolve())) + " && " + shlex.join(command)


def submit(args: argparse.Namespace, spec: JobSpec, command: str, pool: str) -> str:
    cpu = args.tp2_cpu if spec.gpu == 2 else args.cpu
    memory = args.tp2_memory if spec.gpu == 2 else args.memory
    create = [
        "cctl", "job", "create", "--no-input", "-q",
        "--project", args.project,
        "--billing-account-id", args.billing_account_id,
        "--cluster", args.cluster,
        "--resource-pool", pool,
        "--image", args.image,
        "--gpu", str(spec.gpu),
        "--gpu-model", "a100",
        "--cpu", str(cpu),
        "--memory", str(memory),
        "--description", f"o5-session-replay:{spec.name}",
        "--entry", command,
    ]
    if args.dry_run:
        create.append("--dry-run")
        _run(create, capture=False)
        return "dry-run"
    output = _run(create)
    job_id = output.splitlines()[-1].strip().removeprefix("tasks/")
    print(f"submitted {spec.name}: tasks/{job_id}", flush=True)
    return job_id


def status(job_id: str) -> str:
    payload = json.loads(_run(["cctl", "job", "get", job_id, "-o", "json"]))
    task = payload.get("task", payload)
    return str(task.get("status") or "Unknown")


def wait_for(job_id: str, poll_seconds: int) -> str:
    previous = None
    while True:
        current = status(job_id)
        if current != previous:
            print(f"tasks/{job_id}: {current}", flush=True)
            previous = current
        if current in TERMINAL_SUCCESS | TERMINAL_FAILURE:
            return current
        time.sleep(poll_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True, help="API or previously materialized input bundle")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--record-target", choices=("existing", "canonical", "demo-single", "demo-tp2"), default="canonical")
    parser.add_argument("--replay-target", action="append", choices=("canonical", "demo-single", "demo-tp2"))
    parser.add_argument("--forcing", default="all")
    parser.add_argument("--max-units", type=int, default=8)
    parser.add_argument("--capture-layers", action="store_true")
    parser.add_argument("--runner-arg", action="append", default=[], help="one additional run_session.py argument")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--venv", default="/user/weihongliang/MiniCPM-o-Demo-wt-o5-inference-refactor-2026-06-30/.venv-high-cu128")
    parser.add_argument("--project", default="o5")
    parser.add_argument("--billing-account-id", default="N00002")
    parser.add_argument("--cluster", default="langfang_train")
    parser.add_argument("--image", default="cybertron/minicpmo:202506140642-bb9a49")
    parser.add_argument("--resource-pool", choices=("agent-dev", "agent-train"))
    parser.add_argument("--cpu", type=int, default=8)
    parser.add_argument("--memory", type=int, default=128)
    parser.add_argument("--tp2-cpu", type=int, default=16)
    parser.add_argument("--tp2-memory", type=int, default=220)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-wait", action="store_true")
    args = parser.parse_args()

    source = Path(args.session_dir).resolve()
    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    targets = args.replay_target or ["canonical", "demo-single", "demo-tp2"]
    record_output = source
    record_job = None

    if args.record_target != "existing":
        gpu = 2 if args.record_target == "demo-tp2" else 1
        record_output = out_root / f"record-{args.record_target}"
        spec = JobSpec("record", args.record_target, gpu, record_output)
        pool = choose_pool(gpu, args.resource_pool)
        record_job = submit(
            args,
            spec,
            _runner_command(args, spec, source=source, reference=None),
            pool,
        )
        if args.no_wait:
            print(json.dumps({"record_job": record_job, "reference": str(record_output)}), flush=True)
            return 0
        if not args.dry_run:
            state = wait_for(record_job, args.poll_seconds)
            if state not in TERMINAL_SUCCESS:
                raise RuntimeError(f"record job tasks/{record_job} ended as {state}; replay jobs were not submitted")

    specs = [
        JobSpec(f"replay-{target}", target, 2 if target == "demo-tp2" else 1, out_root / f"replay-{target}")
        for target in targets
    ]
    required = sum(spec.gpu for spec in specs)
    pool = choose_pool(required, args.resource_pool)
    jobs: dict[str, str] = {}
    for spec in specs:
        command = _runner_command(args, spec, source=record_output, reference=record_output)
        compare = [
            str(Path(args.venv) / "bin/python"),
            str(Path(args.repo) / "tools/o5replay/compare.py"),
            str(record_output),
            str(spec.output),
            "--out", str(spec.output / "comparison.json"),
        ]
        command += " && " + shlex.join(compare)
        jobs[spec.name] = submit(args, spec, command, pool)

    manifest = {
        "record_job": record_job,
        "reference": str(record_output),
        "replay_jobs": jobs,
        "pool": pool,
        "submitted_at": time.time(),
    }
    (out_root / "matrix_jobs.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    if not args.dry_run and not args.no_wait:
        final = {name: wait_for(job_id, args.poll_seconds) for name, job_id in jobs.items()}
        print(json.dumps({"final": final}, indent=2), flush=True)
        return 0 if all(value in TERMINAL_SUCCESS for value in final.values()) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
