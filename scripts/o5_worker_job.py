"""一个 Worker = 一个 cctl job：按控制面写好的 worker_spec.json 起 backend + worker，就绪后写 worker_ready.json。

用法（在 deploy 池的 job 里，由控制面生成的 entry 调用）::

    <runtime python> scripts/o5_worker_job.py --run-dir <run_dir> [--dry-run]

步骤：
1. 读 ``<run_dir>/worker_spec.json``（Worker 模板 + worker_id）。
2. 复用 ``scripts/run_o5_api.py`` 生成部署 Profile 与运行目录，再按模板写入
   profile_id，以及 Session 没传时用的推荐默认值（预算、调度方式、tts_pad）。
3. 以 worker-only 模式交给 ``scripts/start_fc_service.py``：不起 Gateway，worker 绑 0.0.0.0，
   就绪后由 ``start_o5_tp2_cctl_service.sh`` 写 ``worker_ready.json``；任一子进程退出则 job 退出。

``--dry-run`` 只做 1、2 并打印将要使用的环境，不起进程（本机自检用）。
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

from control_plane.specs import SPEC_FILE, WorkerJobSpec  # noqa: E402
from core.fc_duplex.profiles import O5FcDeploymentProfile, load_fc_deployment_profile  # noqa: E402
from scripts.run_o5_api import (  # noqa: E402
    O5ApiLaunchSettings,
    build_service_environment,
    discover_o5_model_assets,
    load_o5_api_runtime_settings,
    prepare_o5_api_runtime,
    validate_o5_api_runtime,
)


def pod_ip() -> str:
    """本 Pod 在集群内可被开发机直连的地址。"""

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.connect(("10.255.255.255", 1))          # 不发包，只为选出出口网卡地址
        return sock.getsockname()[0]


def prepare(spec: WorkerJobSpec) -> tuple[dict[str, str], Path, Path]:
    """生成 Profile 与运行目录，返回 (环境, profile 路径, runtime python)。"""

    template = spec.template
    run_dir = Path(spec.run_dir)
    internal = load_o5_api_runtime_settings()
    prepared = prepare_o5_api_runtime(
        launch=O5ApiLaunchSettings(model_path=Path(template.model_dir), storage_dir=run_dir, check_only=True),
        internal=internal,
        assets=discover_o5_model_assets(Path(template.model_dir)),
    )
    data = load_fc_deployment_profile(prepared.profile_path).model_dump(mode="json")
    policy = data["unit_policy"]
    policy["non_spoken_budgets_while_listening"] = [template.non_spoken_budget_while_listening]
    policy["non_spoken_budgets_while_speaking"] = [template.non_spoken_budget_while_speaking]
    data.update({
        "profile_id": template.name,
        "unit_policy": policy,
        "non_spoken_scheduling": template.non_spoken_scheduling,
        "tts_pad_protocol": template.tts_pad_protocol,
        "llm_cache": template.llm_cache,
    })
    profile = O5FcDeploymentProfile.model_validate(data)
    prepared.profile_path.write_text(profile.model_dump_json(indent=2) + "\n", encoding="utf-8")
    validate_o5_api_runtime(internal=internal, prepared=prepared)

    env = build_service_environment(project_dir=PROJECT_DIR, internal=internal, prepared=prepared)
    env.update({
        "WORKER_ONLY": "1",
        "WORKER_ID": spec.worker_id,
        "WORKER_HOST": "0.0.0.0",
        "WORKER_ADVERTISE_HOST": pod_ip(),
        "WORKER_READY_DIR": str(run_dir),
        "FC_AUDIO_CACHE_DIR": "/tmp/o5_fc_audio_cache",
    })
    return env, prepared.profile_path, internal.runtime_python


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    spec = WorkerJobSpec.model_validate_json((args.run_dir / SPEC_FILE).read_text(encoding="utf-8"))
    env, profile_path, runtime_python = prepare(spec)
    shown = {k: env[k] for k in sorted(env) if k.startswith(("WORKER_", "FC_", "O5_", "GATEWAY_", "BACKEND_"))}
    print(f"[worker-job] {spec.worker_id} endpoint={spec.endpoint} profile={profile_path}")
    print(json.dumps(shown, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    entry = PROJECT_DIR / "scripts" / "start_fc_service.py"
    os.execvpe(str(runtime_python), [str(runtime_python), str(entry), "--profile", str(profile_path)], env)


if __name__ == "__main__":
    main()
