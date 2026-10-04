"""在一个 Semantic Realtime API v3 Gateway 后面启动 O5 模型副本池。

用途是**离线推理与评测**：调用方按 Unit 锁步驱动 Session（发 ``input.append`` 后等
``response.unit.committed``），不依赖墙钟，所以副本可以是单卡。拓扑完全由调用方给出：

- ``--deployment-mode``：``single_opt``（单卡，LLM decode CUDA graph，推荐）、
  ``single_eager``（单卡，无任何图优化，只做对照）或 ``tp2``（双卡张量并行）。
- ``--gpu-groups``：分号分隔副本、逗号分隔副本内 GPU；省略时按模式用满本机可见 GPU。
- ``--remote-gateway``：多机时只在本机起副本，并注册到另一台机器上已启动的 Gateway。

服务是**通用**的：进程级 Profile 不绑定 UnitPolicy 与 non-spoken 调度策略，
每个 Session 必须在 ``session.init`` 里显式提供（per-checkpoint 的事实由最外层调用方决定）。

该入口只扩展部署拓扑，不复制 Gateway、Worker、Backend 或模型推理逻辑。
现有 ``run_o5_api.sh`` 及其 Online TP2 默认行为不经过本模块。
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Any, Literal
from urllib.error import URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.fc_duplex.profiles import O5FcDeploymentProfile
from scripts.run_o5_api import (
    DEFAULT_INTERNAL_SETTINGS_PATH,
    O5ApiRuntimeSettings,
    _is_backbone_dir,
)


DeploymentMode = Literal["single_eager", "single_opt", "tp2"]
_GROUP_WIDTH: dict[str, int] = {"single_eager": 1, "single_opt": 1, "tp2": 2}
_BACKEND_READY_TIMEOUT_SEC = 1800.0


class ReplicaPoolSettings(BaseModel):
    """副本池使用者显式提供的部署设置。

    参数:
        pt_path: 完整 O5 PT（``.pt`` state_dict）。
        backbone_dir: TP2 使用的 HF sharded backbone；单卡模式必须为 None。
        deployment_mode: 单卡优化、单卡 eager 或双卡 TP2。
        gpu_groups: 每个模型副本独占的 GPU 分组（``CUDA_VISIBLE_DEVICES`` 编号）。
        runtime_dir: 本次运行的配置、日志、cache 与 ready marker 根目录（必须不存在）。
        llm_cache: 每个 Session 的 LLM StaticCache 位置上限；超出时 Session 直接失败。
        remote_gateway: 远端 Gateway 内部注册地址（``http://host:port``）；None 表示本机起 Gateway。
        advertise_host: 注册给 Gateway 的本机 Worker 地址；本机 Gateway 时为 127.0.0.1。
        gateway_port: 本机 Gateway 公共端口。
        gateway_internal_port: 本机 Gateway Worker registry 端口。
        backend_base_port: 第一个 Backend 端口。
        worker_base_port: 第一个 Worker 端口。
        master_base_port: 第一个 TP2 torchrun rendezvous 端口。
        placement: 本机 Gateway 的新会话放置策略（``spread`` / ``pack``），见 ``config.ServiceSectionConfig.placement``；
            只对本机起的 Gateway 生效。
        recording_dir: API 往返数据录制根目录；None = 不录制。指定后本机 Gateway 把每个
            Session 的全部收发帧（含时间戳）忠实写到 ``<recording_dir>/<session_id>/``。
            只对本机起的 Gateway 生效（``--remote-gateway`` 时流量走远端 Gateway，由它录）。
        check_only: 是否只生成和打印运行计划。
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    pt_path: Path
    backbone_dir: Path | None = None
    deployment_mode: DeploymentMode
    gpu_groups: tuple[tuple[int, ...], ...]
    runtime_dir: Path
    llm_cache: int = Field(ge=1)
    remote_gateway: str | None = None
    advertise_host: str = "127.0.0.1"
    gateway_port: int = Field(default=8009, ge=1, le=65535)
    gateway_internal_port: int = Field(default=8010, ge=1, le=65535)
    backend_base_port: int = Field(default=22510, ge=1, le=65535)
    worker_base_port: int = Field(default=22410, ge=1, le=65535)
    master_base_port: int = Field(default=29500, ge=1, le=65535)
    recording_dir: Path | None = None
    placement: Literal["spread", "pack"] = "spread"
    check_only: bool = False

    @field_validator("gpu_groups")
    @classmethod
    def validate_gpu_groups(
        cls,
        value: tuple[tuple[int, ...], ...],
    ) -> tuple[tuple[int, ...], ...]:
        """拒绝空分组、负数 GPU 和跨副本重复占卡。"""

        if not value:
            raise ValueError("至少需要一个 GPU 分组")
        flattened = [gpu for group in value for gpu in group]
        if any(not group for group in value):
            raise ValueError("GPU 分组不能为空")
        if any(gpu < 0 for gpu in flattened):
            raise ValueError("GPU ID 不能为负数")
        if len(flattened) != len(set(flattened)):
            raise ValueError("同一张 GPU 不能出现在多个副本中")
        return value


class ReplicaPlan(BaseModel):
    """一个模型副本的 GPU、端口和运行目录。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    index: int = Field(ge=0)
    gpu_group: tuple[int, ...]
    backend_port: int = Field(ge=1, le=65535)
    worker_port: int = Field(ge=1, le=65535)
    master_port: int = Field(ge=1, le=65535)
    worker_id: str
    replica_dir: Path


class ReplicaPoolPlan(BaseModel):
    """可审计、可直接启动的完整副本池计划。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: DeploymentMode
    pt_path: Path
    gateway_url: str
    registry_url: str
    start_local_gateway: bool
    gateway_port: int
    gateway_internal_port: int
    worker_host: str
    advertise_host: str
    profile_path: Path
    service_config_path: Path
    backbone_dir: Path | None = None
    llm_cache: int = Field(ge=1)
    ready_path: Path
    replicas: tuple[ReplicaPlan, ...]


class ManagedProcess:
    """一个由 supervisor 独占生命周期的子进程。"""

    def __init__(
        self,
        *,
        name: str,
        command: Sequence[str],
        environment: Mapping[str, str],
        log_path: Path,
        cwd: Path,
    ) -> None:
        """启动子进程并把 stdout/stderr 写入独立日志。

        参数:
            name: 可读进程名。
            command: 不经过 shell 的参数序列。
            environment: 完整子进程环境。
            log_path: 日志路径。
            cwd: 工作目录。
        """

        self.name = name
        self.log_path = log_path
        self._log_file: IO[bytes] = log_path.open("wb")
        self.process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def poll(self) -> int | None:
        """返回子进程退出码；运行中返回 None。"""

        return self.process.poll()

    def stop(self, timeout_sec: float = 20.0) -> None:
        """终止完整 process group，并在超时后强制清理。"""

        if self.process.poll() is None:
            os.killpg(self.process.pid, signal.SIGTERM)
            try:
                self.process.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10.0)
        self._log_file.close()


def parse_gpu_groups(raw: str) -> tuple[tuple[int, ...], ...]:
    """解析 ``0;1`` 或 ``0,1;2,3`` 形式的 GPU 分组。

    参数:
        raw: 分号分隔副本、逗号分隔副本内 GPU 的字符串。

    返回:
        GPU 分组元组。
    """

    groups: list[tuple[int, ...]] = []
    for raw_group in raw.split(";"):
        stripped = raw_group.strip()
        if not stripped:
            raise ValueError("GPU 分组中存在空项")
        groups.append(tuple(int(item.strip()) for item in stripped.split(",")))
    return tuple(groups)


def default_gpu_groups(
    mode: DeploymentMode,
    visible_gpu_count: int,
) -> tuple[tuple[int, ...], ...]:
    """按模式用满全部可见 GPU。

    参数:
        mode: 部署模式。
        visible_gpu_count: 本进程可见 GPU 数。

    返回:
        单卡模式每卡一个副本；TP2 相邻两卡一个副本。
    """

    width = _GROUP_WIDTH[mode]
    if visible_gpu_count < width:
        raise ValueError(f"{mode} 至少需要 {width} 张可见 GPU，实际 {visible_gpu_count}")
    if visible_gpu_count % width:
        raise ValueError(f"{mode} 需要 {width} 的整数倍张 GPU，实际 {visible_gpu_count}")
    return tuple(
        tuple(range(start, start + width))
        for start in range(0, visible_gpu_count, width)
    )


def _visible_gpu_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return len([item for item in visible.split(",") if item.strip()])
    result = subprocess.run(
        ["nvidia-smi", "-L"], capture_output=True, text=True, check=True
    )
    return len([line for line in result.stdout.splitlines() if line.startswith("GPU ")])


def _validate_group_width(
    mode: DeploymentMode,
    groups: tuple[tuple[int, ...], ...],
) -> None:
    expected_width = _GROUP_WIDTH[mode]
    invalid = [group for group in groups if len(group) != expected_width]
    if invalid:
        raise ValueError(
            f"{mode} 每个副本必须使用 {expected_width} 张 GPU，非法分组: {invalid}"
        )


def _validate_model_assets(settings: ReplicaPoolSettings) -> None:
    if not settings.pt_path.is_file():
        raise FileNotFoundError(f"pt_path 不存在或不是文件: {settings.pt_path}")
    if settings.deployment_mode == "tp2":
        if settings.backbone_dir is None:
            raise ValueError("tp2 模式必须提供 --backbone-dir")
        if not _is_backbone_dir(settings.backbone_dir):
            raise FileNotFoundError(
                "backbone 缺少 config.json 或 model.safetensors.index.json: "
                f"{settings.backbone_dir}"
            )
    elif settings.backbone_dir is not None:
        raise ValueError("单卡模式不使用 backbone，禁止传 --backbone-dir")


def _ensure_ports_available(ports: Sequence[int]) -> None:
    duplicates = sorted({port for port in ports if ports.count(port) > 1})
    if duplicates:
        raise ValueError(f"运行计划包含重复端口: {duplicates}")
    occupied: list[int] = []
    for port in ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("0.0.0.0", port))
            except OSError:
                occupied.append(port)
    if occupied:
        raise RuntimeError(f"以下端口已被占用: {occupied}")


def _write_json(path: Path, value: BaseModel | dict[str, object]) -> None:
    data = (
        value.model_dump(mode="json", exclude_none=True)
        if isinstance(value, BaseModel)
        else value
    )
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def prepare_runtime_plan(
    *,
    settings: ReplicaPoolSettings,
    internal: O5ApiRuntimeSettings,
) -> ReplicaPoolPlan:
    """生成 session-bound Profile、服务配置、端口和副本目录。

    参数:
        settings: 调用方显式给出的部署设置。
        internal: Demo 仓库维护的运行环境（Python、基础模型 config/tokenizer 目录）。

    返回:
        完整运行计划。
    """

    _validate_group_width(settings.deployment_mode, settings.gpu_groups)
    _validate_model_assets(settings)
    pt_path = settings.pt_path.expanduser().resolve()
    runtime_dir = settings.runtime_dir.expanduser().resolve()
    runtime_dir.mkdir(parents=True, exist_ok=False)
    for name in ("logs", "data", "cache"):
        (runtime_dir / name).mkdir()

    profile = O5FcDeploymentProfile(
        profile_id=f"o5-replica-pool-{settings.deployment_mode}",
        sdk_version=internal.sdk_version,
        model_path=str(internal.base_model_path),
        pt_path=str(pt_path),
        unit_policy=None,
        non_spoken_scheduling=None,
        deployment_mode=settings.deployment_mode,
        backbone_dir=(
            str(settings.backbone_dir.resolve())
            if settings.backbone_dir is not None
            else None
        ),
        tp_size=_GROUP_WIDTH[settings.deployment_mode],
        llm_cache=settings.llm_cache,
        spmd_heartbeat_interval_sec=internal.spmd_heartbeat_interval_sec,
        attn_implementation=internal.attn_implementation,
    )
    profile_path = runtime_dir / "deployment_profile.json"
    # 保留 null：unit_policy=None 是显式的 session-bound 声明，不能被省略成"缺字段"。
    _write_json(profile_path, profile.model_dump(mode="json"))

    service_config_path = runtime_dir / "service_config.json"
    # 录制默认关；调用方显式给 recording_dir 才开（离线评测按 run 指到评测输出目录下）。
    recording: dict[str, Any] = {"enabled": False}
    if settings.recording_dir is not None:
        recording_dir = settings.recording_dir.expanduser().resolve()
        recording_dir.mkdir(parents=True, exist_ok=True)
        recording = {"enabled": True, "dir": str(recording_dir)}
    _write_json(
        service_config_path,
        {
            "service": {
                "data_dir": str(runtime_dir / "data"),
                "placement": settings.placement,
            },
            "recording": recording,
        },
    )

    host_tag = settings.advertise_host.replace(".", "-")
    replicas: list[ReplicaPlan] = []
    for index, gpu_group in enumerate(settings.gpu_groups):
        replica_dir = runtime_dir / f"replica_{index:02d}"
        replica_dir.mkdir()
        (replica_dir / "cache").mkdir()
        replicas.append(
            ReplicaPlan(
                index=index,
                gpu_group=gpu_group,
                backend_port=settings.backend_base_port + index,
                worker_port=settings.worker_base_port + index,
                master_port=settings.master_base_port + index,
                worker_id=f"o5-{settings.deployment_mode}-{host_tag}-{index:02d}",
                replica_dir=replica_dir,
            )
        )

    start_local_gateway = settings.remote_gateway is None
    registry_url = (
        f"http://127.0.0.1:{settings.gateway_internal_port}"
        if start_local_gateway
        else settings.remote_gateway.rstrip("/")
    )
    plan = ReplicaPoolPlan(
        mode=settings.deployment_mode,
        pt_path=pt_path,
        gateway_url=(
            f"http://127.0.0.1:{settings.gateway_port}" if start_local_gateway else ""
        ),
        registry_url=registry_url,
        start_local_gateway=start_local_gateway,
        gateway_port=settings.gateway_port,
        gateway_internal_port=settings.gateway_internal_port,
        worker_host="127.0.0.1" if start_local_gateway else "0.0.0.0",
        advertise_host=settings.advertise_host,
        profile_path=profile_path,
        service_config_path=service_config_path,
        backbone_dir=settings.backbone_dir,
        llm_cache=settings.llm_cache,
        ready_path=runtime_dir / "READY.json",
        replicas=tuple(replicas),
    )
    _write_json(runtime_dir / "runtime_plan.json", plan)
    return plan


def _http_ok(url: str) -> bool:
    try:
        with urlopen(url, timeout=2.0) as response:
            return 200 <= response.status < 300
    except (URLError, TimeoutError, ConnectionError, OSError):
        return False


def _wait_http_all(
    targets: Sequence[tuple[str, ManagedProcess]],
    *,
    timeout_sec: float,
) -> None:
    """并行等待多个进程的健康检查；任何一个提前退出立即失败。"""

    pending = list(targets)
    deadline = time.monotonic() + timeout_sec
    while pending:
        if time.monotonic() > deadline:
            names = ", ".join(process.name for _, process in pending)
            raise TimeoutError(f"等待健康检查超时: {names}")
        still_pending = []
        for url, process in pending:
            exit_code = process.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"{process.name} 提前退出 code={exit_code}, log={process.log_path}"
                )
            if not _http_ok(url):
                still_pending.append((url, process))
        pending = still_pending
        if pending:
            time.sleep(2.0)


def _register_worker(
    *,
    plan: ReplicaPoolPlan,
    replica: ReplicaPlan,
) -> None:
    payload = json.dumps(
        {
            "endpoint": f"{plan.advertise_host}:{replica.worker_port}",
            "gpu_group": ",".join(str(gpu) for gpu in replica.gpu_group),
            "labels": {
                "model": "o5",
                "runtime": plan.mode,
                "profile": plan.profile_path.stem,
            },
        }
    ).encode("utf-8")
    request = Request(
        f"{plan.registry_url}/internal/workers/{replica.worker_id}",
        data=payload,
        method="PUT",
        headers={"content-type": "application/json"},
    )
    with urlopen(request, timeout=10.0) as response:
        if not 200 <= response.status < 300:
            raise RuntimeError(
                f"注册 Worker 失败: {replica.worker_id}, status={response.status}"
            )


def _base_environment(
    *,
    project_dir: Path,
    internal: O5ApiRuntimeSettings,
    plan: ReplicaPoolPlan,
) -> dict[str, str]:
    environment = dict(os.environ)
    existing_pythonpath = environment.get("PYTHONPATH")
    environment.update(
        {
            "PYTHONPATH": (
                f"{project_dir}:{existing_pythonpath}"
                if existing_pythonpath
                else str(project_dir)
            ),
            "FC_DEPLOYMENT_PROFILE": str(plan.profile_path),
            "O5_DEMO_CONFIG_PATH": str(plan.service_config_path),
            "O5_DEPLOY_MODE": plan.mode,
            "O5_LLM_CACHE": str(plan.llm_cache),
            "O5_LLM_GRAPH": "1",
            "O5_ATTN_IMPLEMENTATION": internal.attn_implementation,
            "TOKENIZERS_PARALLELISM": "false",
            # 锁步驱动的离线 Session 不受墙钟时长限制（Gateway 看门狗关闭）。
            "FC_API_SESSION_MAX_DURATION_S": "0",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    for name in ("no_proxy", "NO_PROXY"):
        entries = [item for item in environment.get(name, "").split(",") if item]
        for host in ("127.0.0.1", "localhost", plan.advertise_host):
            if host not in entries:
                entries.append(host)
        environment[name] = ",".join(entries)
    if plan.backbone_dir is not None:
        environment["O5_BACKBONE_DIR"] = str(plan.backbone_dir)
    else:
        environment.pop("O5_BACKBONE_DIR", None)
    return environment


def serve(
    *,
    project_dir: Path,
    internal: O5ApiRuntimeSettings,
    plan: ReplicaPoolPlan,
) -> None:
    """启动 Gateway（可选）与所有副本，并监督到收到终止信号。

    参数:
        project_dir: Demo 仓库根目录。
        internal: 固定运行设置。
        plan: 已生成的运行计划。
    """

    all_ports = [
        *(
            [plan.gateway_port, plan.gateway_internal_port]
            if plan.start_local_gateway
            else []
        ),
        *[replica.backend_port for replica in plan.replicas],
        *[replica.worker_port for replica in plan.replicas],
        *(
            [replica.master_port for replica in plan.replicas]
            if plan.mode == "tp2"
            else []
        ),
    ]
    _ensure_ports_available(all_ports)

    runtime_python = str(internal.runtime_python)
    base_env = _base_environment(project_dir=project_dir, internal=internal, plan=plan)
    logs_dir = plan.ready_path.parent / "logs"
    processes: list[ManagedProcess] = []
    stop_requested = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        if plan.start_local_gateway:
            gateway = ManagedProcess(
                name="gateway",
                command=[
                    runtime_python,
                    str(project_dir / "gateway.py"),
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(plan.gateway_port),
                    "--internal-port",
                    str(plan.gateway_internal_port),
                    "--http",
                ],
                environment=base_env,
                log_path=logs_dir / "gateway.log",
                cwd=project_dir,
            )
            processes.append(gateway)
            _wait_http_all(
                [(f"{plan.registry_url}/health", gateway)], timeout_sec=120.0
            )
        elif not _http_ok(f"{plan.registry_url}/health"):
            raise RuntimeError(f"远端 Gateway 注册口不可达: {plan.registry_url}")

        backends: list[tuple[ReplicaPlan, dict[str, str], ManagedProcess]] = []
        for replica in plan.replicas:
            replica_env = dict(base_env)
            replica_env.update(
                {
                    "CUDA_VISIBLE_DEVICES": ",".join(
                        str(gpu) for gpu in replica.gpu_group
                    ),
                    "TORCHINDUCTOR_CACHE_DIR": str(replica.replica_dir / "cache"),
                    "O5_TORCHRUN_MASTER_PORT": str(replica.master_port),
                    "MASTER_PORT": str(replica.master_port),
                    "TORCHRUN": str(internal.runtime_python.parent / "torchrun"),
                }
            )
            common_backend_args = [
                "--host",
                "127.0.0.1",
                "--port",
                str(replica.backend_port),
                "--fc-deployment-profile",
                str(plan.profile_path),
                "--gpu-id",
                "0",
            ]
            if plan.mode == "tp2":
                backend_command = [
                    "bash",
                    str(project_dir / "core" / "deploy" / "launch_tp2.sh"),
                    *common_backend_args,
                ]
            else:
                backend_command = [
                    runtime_python,
                    "-m",
                    "py_backend.server",
                    *common_backend_args,
                ]
            backend = ManagedProcess(
                name=f"backend-{replica.index}",
                command=backend_command,
                environment=replica_env,
                log_path=logs_dir / f"backend_{replica.index:02d}.log",
                cwd=project_dir,
            )
            processes.append(backend)
            backends.append((replica, replica_env, backend))
        # 所有副本并行加载权重：串行等待会让 8 副本的启动时间乘 8。
        _wait_http_all(
            [
                (f"http://127.0.0.1:{replica.backend_port}/health", backend)
                for replica, _, backend in backends
            ],
            timeout_sec=_BACKEND_READY_TIMEOUT_SEC,
        )

        workers: list[tuple[ReplicaPlan, ManagedProcess]] = []
        for replica, replica_env, _ in backends:
            worker = ManagedProcess(
                name=f"worker-{replica.index}",
                command=[
                    runtime_python,
                    str(project_dir / "worker.py"),
                    "--host",
                    plan.worker_host,
                    "--port",
                    str(replica.worker_port),
                    "--gpu-id",
                    "0",
                    "--worker-index",
                    str(replica.index),
                    "--backend-server-url",
                    f"http://127.0.0.1:{replica.backend_port}",
                ],
                environment=replica_env,
                log_path=logs_dir / f"worker_{replica.index:02d}.log",
                cwd=project_dir,
            )
            processes.append(worker)
            workers.append((replica, worker))
        _wait_http_all(
            [
                (f"http://127.0.0.1:{replica.worker_port}/health", worker)
                for replica, worker in workers
            ],
            timeout_sec=120.0,
        )
        for replica, _ in workers:
            _register_worker(plan=plan, replica=replica)

        _write_json(
            plan.ready_path,
            {
                "status": "ready",
                "mode": plan.mode,
                "pt_path": str(plan.pt_path),
                "gateway_url": plan.gateway_url,
                "registry_url": plan.registry_url,
                "replica_count": len(plan.replicas),
                "worker_ids": [replica.worker_id for replica in plan.replicas],
            },
        )
        print(
            "[O5_REPLICA_POOL_READY] "
            f"mode={plan.mode} replicas={len(plan.replicas)} "
            f"gateway={plan.gateway_url or plan.registry_url}",
            flush=True,
        )

        while not stop_requested:
            for process in processes:
                exit_code = process.poll()
                if exit_code is not None:
                    raise RuntimeError(
                        f"{process.name} 异常退出 code={exit_code}, "
                        f"log={process.log_path}"
                    )
            time.sleep(2.0)
    finally:
        plan.ready_path.unlink(missing_ok=True)
        for process in reversed(processes):
            process.stop()
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


def build_argument_parser() -> argparse.ArgumentParser:
    """构造副本池命令行。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt-path", type=Path, required=True)
    parser.add_argument("--backbone-dir", type=Path, default=None)
    parser.add_argument(
        "--deployment-mode",
        choices=tuple(_GROUP_WIDTH),
        required=True,
    )
    parser.add_argument(
        "--gpu-groups",
        default=None,
        help="分号分隔副本，逗号分隔副本内 GPU，如 '0;1' 或 '0,1;2,3'；省略=用满可见 GPU",
    )
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument(
        "--llm-cache",
        type=int,
        default=None,
        help="每个 Session 的 LLM 位置上限；默认取内部设置 llm_cache",
    )
    parser.add_argument(
        "--remote-gateway",
        default=None,
        help="多机：远端 Gateway 内部注册地址 http://host:port；省略=本机起 Gateway",
    )
    parser.add_argument(
        "--advertise-host",
        default=None,
        help="注册给远端 Gateway 的本机 IP；仅 --remote-gateway 时需要",
    )
    parser.add_argument("--gateway-port", type=int, default=8009)
    parser.add_argument("--gateway-internal-port", type=int, default=8010)
    parser.add_argument("--backend-base-port", type=int, default=22510)
    parser.add_argument("--worker-base-port", type=int, default=22410)
    parser.add_argument("--master-base-port", type=int, default=29500)
    parser.add_argument(
        "--recording-dir",
        type=Path,
        default=None,
        help="开启 API 往返数据录制并写到 <dir>/<session_id>/；省略=不录制",
    )
    parser.add_argument(
        "--placement",
        choices=("spread", "pack"),
        default="spread",
        help="本机 Gateway 的新会话放置策略；pack=本机优先、其次预计最晚跑完的扩容主机（弹性缩容用）",
    )
    parser.add_argument(
        "--internal-settings",
        type=Path,
        default=DEFAULT_INTERNAL_SETTINGS_PATH,
    )
    parser.add_argument("--check-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """解析设置、生成计划，并按需启动服务。"""

    args = build_argument_parser().parse_args(argv)
    try:
        internal = O5ApiRuntimeSettings.model_validate_json(
            args.internal_settings.read_text(encoding="utf-8")
        )
        gpu_groups = (
            parse_gpu_groups(args.gpu_groups)
            if args.gpu_groups
            else default_gpu_groups(args.deployment_mode, _visible_gpu_count())
        )
        if args.remote_gateway is not None and not args.advertise_host:
            raise ValueError("--remote-gateway 需要同时给出本机 --advertise-host")
        settings = ReplicaPoolSettings(
            pt_path=args.pt_path,
            backbone_dir=args.backbone_dir,
            deployment_mode=args.deployment_mode,
            gpu_groups=gpu_groups,
            runtime_dir=args.runtime_dir,
            llm_cache=args.llm_cache or internal.llm_cache,
            remote_gateway=args.remote_gateway,
            advertise_host=args.advertise_host or "127.0.0.1",
            gateway_port=args.gateway_port,
            gateway_internal_port=args.gateway_internal_port,
            backend_base_port=args.backend_base_port,
            worker_base_port=args.worker_base_port,
            master_base_port=args.master_base_port,
            recording_dir=args.recording_dir,
            placement=args.placement,
            check_only=args.check_only,
        )
        plan = prepare_runtime_plan(settings=settings, internal=internal)
        if settings.check_only:
            print(json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, indent=2))
            return 0
        serve(
            project_dir=Path(__file__).resolve().parents[1],
            internal=internal,
            plan=plan,
        )
    except Exception as exc:
        print(
            f"[o5-replica-pool] error: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
