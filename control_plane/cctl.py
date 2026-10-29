"""cctl 的最小封装：在 deploy 资源池提交 / 查询 / 停止 Worker job。

只用三个子命令：``job create``、``job get``、``job stop --yes``。测试一律注入假实现
（见 ``tests/control_plane``），不碰真集群。
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shlex
from dataclasses import dataclass, field
from typing import Optional, Protocol

logger = logging.getLogger("control_plane.cctl")

TERMINAL_STATUSES = frozenset({"Succeeded", "Failed", "Killed", "Stopped", "Error", "Cancelled"})


@dataclass(frozen=True)
class JobSettings:
    """deploy 池的 job 规格（2×A100 TP2；CPU / 内存按用户 2026-10-12 定：16 核、256 GiB）。"""

    project: str = "swy-o5-deploy"
    cluster: str = "langfang_train"
    resource_pool: str = "deploy"
    billing_account_id: str = "N00002"
    image: str = "model-align/ngc-train:hjsun_0313_omni_transformers_52_0603"
    gpu: int = 2
    gpu_model: str = "A100"
    cpu: int = 16
    memory_gi: int = 256
    priority: str = "NORMAL"


_ENV_PREFIX = "CCTL_JOB_"
_INT_FIELDS = frozenset({"gpu", "cpu", "memory_gi"})


def job_settings_from_env() -> JobSettings:
    """按环境变量覆盖 job 规格(不设任何变量时与 ``JobSettings()`` 完全一致)。

    独立部署(如 human_eval_1029_rejection)用同一份代码起第二套网关,只通过 env
    切换资源池 / 项目等;默认路径不读任何 env,行为不变。变量名 = ``CCTL_JOB_<字段名>``,
    整数字段(gpu / cpu / memory_gi)按 int 解析,解析失败抛 CctlError。
    """

    overrides: dict = {}
    for field_name in JobSettings.__dataclass_fields__:
        raw = os.environ.get(_ENV_PREFIX + field_name.upper())
        if raw is None or raw == "":
            continue
        if field_name in _INT_FIELDS:
            try:
                overrides[field_name] = int(raw)
            except ValueError:
                raise CctlError(f"{_ENV_PREFIX}{field_name.upper()} 不是整数: {raw!r}")
        else:
            overrides[field_name] = raw
    return JobSettings(**overrides)


class CctlError(RuntimeError):
    pass


class JobRunner(Protocol):
    async def create(self, *, description: str, entry: str) -> int: ...
    async def status(self, job_id: int) -> str: ...
    async def stop(self, job_id: int) -> None: ...
    async def find_by_description(self, description: str) -> Optional[int]: ...


CYBERTRON_JOB_URL = "https://cybertron.modelbest.co/job-detail?id={job_id}&cluster={cluster}"


def job_url(job_id: int, cluster: str) -> str:
    """该 job 在 Cybertron 网页上的详情页。"""

    return CYBERTRON_JOB_URL.format(job_id=job_id, cluster=cluster)


def description_for(worker_id: str) -> str:
    """job 描述里带唯一标记，提交报错时据此查有没有其实已经建出来（这种情况实际遇到过）。"""

    return f"o5-worker {worker_id} (control-plane)"


@dataclass
class CctlRunner:
    settings: JobSettings = field(default_factory=JobSettings)
    binary: str = os.environ.get("CCTL_BIN") or os.path.expanduser("~/.local/bin/cctl")
    timeout_s: float = 90.0

    def _env(self) -> dict:
        env = dict(os.environ)
        env.setdefault("CCTL_PROFILE", "modelbest")
        env["CCTL_NO_UPDATE_CHECK"] = "1"
        return env

    async def _run(self, *args: str) -> dict:
        proc = await asyncio.create_subprocess_exec(
            self.binary, *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=self._env(),
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=self.timeout_s)
        except asyncio.TimeoutError as exc:
            proc.kill()
            raise CctlError(f"cctl {args[0]} {args[1]} 超时") from exc
        text = out.decode("utf-8", "replace")
        match = re.search(r"\{.*\}", text, re.S)
        if match is None:
            raise CctlError(f"cctl {' '.join(args[:2])} 无 JSON 输出: {err.decode('utf-8', 'replace')[-300:]}")
        data = json.loads(match.group(0))
        if isinstance(data, dict) and data.get("code") and not data.get("name") and not data.get("id"):
            raise CctlError(f"cctl {' '.join(args[:2])} 失败: {data.get('code')} {data.get('message')}")
        return data

    async def create(self, *, description: str, entry: str) -> int:
        s = self.settings
        data = await self._run(
            "job", "create", "--project", s.project, "--cluster", s.cluster, "--resource-pool", s.resource_pool,
            "--billing-account-id", s.billing_account_id, "--image", s.image, "--gpu", str(s.gpu),
            "--gpu-model", s.gpu_model, "--cpu", str(s.cpu), "--memory", str(s.memory_gi),
            "--priority", s.priority, "--description", description, "--entry", entry, "--no-input",
        )
        name = str(data.get("name") or data.get("id") or "")
        match = re.search(r"(\d+)$", name)
        if match is None:
            raise CctlError(f"cctl job create 返回里没有 job id: {data}")
        return int(match.group(1))

    async def status(self, job_id: int) -> str:
        data = await self._run("job", "get", str(job_id), "-o", "json")
        return str(data.get("status") or data.get("state") or "Unknown")

    async def stop(self, job_id: int) -> None:
        await self._run("job", "stop", str(job_id), "--yes")

    async def find_by_description(self, description: str) -> Optional[int]:
        data = await self._run("job", "list", "--project", self.settings.project)
        for task in data.get("tasks") or []:
            if task.get("description") == description:
                match = re.search(r"(\d+)$", str(task.get("name") or ""))
                if match:
                    return int(match.group(1))
        return None


def worker_entry(*, repo_dir: str, python: str, run_dir: str) -> str:
    """Worker job 的 entry：在仓库目录里跑 o5_worker_job.py，日志落到 run_dir/job.log。"""

    inner = (f"cd {shlex.quote(repo_dir)} && {shlex.quote(python)} scripts/o5_worker_job.py "
             f"--run-dir {shlex.quote(run_dir)} > {shlex.quote(run_dir)}/job.log 2>&1")
    return f"bash -lc {shlex.quote(inner)}"
