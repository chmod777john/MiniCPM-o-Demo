"""控制面与 Worker job 共用的数据结构：Worker 模板、job 规格、就绪文件。

一个 **Worker 模板** = 「加载哪个 checkpoint、用什么口径」，即控制面板上的 preset；
扩容就是用同一个模板再起一个 Worker。模板名同时作为部署 Profile 的 ``profile_id``，
所以同一模板起的 Worker 对外报告同一个模型。

job 与控制面之间只通过共享存储上的两个文件交接（Pod 不需要能连回开发机）：

- ``<run_dir>/worker_spec.json``：控制面写，job 启动时读；
- ``<run_dir>/worker_ready.json``：job 在 backend 与 worker 都健康后写，控制面读到即登记 Worker。
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")


def _check_name(value: str) -> str:
    if not NAME_RE.match(value):
        raise ValueError("名字只能是 1-48 位小写字母、数字或连字符，且以字母或数字开头")
    return value


class WorkerTemplate(BaseModel):
    """Worker 模板：一个 checkpoint 及其训练口径。"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="模板名；也是部署 Profile 的 profile_id")
    model_dir: str = Field(min_length=1, description="checkpoint 目录（含 model.pt 与 backbone/）")
    tts_pad_protocol: Literal["legacy", "revised"] = Field(
        description="checkpoint 训练侧口径：SDK ≤0.0.6a3 训的是 legacy，≥0.0.6a4 是 revised（不要按目录日期推）",
    )
    non_spoken_budget_while_listening: int = Field(default=45, ge=1)
    non_spoken_budget_while_speaking: int = Field(default=25, ge=1)
    unit_sec: float = Field(default=1.0, gt=0)
    non_spoken_scheduling: Literal["quality", "latency"] = "quality"
    llm_cache: int = Field(default=32768, ge=1)
    gpus: Literal[2] = Field(default=2, description="目前只支持 2×A100 TP2")
    note: str = ""

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        return _check_name(value)

    @field_validator("model_dir")
    @classmethod
    def _absolute(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("model_dir 必须是绝对路径")
        return value


class WorkerJobSpec(BaseModel):
    """一次 Worker job 的完整规格（写进 run_dir/worker_spec.json）。"""

    model_config = ConfigDict(extra="forbid")

    worker_id: str
    endpoint: str
    template: WorkerTemplate
    run_dir: str

    @field_validator("worker_id", "endpoint")
    @classmethod
    def _names(cls, value: str) -> str:
        return _check_name(value)


class WorkerReady(BaseModel):
    """job 写出的就绪信息。"""

    model_config = ConfigDict(extra="ignore")

    worker_id: str
    host: str
    port: int
    model_info: Dict[str, Any] = Field(default_factory=dict)
    ready_at: float
    hostname: Optional[str] = None


SPEC_FILE = "worker_spec.json"
READY_FILE = "worker_ready.json"


def write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(tmp, path)


def read_ready(run_dir: Path) -> Optional[WorkerReady]:
    path = run_dir / READY_FILE
    if not path.is_file():
        return None
    try:
        return WorkerReady.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_ready(run_dir: Path, *, worker_id: str, host: str, port: int,
                model_info: Dict[str, Any]) -> WorkerReady:
    ready = WorkerReady(worker_id=worker_id, host=host, port=port, model_info=model_info,
                        ready_at=time.time(), hostname=os.uname().nodename)
    write_json_atomic(run_dir / READY_FILE, ready.model_dump(mode="json"))
    return ready


if __name__ == "__main__":
    # job 侧：python -m control_plane.specs write-ready <run_dir> <worker_id> <host> <port>
    import sys

    from core.fc_duplex.model_info import worker_model_info

    command, run_dir, worker_id, host, port = sys.argv[1:6]
    if command != "write-ready":
        raise SystemExit(f"unknown command {command}")
    written = write_ready(Path(run_dir), worker_id=worker_id, host=host, port=int(port),
                          model_info=worker_model_info())
    print(f"[worker-ready] {written.worker_id} {written.host}:{written.port} profile={written.model_info.get('profile_id')}")
