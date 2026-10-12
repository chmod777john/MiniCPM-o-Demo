"""控制面持久状态：端点、Worker 模板、Worker 实例与其生命周期。

整份状态是一个 JSON 文件（原子写）。开发机上的控制面进程重启后从它恢复：在跑的 Worker
是独立的 cctl job，不受影响，控制面按 job 状态与就绪文件把它们重新接管回来。
"""

from __future__ import annotations

import json
import time
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from control_plane.specs import WorkerTemplate, write_json_atomic


class WorkerPhase(str, Enum):
    """Worker 生命周期（控制面视角）。"""

    SUBMITTING = "submitting"   # 正在向 cctl 提交 job
    QUEUED = "queued"           # job 已提交，资源池排队中
    LOADING = "loading"         # job 已运行，模型加载中（还没有就绪文件）
    READY = "ready"             # 已登记进端点，可接会话（是否正在服务看 Gateway 的池状态）
    DRAINING = "draining"       # 驱逐中：不再接新会话，手上的会话结束后停 job
    STOPPING = "stopping"       # 已要求 cctl 停 job，等它结束
    STOPPED = "stopped"         # 已结束（按要求释放）
    FAILED = "failed"           # 非预期结束：提交失败、job 退出、加载超时

    @property
    def active(self) -> bool:
        return self not in (WorkerPhase.STOPPED, WorkerPhase.FAILED)


class PhaseEvent(BaseModel):
    phase: WorkerPhase
    at: float
    note: str = ""


class WorkerRecord(BaseModel):
    worker_id: str
    endpoint: str
    template: WorkerTemplate
    run_dir: str
    phase: WorkerPhase = WorkerPhase.SUBMITTING
    job_id: Optional[int] = None
    job_status: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    created_at: float = Field(default_factory=time.time)
    history: List[PhaseEvent] = Field(default_factory=list)
    error: Optional[str] = None
    release_mode: Optional[Literal["drain", "force"]] = None
    move_to: Optional[str] = None
    """「移动到别的端点」：等它空闲后从原端点摘下、登记到这个端点。"""

    def set_phase(self, phase: WorkerPhase, note: str = "") -> None:
        if self.phase == phase and self.history:
            return
        self.phase = phase
        self.history.append(PhaseEvent(phase=phase, at=time.time(), note=note))

    @property
    def phase_since(self) -> float:
        return self.history[-1].at if self.history else self.created_at


class EndpointRecord(BaseModel):
    name: str
    description: str = ""
    created_at: float = Field(default_factory=time.time)


class ControlState(BaseModel):
    endpoints: Dict[str, EndpointRecord] = Field(default_factory=dict)
    templates: Dict[str, WorkerTemplate] = Field(default_factory=dict)
    workers: Dict[str, WorkerRecord] = Field(default_factory=dict)
    worker_seq: Dict[str, int] = Field(default_factory=dict)
    """每个端点下 Worker 编号的计数器（编号不复用，便于在 cctl / 日志里追溯）。"""


class StateStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.state = ControlState()

    def load(self) -> ControlState:
        if self.path.is_file():
            self.state = ControlState.model_validate(json.loads(self.path.read_text(encoding="utf-8")))
        return self.state

    def save(self) -> None:
        write_json_atomic(self.path, self.state.model_dump(mode="json"))
