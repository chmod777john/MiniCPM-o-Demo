"""控制面生命周期：提交 / 排队 / 加载 / 就绪登记 / 扩容 / 驱逐 / 强制关闭 / 移动 / 故障 / 重启接管。

cctl 一律用假实现（FakeJobs），不碰真集群；Worker 地址指向本机无人监听的端口，
Gateway 健康检查会判它离线，但登记、移出、派发这些池操作照常可测。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, List, Optional

import pytest

from control_plane.cctl import CctlError, description_for
from control_plane.manager import ControlPlane, ControlPlaneError
from control_plane.specs import SPEC_FILE, WorkerJobSpec, WorkerTemplate, write_ready
from control_plane.state import WorkerPhase
from gateway_modules.endpoints import EndpointRegistry
from gateway_modules.live_sessions import LiveSession, LiveSessionRegistry
from gateway_modules.models import GatewayWorkerStatus
from gateway_modules.worker_pool import WorkerPool

TEMPLATE = WorkerTemplate(name="f9t4-rejection", model_dir="/abs/ckpt", tts_pad_protocol="legacy")


class FakeJobs:
    def __init__(self) -> None:
        self.next_id = 1000
        self.statuses: Dict[int, str] = {}
        self.descriptions: Dict[int, str] = {}
        self.stopped: List[int] = []
        self.fail_create: Optional[str] = None      # None / "lost" (报错但其实建出来了) / "error"
        self.create_gate: Optional[asyncio.Event] = None
        self.status_gate: Optional[asyncio.Event] = None   # 模拟 Cybertron API 很慢
        self.status_error: Optional[str] = None

    async def create(self, *, description: str, entry: str) -> int:
        if self.create_gate is not None:
            await self.create_gate.wait()
        self.next_id += 1
        if self.fail_create == "error":
            raise CctlError("internal_error")
        self.statuses[self.next_id] = "Queued"
        self.descriptions[self.next_id] = description
        if self.fail_create == "lost":
            raise CctlError("internal_error: read response: unexpected EOF")
        return self.next_id

    async def status(self, job_id: int) -> str:
        if self.status_gate is not None:
            await self.status_gate.wait()
        if self.status_error is not None:
            raise CctlError(self.status_error)
        return self.statuses[job_id]

    async def stop(self, job_id: int) -> None:
        self.stopped.append(job_id)
        self.statuses[job_id] = "Killed"

    async def find_by_description(self, description: str) -> Optional[int]:
        return next((i for i, d in self.descriptions.items() if d == description), None)


def _pool() -> WorkerPool:
    return WorkerPool(worker_addresses=[])


async def _plane(root: Path, jobs: FakeJobs, registry: Optional[EndpointRegistry] = None,
                 live: Optional[LiveSessionRegistry] = None, **kw) -> ControlPlane:
    registry = registry or EndpointRegistry(_pool)
    if registry.get("default") is None:
        await registry.start()
    plane = ControlPlane(root=root, registry=registry, live_sessions=live or LiveSessionRegistry(), jobs=jobs,
                         repo_dir="/repo", python="/venv/bin/python", job_poll_s=0.0, **kw)
    await plane.start(run_loop=False)
    return plane


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


async def _ready_worker(plane: ControlPlane, jobs: FakeJobs, endpoint: str = "alpha") -> str:
    if endpoint not in plane.state.endpoints:
        await plane.create_endpoint(endpoint)
    if TEMPLATE.name not in plane.state.templates:
        await plane.put_template(TEMPLATE)
    record = await plane.add_worker(endpoint, TEMPLATE.name)
    await _settle()
    jobs.statuses[record.job_id] = "Running"
    await plane.tick()
    write_ready(Path(record.run_dir), worker_id=record.worker_id, host="127.0.0.1", port=1,
                model_info={"profile_id": TEMPLATE.name})
    await plane.tick()
    assert plane.state.workers[record.worker_id].phase == WorkerPhase.READY
    return record.worker_id


@pytest.mark.asyncio
async def test_lifecycle_submit_queue_load_ready(tmp_path: Path) -> None:
    jobs = FakeJobs()
    plane = await _plane(tmp_path, jobs)
    await plane.create_endpoint("alpha")
    await plane.put_template(TEMPLATE)
    record = await plane.add_worker("alpha", TEMPLATE.name)
    assert record.worker_id == "alpha-1" and record.phase == WorkerPhase.SUBMITTING
    await _settle()
    assert record.phase == WorkerPhase.QUEUED and jobs.descriptions[record.job_id] == description_for("alpha-1")
    spec = WorkerJobSpec.model_validate_json((Path(record.run_dir) / SPEC_FILE).read_text())
    assert spec.template == TEMPLATE and spec.endpoint == "alpha"

    await plane.tick()
    assert record.phase == WorkerPhase.QUEUED                 # cctl 还在排队
    jobs.statuses[record.job_id] = "Running"
    await plane.tick()
    assert record.phase == WorkerPhase.LOADING
    write_ready(Path(record.run_dir), worker_id="alpha-1", host="127.0.0.1", port=1, model_info={"profile_id": "p"})
    await plane.tick()
    assert record.phase == WorkerPhase.READY
    pool = plane.registry.get("alpha")
    assert pool.get_worker("alpha-1").model_info == {"profile_id": "p"}
    assert [h.phase for h in record.history] == [WorkerPhase.SUBMITTING, WorkerPhase.QUEUED,
                                                 WorkerPhase.LOADING, WorkerPhase.READY]


@pytest.mark.asyncio
async def test_scale_up_clones_template_into_same_endpoint(tmp_path: Path) -> None:
    jobs = FakeJobs()
    plane = await _plane(tmp_path, jobs)
    first = await _ready_worker(plane, jobs)
    clone = await plane.scale_up(first)
    await _settle()
    assert clone.worker_id == "alpha-2" and clone.endpoint == "alpha" and clone.template == TEMPLATE
    assert clone.phase == WorkerPhase.QUEUED and clone.job_id != plane.state.workers[first].job_id


@pytest.mark.asyncio
async def test_scale_up_uses_current_template_not_worker_snapshot(tmp_path: Path) -> None:
    """模板改过（例如修正了 checkpoint 目录）后，在失败的 Worker 上重新起要用新内容，不能原样重试旧快照。"""

    jobs = FakeJobs()
    plane = await _plane(tmp_path, jobs)
    await plane.create_endpoint("alpha")
    await plane.put_template(TEMPLATE)
    first = await plane.add_worker("alpha", TEMPLATE.name)
    await _settle()
    jobs.statuses[first.job_id] = "Failed"
    await plane.tick()
    assert first.phase == WorkerPhase.FAILED

    fixed = TEMPLATE.model_copy(update={"model_dir": "/abs/ckpt-fixed"})
    await plane.put_template(fixed)
    view = plane._worker_view(first, 0.0)
    assert view["template_changes"] == [{"field": "model_dir", "worker": "/abs/ckpt", "current": "/abs/ckpt-fixed"}]

    retry = await plane.scale_up(first.worker_id)
    await _settle()
    assert retry.template == fixed and first.template == TEMPLATE        # 旧 Worker 的快照保持原样
    spec = WorkerJobSpec.model_validate_json((Path(retry.run_dir) / SPEC_FILE).read_text())
    assert spec.template.model_dir == "/abs/ckpt-fixed"
    assert plane._worker_view(retry, 0.0)["template_changes"] == []


@pytest.mark.asyncio
async def test_template_changes_ignores_note_and_reports_deleted_template(tmp_path: Path) -> None:
    jobs = FakeJobs()
    plane = await _plane(tmp_path, jobs)
    worker_id = await _ready_worker(plane, jobs)
    record = plane.state.workers[worker_id]
    await plane.put_template(TEMPLATE.model_copy(update={"note": "只改备注"}))
    assert plane._worker_view(record, 0.0)["template_changes"] == []     # 备注不影响运行，不算旧版
    await plane.delete_template(TEMPLATE.name)
    assert plane._worker_view(record, 0.0)["template_changes"] is None
    with pytest.raises(ControlPlaneError, match="已删除"):
        await plane.scale_up(worker_id)


@pytest.mark.asyncio
async def test_drain_waits_for_the_live_session(tmp_path: Path) -> None:
    jobs = FakeJobs()
    plane = await _plane(tmp_path, jobs)
    wid = await _ready_worker(plane, jobs)
    pool = plane.registry.get("alpha")
    worker = pool.get_worker(wid)
    worker.status = GatewayWorkerStatus.IDLE
    _, future = pool.enqueue("audio_duplex")                 # 一个会话正占着它
    assert future.result() is worker

    record = await plane.drain(wid)
    assert record.phase == WorkerPhase.DRAINING and worker.draining
    await plane.tick()
    assert record.phase == WorkerPhase.DRAINING and not jobs.stopped    # 会话没结束，不停 job
    _, waiting = pool.enqueue("audio_duplex")
    assert not waiting.done()                                          # 不再派新会话

    pool.release_worker(worker, request_type="audio_duplex", duration_s=5.0)
    await plane.tick()
    assert record.phase == WorkerPhase.STOPPING and jobs.stopped == [record.job_id]
    assert pool.get_worker(wid) is None
    await plane.tick()
    assert record.phase == WorkerPhase.STOPPED


@pytest.mark.asyncio
async def test_force_close_interrupts_session_and_stops_job(tmp_path: Path) -> None:
    jobs = FakeJobs()
    live = LiveSessionRegistry()
    plane = await _plane(tmp_path, jobs, live=live)
    wid = await _ready_worker(plane, jobs)
    session = LiveSession(worker_id=wid, ticket_id="q_1", endpoint="alpha")
    live.register(session)
    record = await plane.force_close(wid)
    assert session.close_event.is_set() and session.close_reason == "worker_removed"
    assert plane.registry.get("alpha").get_worker(wid) is None
    assert record.phase == WorkerPhase.STOPPING and jobs.stopped == [record.job_id]
    await plane.tick()
    assert record.phase == WorkerPhase.STOPPED and record.release_mode == "force"


@pytest.mark.asyncio
async def test_job_dying_while_ready_marks_failed_and_unregisters(tmp_path: Path) -> None:
    jobs = FakeJobs()
    plane = await _plane(tmp_path, jobs)
    wid = await _ready_worker(plane, jobs)
    record = plane.state.workers[wid]
    jobs.statuses[record.job_id] = "Failed"
    await plane.tick()
    assert record.phase == WorkerPhase.FAILED and "非预期结束" in record.error
    assert plane.registry.get("alpha").get_worker(wid) is None


@pytest.mark.asyncio
async def test_job_ending_before_ready_is_failed(tmp_path: Path) -> None:
    jobs = FakeJobs()
    plane = await _plane(tmp_path, jobs)
    await plane.create_endpoint("alpha")
    await plane.put_template(TEMPLATE)
    record = await plane.add_worker("alpha", TEMPLATE.name)
    await _settle()
    jobs.statuses[record.job_id] = "Failed"
    await plane.tick()
    assert record.phase == WorkerPhase.FAILED and "job.log" in record.error


@pytest.mark.asyncio
async def test_load_timeout_stops_job(tmp_path: Path) -> None:
    jobs = FakeJobs()
    plane = await _plane(tmp_path, jobs, load_timeout_s=0.0)
    await plane.create_endpoint("alpha")
    await plane.put_template(TEMPLATE)
    record = await plane.add_worker("alpha", TEMPLATE.name)
    await _settle()
    jobs.statuses[record.job_id] = "Running"
    await plane.tick()          # queued → loading
    await plane.tick()          # 超时 → 停 job
    assert record.phase == WorkerPhase.STOPPING and jobs.stopped == [record.job_id]
    await plane.tick()
    assert record.phase == WorkerPhase.FAILED and "加载超过" in record.error


@pytest.mark.asyncio
async def test_submit_error_that_actually_created_the_job_is_adopted(tmp_path: Path) -> None:
    jobs = FakeJobs()
    jobs.fail_create = "lost"
    plane = await _plane(tmp_path, jobs)
    await plane.create_endpoint("alpha")
    await plane.put_template(TEMPLATE)
    record = await plane.add_worker("alpha", TEMPLATE.name)
    await _settle()
    assert record.phase == WorkerPhase.QUEUED and record.job_id == 1001 and len(jobs.statuses) == 1


@pytest.mark.asyncio
async def test_submit_error_marks_failed(tmp_path: Path) -> None:
    jobs = FakeJobs()
    jobs.fail_create = "error"
    plane = await _plane(tmp_path, jobs)
    await plane.create_endpoint("alpha")
    await plane.put_template(TEMPLATE)
    record = await plane.add_worker("alpha", TEMPLATE.name)
    await _settle()
    assert record.phase == WorkerPhase.FAILED and "提交 job 失败" in record.error


@pytest.mark.asyncio
async def test_cancel_during_submit_still_stops_the_created_job(tmp_path: Path) -> None:
    jobs = FakeJobs()
    jobs.create_gate = asyncio.Event()
    plane = await _plane(tmp_path, jobs)
    await plane.create_endpoint("alpha")
    await plane.put_template(TEMPLATE)
    record = await plane.add_worker("alpha", TEMPLATE.name)
    await plane.drain(record.worker_id)                     # 提交还没返回就撤销
    await plane.tick()
    assert record.phase == WorkerPhase.STOPPING             # 等提交返回，不能提前判停
    jobs.create_gate.set()
    await _settle()
    assert jobs.stopped == [record.job_id]
    await plane.tick()
    assert record.phase == WorkerPhase.STOPPED


@pytest.mark.asyncio
async def test_move_idle_worker_switches_endpoint_without_restart(tmp_path: Path) -> None:
    jobs = FakeJobs()
    plane = await _plane(tmp_path, jobs)
    wid = await _ready_worker(plane, jobs)
    await plane.create_endpoint("beta")
    record = await plane.move(wid, "beta")
    assert record.endpoint == "beta" and record.move_to is None and not jobs.stopped
    assert plane.registry.get("alpha").get_worker(wid) is None
    moved = plane.registry.get("beta").get_worker(wid)
    assert moved is not None and not moved.draining


@pytest.mark.asyncio
async def test_slow_cctl_poll_does_not_block_panel_actions(tmp_path: Path) -> None:
    """后台查 cctl 很慢时，面板操作（这里是移动）不能排在整轮查询后面。"""

    jobs = FakeJobs()
    plane = await _plane(tmp_path, jobs)
    wid = await _ready_worker(plane, jobs)
    await plane.create_endpoint("beta")
    jobs.status_gate = asyncio.Event()
    tick = asyncio.create_task(plane.tick())
    await _settle()
    assert not tick.done()                                 # 卡在 cctl 查询上
    record = await asyncio.wait_for(plane.move(wid, "beta"), timeout=1.0)
    assert record.endpoint == "beta"
    jobs.status_gate.set()
    await tick
    assert plane.state.workers[wid].endpoint == "beta"


@pytest.mark.asyncio
async def test_transient_cctl_poll_error_clears_on_next_success(tmp_path: Path) -> None:
    jobs = FakeJobs()
    plane = await _plane(tmp_path, jobs)
    wid = await _ready_worker(plane, jobs)
    jobs.status_error = "cctl job get 无 JSON 输出: TLS handshake timeout"
    await plane.tick()
    record = plane.state.workers[wid]
    assert record.error and record.phase == WorkerPhase.READY
    jobs.status_error = None
    await plane.tick()
    assert record.error is None and record.phase == WorkerPhase.READY


@pytest.mark.asyncio
async def test_endpoint_with_workers_cannot_be_deleted(tmp_path: Path) -> None:
    jobs = FakeJobs()
    plane = await _plane(tmp_path, jobs)
    await _ready_worker(plane, jobs)
    with pytest.raises(ControlPlaneError, match="还有 Worker"):
        await plane.delete_endpoint("alpha")
    with pytest.raises(ControlPlaneError, match="保留端点"):
        await plane.create_endpoint("default")


@pytest.mark.asyncio
async def test_restart_readopts_ready_workers(tmp_path: Path) -> None:
    jobs = FakeJobs()
    plane = await _plane(tmp_path, jobs)
    wid = await _ready_worker(plane, jobs)
    await plane.stop()

    fresh = EndpointRegistry(_pool)                          # 模拟开发机上的 Gateway 进程重启
    again = await _plane(tmp_path, jobs, registry=fresh)
    assert again.state.workers[wid].phase == WorkerPhase.READY
    assert fresh.get("alpha").get_worker(wid) is not None


@pytest.mark.asyncio
async def test_forget_only_finished_workers(tmp_path: Path) -> None:
    jobs = FakeJobs()
    plane = await _plane(tmp_path, jobs)
    wid = await _ready_worker(plane, jobs)
    with pytest.raises(ControlPlaneError, match="只能删除已结束"):
        await plane.forget(wid)
    await plane.force_close(wid)
    await plane.tick()
    await plane.forget(wid)
    assert wid not in plane.state.workers
