"""控制面核心：端点 / Worker 模板 / Worker 生命周期，与 Gateway 的端点池保持一致。

生命周期推进（后台每 ``tick_s`` 秒一轮，所有改动都在同一把锁里）::

    submitting ─cctl create→ queued ─job Running→ loading ─就绪文件出现→ ready（登记进端点）
    ready ─驱逐→ draining ─手上会话结束→ 移出端点、停 job → stopping ─job 结束→ stopped
    ready ─强制关闭→ 打断会话、移出端点、停 job → stopping → stopped
    任一阶段 job 非预期结束 / 加载超时 / 提交失败 → failed（保留原因）

轮询节奏：就绪文件在共享存储上，每轮都看（开销小）；cctl 状态每 ``job_poll_s`` 秒查一次，
避免高频打 cctl。
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from control_plane.cctl import (
    CctlError, JobRunner, JobSettings, TERMINAL_STATUSES, description_for, job_url, worker_entry,
)
from control_plane.specs import SPEC_FILE, WorkerJobSpec, WorkerTemplate, read_ready, write_json_atomic
from control_plane.state import EndpointRecord, PhaseEvent, StateStore, WorkerPhase, WorkerRecord
from gateway_modules.endpoints import DEFAULT_ENDPOINT, EndpointError, EndpointRegistry
from gateway_modules.live_sessions import LiveSessionRegistry
from gateway_modules.worker_pool import WorkerPool

logger = logging.getLogger("control_plane")


class ControlPlaneError(ValueError):
    """可展示给面板用户的操作错误（参数不对、状态不允许）。"""


class ControlPlane:
    def __init__(
        self,
        *,
        root: Path,
        registry: EndpointRegistry,
        live_sessions: LiveSessionRegistry,
        jobs: JobRunner,
        repo_dir: str,
        python: str,
        tick_s: float = 5.0,
        job_poll_s: float = 60.0,
        load_timeout_s: float = 1800.0,
    ) -> None:
        self.root = root
        self.store = StateStore(root / "state.json")
        self.registry = registry
        self.live_sessions = live_sessions
        self.jobs = jobs
        self.repo_dir = repo_dir
        self.python = python
        self.tick_s = tick_s
        self.job_poll_s = job_poll_s
        self.load_timeout_s = load_timeout_s
        self._lock = asyncio.Lock()
        self._loop_task: Optional[asyncio.Task] = None
        self._next_job_poll: Dict[str, float] = {}
        self._submit_tasks: Dict[str, asyncio.Task] = {}

    @property
    def state(self):
        return self.store.state

    # ------------------------------------------------------------ 启停

    async def start(self, *, run_loop: bool = True) -> None:
        """恢复状态：建出端点，把仍在跑的 Worker 重新接管回来，然后开始后台推进。"""

        async with self._lock:
            self.store.load()
            for name in self.state.endpoints:
                if self.registry.get(name) is None:
                    await self.registry.create(name)
            for record in self.state.workers.values():
                if record.phase in (WorkerPhase.READY, WorkerPhase.DRAINING):
                    await self._register(record, draining=record.phase == WorkerPhase.DRAINING)
                if record.phase == WorkerPhase.SUBMITTING:
                    # 上次提交到一半进程就退了：按描述查 job 是否其实已经建出来。
                    self._spawn_submit(record)
            self.store.save()
        if run_loop:
            self._loop_task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        for task in self._submit_tasks.values():
            task.cancel()

    async def _loop(self) -> None:
        while True:
            await asyncio.sleep(self.tick_s)
            try:
                await self.tick()
            except Exception:  # noqa: BLE001 后台循环不能因单次异常退出
                logger.exception("control plane tick failed")

    # ------------------------------------------------------------ 端点

    async def create_endpoint(self, name: str, description: str = "") -> EndpointRecord:
        async with self._lock:
            if name == DEFAULT_ENDPOINT:
                raise ControlPlaneError("default 是保留端点（旧地址 /v1/realtime），不能在面板上管理")
            try:
                await self.registry.create(name)
            except EndpointError as exc:
                raise ControlPlaneError(str(exc)) from exc
            record = EndpointRecord(name=name, description=description)
            self.state.endpoints[name] = record
            self.store.save()
            return record

    async def update_endpoint(self, name: str, description: str) -> EndpointRecord:
        async with self._lock:
            record = self._endpoint(name)
            record.description = description
            self.store.save()
            return record

    async def delete_endpoint(self, name: str) -> None:
        async with self._lock:
            self._endpoint(name)
            active = [w.worker_id for w in self.state.workers.values()
                      if w.phase.active and (w.endpoint == name or w.move_to == name)]
            if active:
                raise ControlPlaneError(f"端点 {name} 下还有 Worker：{', '.join(active)}，先释放它们")
            try:
                await self.registry.delete(name)
            except EndpointError as exc:
                raise ControlPlaneError(str(exc)) from exc
            del self.state.endpoints[name]
            self.store.save()

    # ------------------------------------------------------------ 模板

    async def put_template(self, template: WorkerTemplate) -> WorkerTemplate:
        async with self._lock:
            self.state.templates[template.name] = template
            self.store.save()
            return template

    async def delete_template(self, name: str) -> None:
        async with self._lock:
            if name not in self.state.templates:
                raise ControlPlaneError(f"模板 {name} 不存在")
            del self.state.templates[name]          # 已起的 Worker 各自保存了一份模板副本，不受影响
            self.store.save()

    # ------------------------------------------------------------ Worker

    async def add_worker(self, endpoint: str, template_name: str) -> WorkerRecord:
        async with self._lock:
            self._endpoint(endpoint)
            template = self.state.templates.get(template_name)
            if template is None:
                raise ControlPlaneError(f"模板 {template_name} 不存在")
            return self._new_worker(endpoint, template)

    async def scale_up(self, worker_id: str) -> WorkerRecord:
        """扩容：用该 Worker 的模板，在同一端点再起一个副本。"""

        async with self._lock:
            source = self._worker(worker_id)
            self._endpoint(source.endpoint)
            return self._new_worker(source.endpoint, source.template)

    async def drain(self, worker_id: str) -> WorkerRecord:
        """驱逐：不再派新会话，手上的会话结束后停 job。还没就绪的 Worker 直接撤销。"""

        async with self._lock:
            record = self._worker(worker_id)
            self._require_active(record)
            record.release_mode = "drain"
            if record.phase == WorkerPhase.READY:
                pool = self.registry.get(record.endpoint)
                if pool is not None:
                    pool.set_draining(worker_id, True)
                record.set_phase(WorkerPhase.DRAINING, "驱逐：等手上的会话结束")
            elif record.phase != WorkerPhase.DRAINING:
                await self._stop_job(record, "撤销启动")
            self.store.save()
            return record

    async def force_close(self, worker_id: str) -> WorkerRecord:
        """强制关闭：不管有没有会话，立刻打断、移出端点并停 job。"""

        async with self._lock:
            record = self._worker(worker_id)
            self._require_active(record)
            record.release_mode = "force"
            interrupted = self.live_sessions.request_close(worker_id, "worker_removed")
            self._remove_from_pool(record)
            await self._stop_job(record, "强制关闭" + ("（打断了进行中的会话）" if interrupted else ""))
            self.store.save()
            return record

    async def move(self, worker_id: str, endpoint: str) -> WorkerRecord:
        """移到别的端点：先摘流，等它空闲后从原端点摘下、登记到新端点；job 不重启。"""

        async with self._lock:
            record = self._worker(worker_id)
            self._require_active(record)
            self._endpoint(endpoint)
            if endpoint == record.endpoint:
                raise ControlPlaneError("已经在这个端点下")
            if record.phase in (WorkerPhase.DRAINING, WorkerPhase.STOPPING):
                raise ControlPlaneError("Worker 正在释放，不能移动")
            if record.phase != WorkerPhase.READY:
                record.endpoint = endpoint           # 还没登记进任何端点：直接改目标
            else:
                record.move_to = endpoint
                pool = self.registry.get(record.endpoint)
                if pool is not None:
                    pool.set_draining(worker_id, True)
                await self._finish_move_if_idle(record)
            self.store.save()
            return record

    async def forget(self, worker_id: str) -> None:
        """从列表里删掉已结束（stopped / failed）的 Worker 记录。"""

        async with self._lock:
            record = self._worker(worker_id)
            if record.phase.active:
                raise ControlPlaneError("只能删除已结束的 Worker；在跑的先驱逐或强制关闭")
            if record.job_id is not None and record.job_status not in TERMINAL_STATUSES:
                raise ControlPlaneError("job 还没确认结束，稍后再删")
            del self.state.workers[worker_id]
            self.store.save()

    # ------------------------------------------------------------ 推进

    async def tick(self) -> None:
        async with self._lock:
            now = time.time()
            for record in list(self.state.workers.values()):
                try:
                    await self._advance(record, now)
                except CctlError as exc:
                    record.error = str(exc)
                    logger.warning("worker %s: %s", record.worker_id, exc)
            self.store.save()

    async def _advance(self, record: WorkerRecord, now: float) -> None:
        phase = record.phase
        if phase in (WorkerPhase.QUEUED, WorkerPhase.LOADING):
            ready = read_ready(Path(record.run_dir))
            if ready is not None and ready.worker_id == record.worker_id:
                record.host, record.port = ready.host, ready.port
                await self._register(record)
                record.set_phase(WorkerPhase.READY, f"就绪：{ready.host}:{ready.port}")
                return
            status = await self._poll_job(record, now)
            if status in TERMINAL_STATUSES:
                record.error = f"job 在就绪前结束（{status}），见 {record.run_dir}/job.log"
                record.set_phase(WorkerPhase.FAILED, record.error)
            elif status == "Running" and phase == WorkerPhase.QUEUED:
                record.set_phase(WorkerPhase.LOADING, "job 已运行，加载模型")
            elif phase == WorkerPhase.LOADING and now - record.phase_since > self.load_timeout_s:
                record.error = f"加载超过 {int(self.load_timeout_s)} 秒，已停 job；见 {record.run_dir}/job.log"
                await self._stop_job(record, record.error, failed=True)
        elif phase == WorkerPhase.READY:
            pool = self.registry.get(record.endpoint)
            if pool is not None and pool.get_worker(record.worker_id) is None:
                await self._register(record)          # 例如 Gateway 进程重启过
            if record.move_to:
                await self._finish_move_if_idle(record)
            status = await self._poll_job(record, now)
            if status in TERMINAL_STATUSES:
                self._remove_from_pool(record)
                record.error = f"job 非预期结束（{status}），见 {record.run_dir}/job.log"
                record.set_phase(WorkerPhase.FAILED, record.error)
        elif phase == WorkerPhase.DRAINING:
            pool = self.registry.get(record.endpoint)
            worker = pool.get_worker(record.worker_id) if pool is not None else None
            if worker is None or not worker.is_busy:
                self._remove_from_pool(record)
                await self._stop_job(record, "驱逐完成：手上已无会话")
        elif phase == WorkerPhase.STOPPING:
            if record.job_id is None:
                if self._submitting(record.worker_id):
                    return                    # 提交还没返回：等它拿到 job id 再停，避免 job 无人管理
                record.set_phase(WorkerPhase.STOPPED if record.release_mode else WorkerPhase.FAILED, "无 job")
                return
            status = await self._poll_job(record, now, interval=15.0)
            if status in TERMINAL_STATUSES:
                failed = record.release_mode is None
                record.set_phase(WorkerPhase.FAILED if failed else WorkerPhase.STOPPED, f"job {status}")

    async def _poll_job(self, record: WorkerRecord, now: float, interval: Optional[float] = None) -> Optional[str]:
        if record.job_id is None:
            return None
        if now < self._next_job_poll.get(record.worker_id, 0.0):
            return record.job_status
        record.job_status = await self.jobs.status(record.job_id)
        self._next_job_poll[record.worker_id] = now + min(interval or self.job_poll_s, self.job_poll_s)
        return record.job_status

    # ------------------------------------------------------------ 内部

    def _new_worker(self, endpoint: str, template: WorkerTemplate) -> WorkerRecord:
        seq = self.state.worker_seq.get(endpoint, 0) + 1
        self.state.worker_seq[endpoint] = seq
        worker_id = f"{endpoint}-{seq}"
        run_dir = self.root / "workers" / worker_id
        record = WorkerRecord(worker_id=worker_id, endpoint=endpoint, template=template.model_copy(),
                              run_dir=str(run_dir))
        record.set_phase(WorkerPhase.SUBMITTING, f"模板 {template.name}")
        self.state.workers[worker_id] = record
        self.store.save()
        self._spawn_submit(record)
        return record

    def _schedule_poll(self, worker_id: str, delay: float) -> None:
        """下次查 cctl 状态的时间；不晚于常规轮询间隔。"""

        self._next_job_poll[worker_id] = time.time() + min(delay, self.job_poll_s)

    def _spawn_submit(self, record: WorkerRecord) -> None:
        task = asyncio.create_task(self._submit(record.worker_id))
        self._submit_tasks[record.worker_id] = task
        task.add_done_callback(lambda _t, wid=record.worker_id: self._submit_tasks.pop(wid, None))

    def _submitting(self, worker_id: str) -> bool:
        task = self._submit_tasks.get(worker_id)
        return task is not None and not task.done()

    async def _submit(self, worker_id: str) -> None:
        """提交 cctl job（锁外执行，cctl 要好几秒）；报错时先按描述查是不是其实已经建出来了。"""

        record = self.state.workers.get(worker_id)
        if record is None:
            return
        run_dir = Path(record.run_dir)
        spec = WorkerJobSpec(worker_id=worker_id, endpoint=record.endpoint, template=record.template,
                             run_dir=str(run_dir))
        write_json_atomic(run_dir / SPEC_FILE, spec.model_dump(mode="json"))
        description = description_for(worker_id)
        job_id: Optional[int] = None
        error: Optional[str] = None
        if record.job_id is None:
            try:
                job_id = await self.jobs.find_by_description(description)
                if job_id is None:
                    job_id = await self.jobs.create(
                        description=description,
                        entry=worker_entry(repo_dir=self.repo_dir, python=self.python, run_dir=str(run_dir)),
                    )
            except CctlError as exc:
                try:
                    job_id = await self.jobs.find_by_description(description)
                except CctlError:
                    job_id = None
                if job_id is None:
                    error = f"提交 job 失败：{exc}"
        async with self._lock:
            record = self.state.workers.get(worker_id)
            if record is None or record.phase != WorkerPhase.SUBMITTING:
                if record is not None and job_id is not None and record.job_id is None:
                    # 提交期间被撤销：记下 job 并立即停掉，交给 STOPPING 确认结束。
                    record.job_id = job_id
                    await self.jobs.stop(job_id)
                    record.set_phase(WorkerPhase.STOPPING, f"撤销：停掉刚建出的 job {job_id}")
                    self._schedule_poll(worker_id, 15.0)
                self.store.save()
                return
            if job_id is not None:
                record.job_id = job_id
                record.set_phase(WorkerPhase.QUEUED, f"job {job_id}")
                self._schedule_poll(worker_id, 10.0)
            else:
                record.error = error
                record.set_phase(WorkerPhase.FAILED, error or "提交失败")
            self.store.save()

    async def _register(self, record: WorkerRecord, *, draining: bool = False) -> None:
        ready = read_ready(Path(record.run_dir))
        if ready is None:
            return
        pool = self.registry.get(record.endpoint)
        if pool is None:
            return
        await pool.register_worker(
            worker_id=record.worker_id, endpoint=f"{ready.host}:{ready.port}",
            labels={"template": record.template.name}, model_info=ready.model_info,
        )
        if draining or record.move_to:
            pool.set_draining(record.worker_id, True)

    def _remove_from_pool(self, record: WorkerRecord) -> None:
        pool = self.registry.get(record.endpoint)
        if pool is not None:
            pool.remove_worker(record.worker_id)

    async def _finish_move_if_idle(self, record: WorkerRecord) -> None:
        """手上没有会话时：从原端点摘下、登记到目标端点（job 不重启）。"""

        pool = self.registry.get(record.endpoint)
        worker = pool.get_worker(record.worker_id) if pool is not None else None
        if worker is not None and worker.is_busy:
            return
        if pool is not None:
            pool.remove_worker(record.worker_id)
        note = f"从 {record.endpoint} 移到 {record.move_to}"
        record.endpoint, record.move_to = record.move_to, None
        record.history.append(PhaseEvent(phase=record.phase, at=time.time(), note=note))
        await self._register(record)

    async def _stop_job(self, record: WorkerRecord, note: str, *, failed: bool = False) -> None:
        if record.job_id is not None and record.job_status not in TERMINAL_STATUSES:
            await self.jobs.stop(record.job_id)
            self._schedule_poll(record.worker_id, 15.0)
        if failed:
            record.release_mode = None
        record.set_phase(WorkerPhase.STOPPING, note)

    def _endpoint(self, name: str) -> EndpointRecord:
        record = self.state.endpoints.get(name)
        if record is None:
            raise ControlPlaneError(f"端点 {name} 不存在")
        return record

    def _worker(self, worker_id: str) -> WorkerRecord:
        record = self.state.workers.get(worker_id)
        if record is None:
            raise ControlPlaneError(f"Worker {worker_id} 不存在")
        return record

    @staticmethod
    def _require_active(record: WorkerRecord) -> None:
        if not record.phase.active:
            raise ControlPlaneError(f"Worker {record.worker_id} 已结束（{record.phase.value}）")

    # ------------------------------------------------------------ 面板视图

    def snapshot(self) -> Dict[str, Any]:
        now = time.time()
        endpoints: List[Dict[str, Any]] = []
        for name, endpoint in self.state.endpoints.items():
            pool: Optional[WorkerPool] = self.registry.get(name)
            workers = [self._worker_view(w, now) for w in self.state.workers.values()
                       if w.endpoint == name]
            endpoints.append({
                **endpoint.model_dump(mode="json"),
                "queue_length": pool.queue_length if pool else 0,
                "available": pool.idle_count if pool else 0,
                "workers": workers,
            })
        return {
            "endpoints": endpoints,
            "templates": [t.model_dump(mode="json") for t in self.state.templates.values()],
            "now": now,
        }

    def _worker_view(self, record: WorkerRecord, now: float) -> Dict[str, Any]:
        view = record.model_dump(mode="json")
        view["phase_since"] = record.phase_since
        cluster = getattr(getattr(self.jobs, "settings", None), "cluster", None) or JobSettings().cluster
        view["job_url"] = job_url(record.job_id, cluster) if record.job_id is not None else None
        pool = self.registry.get(record.endpoint)
        worker = pool.get_worker(record.worker_id) if pool is not None else None
        live = self.live_sessions.get(record.worker_id)
        view["gateway"] = None if worker is None else {
            "status": worker.status.value,
            "busy": worker.is_busy,
            "draining": worker.draining,
            "session_started_at": worker.task_started_at.timestamp() if worker.task_started_at else None,
            "ticket_id": live.ticket_id if live else None,
        }
        return view

    # ------------------------------------------------------------ 公开状态视图（无需登录）

    def public_snapshot(self) -> Dict[str, Any]:
        """对外公开的状态：端点、模型名、各 Worker 的占用情况。

        只给使用者判断「现在能不能用、要不要等」所需的信息；**不含** job id、Pod 内网地址、
        checkpoint 路径与报错原文（报错里常有目录路径）。已结束的 Worker 不列出。
        """

        now = time.time()
        endpoints: List[Dict[str, Any]] = []
        for name, endpoint in self.state.endpoints.items():
            pool = self.registry.get(name)
            active = [w for w in self.state.workers.values() if w.endpoint == name and w.phase.active]
            counts: Dict[str, int] = {}
            for w in active:
                counts[w.template.name] = counts.get(w.template.name, 0) + 1
            main = max(counts, key=counts.get) if counts else None
            main_template = next((w.template for w in active if w.template.name == main), None)
            workers = []
            for record in active:
                worker = pool.get_worker(record.worker_id) if pool is not None else None
                busy = bool(worker is not None and worker.is_busy)
                workers.append({
                    "worker_id": record.worker_id,
                    "state": _public_state(record, worker),
                    "since": record.phase_since,
                    "session_started_at": (worker.task_started_at.timestamp()
                                           if busy and worker.task_started_at else None),
                    "model": record.template.name,
                    "tts_pad_protocol": record.template.tts_pad_protocol,
                })
            endpoints.append({
                "name": name,
                "description": endpoint.description,
                "model": main,
                "tts_pad_protocol": main_template.tts_pad_protocol if main_template else None,
                "queue_length": pool.queue_length if pool else 0,
                "available": pool.idle_count if pool else 0,
                "workers": workers,
            })
        return {"now": now, "endpoints": endpoints}


def _public_state(record: WorkerRecord, worker: Any) -> str:
    """公开状态分类：queued / loading / idle / busy / unavailable / leaving。"""

    if record.phase in (WorkerPhase.SUBMITTING, WorkerPhase.QUEUED):
        return "queued"
    if record.phase == WorkerPhase.LOADING:
        return "loading"
    if record.phase in (WorkerPhase.DRAINING, WorkerPhase.STOPPING) or record.move_to:
        return "leaving"
    if worker is None:
        return "unavailable"
    if worker.is_busy:
        return "busy"
    return "idle" if worker.is_idle else "unavailable"
