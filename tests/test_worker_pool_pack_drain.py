"""pack 放置与摘流（drain）的单元测试。无需 GPU，纯调度逻辑。

pack 的目标：离线评测收尾时让远端扩容主机尽早「整机空闲」从而可以退出。规则：
1. 本机（127.0.0.1 / localhost 注册的）Worker 有空就用本机；
2. 否则放到「预计全部跑完最晚」的远端主机；完全空闲的远端主机排在最后；
3. 并列时选更早注册的主机。
"""

from datetime import datetime, timedelta

import pytest

from gateway_modules.models import EtaConfig, GatewayWorkerStatus
from gateway_modules.worker_pool import WorkerConnection, WorkerPool


def make_pool(hosts: dict, placement: str = "pack") -> WorkerPool:
    """hosts = {host: 副本数}；按 dict 顺序注册（registered_at 依次递增）。"""
    pool = WorkerPool(
        worker_addresses=[],
        eta_config=EtaConfig(eta_omni_duplex_s=600.0),
        placement=placement,
    )
    t0 = datetime(2026, 10, 4, 12, 0, 0)
    order = 0
    for host, n in hosts.items():
        for i in range(n):
            wid = f"{host}-{i}"
            pool.workers[wid] = WorkerConnection(
                worker_id=wid, host=host, port=22400 + i, gpu_id=order,
                status=GatewayWorkerStatus.IDLE, registered_at=t0 + timedelta(seconds=order),
            )
            order += 1
    return pool


def busy(pool: WorkerPool, wid: str, started_ago_s: float) -> None:
    w = pool.workers[wid]
    w.mark_busy(GatewayWorkerStatus.DUPLEX_ACTIVE, "omni_duplex")
    w.task_started_at = datetime.now() - timedelta(seconds=started_ago_s)


class TestPackPlacement:
    def test_home_host_first(self):
        pool = make_pool({"10.0.0.2": 2, "127.0.0.1": 2})
        assert pool._get_idle_worker("omni_duplex").host == "127.0.0.1"

    def test_fills_first_registered_remote_when_all_remote_idle(self):
        pool = make_pool({"127.0.0.1": 1, "10.0.0.2": 2, "10.0.0.3": 2})
        busy(pool, "127.0.0.1-0", 10)
        assert pool._get_idle_worker("omni_duplex").host == "10.0.0.2"

    def test_does_not_wake_idle_remote_host(self):
        # 10.0.0.2 完全空闲；10.0.0.3 有一个在跑 → 放 10.0.0.3，让 10.0.0.2 保持可释放
        pool = make_pool({"127.0.0.1": 1, "10.0.0.2": 2, "10.0.0.3": 2})
        busy(pool, "127.0.0.1-0", 10)
        busy(pool, "10.0.0.3-0", 10)
        assert pool._get_idle_worker("omni_duplex").host == "10.0.0.3"

    def test_latest_release_wins_over_busiest(self):
        """用户指出的反例：7 个快跑完的槽，不如 1 个刚开始的槽活得久。"""
        pool = make_pool({"127.0.0.1": 1, "10.0.0.2": 8, "10.0.0.3": 8})
        busy(pool, "127.0.0.1-0", 10)
        # 历史上会话都在 1000 s 左右结束
        pool._durations.extend([1000.0] * 50)
        for i in range(7):                       # 10.0.0.2：7 个槽都已跑了 950 s，快结束
            busy(pool, f"10.0.0.2-{i}", 950)
        busy(pool, "10.0.0.3-0", 5)              # 10.0.0.3：1 个刚开始，还要 ~995 s
        assert pool._get_idle_worker("omni_duplex").host == "10.0.0.3"

    def test_spread_unchanged(self):
        pool = make_pool({"10.0.0.2": 1, "127.0.0.1": 1}, placement="spread")
        # spread 不偏向本机，按 GPU 负载 + worker_id
        assert pool._get_idle_worker("omni_duplex").worker_id == "10.0.0.2-0"

    def test_invalid_placement_rejected(self):
        with pytest.raises(ValueError):
            WorkerPool(worker_addresses=[], placement="nope")


class TestExpectedRemaining:
    def test_conditional_on_elapsed(self):
        pool = make_pool({"127.0.0.1": 1})
        pool._durations.extend([100.0] * 10 + [1000.0] * 10)
        assert pool._expected_remaining(0, "omni_duplex") == pytest.approx(550.0)
        # 已跑 500 s → 只剩那些 1000 s 的会话
        assert pool._expected_remaining(500, "omni_duplex") == pytest.approx(500.0)

    def test_fallback_to_eta_when_few_samples(self):
        pool = make_pool({"127.0.0.1": 1})
        assert pool._expected_remaining(100, "omni_duplex") == pytest.approx(500.0)

    def test_release_records_duration(self):
        pool = make_pool({"127.0.0.1": 1})
        w = pool.workers["127.0.0.1-0"]
        w.mark_busy(GatewayWorkerStatus.DUPLEX_ACTIVE, "omni_duplex")
        pool.release_worker(w, request_type="omni_duplex", duration_s=42.0)
        assert list(pool._durations) == [42.0]


class TestDrain:
    def test_drained_worker_not_dispatched(self):
        pool = make_pool({"127.0.0.1": 1, "10.0.0.2": 1})
        pool.set_draining("127.0.0.1-0", True)
        assert pool._get_idle_worker("omni_duplex").host == "10.0.0.2"
        pool.set_draining("10.0.0.2-0", True)
        assert pool._get_idle_worker("omni_duplex") is None

    def test_drain_keeps_running_session_and_survives_release(self):
        pool = make_pool({"10.0.0.2": 1})
        w = pool.workers["10.0.0.2-0"]
        w.mark_busy(GatewayWorkerStatus.DUPLEX_ACTIVE, "omni_duplex")
        pool.set_draining(w.worker_id, True)
        assert w.is_busy                          # 在跑会话不受影响
        pool.release_worker(w, request_type="omni_duplex", duration_s=1.0)
        assert w.status == GatewayWorkerStatus.IDLE and not w.is_idle   # 空闲了但仍不接单

    def test_undrain_dispatches_waiting_request(self):
        import asyncio

        async def run():
            pool = make_pool({"10.0.0.2": 1})
            pool.set_draining("10.0.0.2-0", True)
            ticket, fut = pool.enqueue("omni_duplex")
            assert not fut.done()
            pool.set_draining("10.0.0.2-0", False)
            assert fut.done() and fut.result().worker_id == "10.0.0.2-0"

        asyncio.run(run())

    def test_unknown_worker(self):
        pool = make_pool({"127.0.0.1": 1})
        assert pool.set_draining("nope", True) is None

    def test_info_exposes_draining(self):
        pool = make_pool({"10.0.0.2": 1})
        pool.set_draining("10.0.0.2-0", True)
        assert pool.workers["10.0.0.2-0"].to_info().draining is True
