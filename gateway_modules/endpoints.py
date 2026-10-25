"""Internal pool registry: each pool owns worker instances, a queue and health checks.

Public APIs select a model ID via model_catalog; pool names are control-plane details.
The default pool exists for deployment registration, not as an implicit public model.
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from .worker_pool import WorkerConnection, WorkerPool

logger = logging.getLogger("gateway.endpoints")

DEFAULT_ENDPOINT = "default"
ENDPOINT_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")


class EndpointError(ValueError):
    """端点名不合法、重名、不存在或不可删。"""


def validate_endpoint_name(name: str) -> str:
    if not ENDPOINT_NAME_RE.match(name or ""):
        raise EndpointError("端点名只能是 1-40 位小写字母、数字或连字符，且以字母或数字开头")
    return name


class EndpointRegistry:
    """端点名 → WorkerPool。所有端点共用同一套池参数（由 ``pool_factory`` 决定）。"""

    def __init__(self, pool_factory: Callable[[], WorkerPool]) -> None:
        self._pool_factory = pool_factory
        self._pools: Dict[str, WorkerPool] = {}

    async def start(self, default_pool: Optional[WorkerPool] = None) -> WorkerPool:
        """建好并启动 default 端点，返回它的池（可传入预先配置好静态 Worker 的池）。"""

        pool = default_pool or self._pool_factory()
        await pool.start()
        self._pools[DEFAULT_ENDPOINT] = pool
        return pool

    async def stop(self) -> None:
        for pool in self._pools.values():
            await pool.stop()

    def names(self) -> List[str]:
        return list(self._pools)

    def items(self) -> Iterator[Tuple[str, WorkerPool]]:
        return iter(list(self._pools.items()))

    def get(self, name: str) -> Optional[WorkerPool]:
        return self._pools.get(name)

    async def create(self, name: str) -> WorkerPool:
        validate_endpoint_name(name)
        if name in self._pools:
            raise EndpointError(f"端点 {name} 已存在")
        pool = self._pool_factory()
        await pool.start()
        self._pools[name] = pool
        logger.info("Endpoint created: %s", name)
        return pool

    async def delete(self, name: str) -> None:
        if name == DEFAULT_ENDPOINT:
            raise EndpointError("default 端点不能删除")
        pool = self._pools.get(name)
        if pool is None:
            raise EndpointError(f"端点 {name} 不存在")
        if pool.workers:
            raise EndpointError(f"端点 {name} 下还有 {len(pool.workers)} 个 Worker，先释放它们")
        await pool.stop()
        del self._pools[name]
        logger.info("Endpoint deleted: %s", name)

    def find_worker(self, worker_id: str) -> Optional[Tuple[str, WorkerPool, WorkerConnection]]:
        """按 worker_id 找到它所在的端点。一个 Worker 只属于一个端点。"""

        for name, pool in self._pools.items():
            worker = pool.get_worker(worker_id)
            if worker is not None:
                return name, pool, worker
        return None
