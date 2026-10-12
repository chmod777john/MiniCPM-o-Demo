"""正在进行的 API 会话登记：按 worker_id 找到会话并要求它收尾（强制关闭 Worker 用）。

透传协程在拿到 Worker 后登记一个 ``LiveSession``，会话结束时注销。
``request_close`` 只置一个事件；真正发 ``session.closed`` 和关闭连接仍由透传协程自己完成，
避免两处同时写同一条 WebSocket。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class LiveSession:
    worker_id: str
    ticket_id: str
    endpoint: str
    close_event: asyncio.Event = field(default_factory=asyncio.Event)
    close_reason: Optional[str] = None


class LiveSessionRegistry:
    def __init__(self) -> None:
        self._by_worker: Dict[str, LiveSession] = {}

    def register(self, session: LiveSession) -> None:
        self._by_worker[session.worker_id] = session

    def unregister(self, session: LiveSession) -> None:
        if self._by_worker.get(session.worker_id) is session:
            del self._by_worker[session.worker_id]

    def get(self, worker_id: str) -> Optional[LiveSession]:
        return self._by_worker.get(worker_id)

    def request_close(self, worker_id: str, reason: str) -> bool:
        """让该 Worker 上的会话收尾；没有会话返回 False。"""

        session = self._by_worker.get(worker_id)
        if session is None:
            return False
        session.close_reason = reason
        session.close_event.set()
        return True
