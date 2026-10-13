"""Opt-in transport reconnection for ordinary desktop duplex sessions.

The existing gateway relay retains its worker connection for a bounded grace
period. Reattaching a browser never repeats session.init or model input.
"""
from __future__ import annotations

import asyncio
import json
import logging
import secrets
from collections import deque
from contextlib import suppress
from typing import Any, Awaitable, Callable

from fastapi import WebSocket, WebSocketDisconnect

_GRACE_SECONDS = 30
_MAX_BUFFER_BYTES = 16 * 1024 * 1024
logger = logging.getLogger(__name__)
_sessions: dict[str, "RetainedSocket"] = {}


class RetainedSocket:
    def __init__(self, ws: WebSocket, identity: tuple[str, str]):
        self.original = ws
        self.identity = identity
        self.token = secrets.token_urlsafe(32)
        self.socket: WebSocket | None = None
        self.inputs: asyncio.Queue = asyncio.Queue()
        self.pending: deque[str] = deque()
        self.pending_bytes = 0
        self.lock = asyncio.Lock()
        self.expiry: asyncio.Task | None = None
        self.task: asyncio.Task | None = None
        self.closed = False
        self.token_delivered = False

    def __getattr__(self, name):
        return getattr(self.original, name)

    async def accept(self):
        # Physical sockets are accepted by bind(), before the existing relay runs.
        pass

    async def iter_text(self):
        while True:
            raw = await self.inputs.get()
            if raw is None:
                return
            yield raw

    async def send_json(self, data: Any):
        await self.send_text(json.dumps(data, ensure_ascii=False))

    async def send_text(self, raw: str):
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            data = {}
        is_created = isinstance(data, dict) and data.get("type") == "session.created"
        if is_created:
            data.update(reconnect_token=self.token, reconnect_timeout_s=_GRACE_SECONDS)
            raw = json.dumps(data, ensure_ascii=False)
        async with self.lock:
            if self.closed:
                raise WebSocketDisconnect()
            if self.socket is not None:
                try:
                    await self.socket.send_text(raw)
                    if is_created:
                        self.token_delivered = True
                    return
                except (WebSocketDisconnect, RuntimeError, OSError):
                    await self._detach_locked(self.socket)
                    if self.closed:
                        raise WebSocketDisconnect()
            size = len(raw.encode("utf-8"))
            if self.pending_bytes + size > _MAX_BUFFER_BYTES:
                await self._finish_locked(1013, "Reconnect output buffer full")
                raise WebSocketDisconnect()
            self.pending.append(raw)
            self.pending_bytes += size

    def _expire_later(self):
        if self.closed or (self.expiry is not None and not self.expiry.done()):
            return
        async def expire():
            await asyncio.sleep(_GRACE_SECONDS)
            await self.close(code=1000, reason="Reconnect grace expired")
        self.expiry = asyncio.create_task(expire())

    async def bind(self, ws: WebSocket, *, resumed: bool) -> bool:
        async with self.lock:
            if self.closed or self.socket is not None:
                await ws.close(code=4409, reason="Session unavailable or already attached")
                return False
            await ws.accept()
            self.socket = ws
            try:
                if resumed:
                    await ws.send_json({"type": "session.transport_resumed",
                                        "reconnect_timeout_s": _GRACE_SECONDS})
                while self.pending:
                    raw = self.pending[0]
                    await ws.send_text(raw)
                    self.pending.popleft()
                    self.pending_bytes -= len(raw.encode("utf-8"))
            except (WebSocketDisconnect, RuntimeError, OSError):
                await self._detach_locked(ws)
                return False
            if self.expiry is not None:
                self.expiry.cancel()
                self.expiry = None
            return True

    async def _detach_locked(self, ws: WebSocket):
        if self.socket is not ws:
            return
        self.socket = None
        with suppress(WebSocketDisconnect, RuntimeError, OSError):
            await ws.close()
        if self.token_delivered:
            self._expire_later()
        else:
            # Before session.created there is no token for the client to resume.
            # Cancel its queue slot / preparation instead of retaining an orphan.
            await self._finish_locked(1000, "Client left before session creation")

    async def detach(self, ws: WebSocket):
        async with self.lock:
            await self._detach_locked(ws)

    async def receive(self, ws: WebSocket):
        try:
            async for raw in ws.iter_text():
                if self.socket is not ws or self.closed:
                    break
                await self.inputs.put(raw)
        except (WebSocketDisconnect, RuntimeError, OSError):
            pass
        finally:
            await self.detach(ws)

    async def _finish_locked(self, code: int, reason: str):
        if self.closed:
            return
        self.closed = True
        _sessions.pop(self.token, None)
        await self.inputs.put(None)
        if self.expiry is not None and self.expiry is not asyncio.current_task():
            self.expiry.cancel()
        ws, self.socket = self.socket, None
        self.pending.clear()
        self.pending_bytes = 0
        if ws is not None:
            with suppress(WebSocketDisconnect, RuntimeError, OSError):
                await ws.close(code=code, reason=reason)

    async def close(self, code=1000, reason=""):
        async with self.lock:
            await self._finish_locked(code, reason)


async def serve(ws: WebSocket, *, identity: tuple[str, str],
                run: Callable[[Any], Awaitable[None]]):
    token = ws.query_params.get("reconnect_token")
    if token:
        retained = _sessions.get(token)
        if retained is None or retained.identity != identity:
            await ws.close(code=4404, reason="Reconnect session expired")
            return
        try:
            if await retained.bind(ws, resumed=True):
                await retained.receive(ws)
        finally:
            await retained.detach(ws)
        return

    retained = RetainedSocket(ws, identity)
    if not await retained.bind(ws, resumed=False):
        return
    _sessions[retained.token] = retained

    async def relay():
        try:
            await run(retained)
        except Exception:
            logger.exception("Ordinary duplex relay failed: endpoint=%s mode=%s", *identity)
        finally:
            await retained.close()

    retained.task = asyncio.create_task(relay())
    # Consume the task result; the existing relay logs protocol/worker failures.
    def completed(task):
        if not task.cancelled():
            task.exception()
    retained.task.add_done_callback(completed)
    await retained.receive(ws)
