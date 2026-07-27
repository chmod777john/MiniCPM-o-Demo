"""Semantic Realtime API v2 的 Python WebSocket 客户端。

客户端记录完整双向 history，逐 Unit 发送 canonical float32 音频，等待
``response.unit.committed``，并在 TrainingData 指定的 Unit 前注入原始字符串工具结果。
所有下行事件原样保留，包括 ``response.non_spoken.end``。
"""

from __future__ import annotations

import asyncio
import json
import ssl
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

import websockets

from .models import (
    FcApiEvaluationErrorCategory,
    FcApiExpectedToolCall,
    FcApiHistoryEntry,
    FcApiSemanticSessionResult,
    FcApiToolResponseSchedule,
    FcApiTrainingDataScenario,
    JsonObject,
)


@dataclass(frozen=True)
class FcApiSemanticV2ClientError(RuntimeError):
    """带稳定分类的 Semantic API client 错误。"""

    category: FcApiEvaluationErrorCategory
    message: str
    unit_index: int | None = None
    event_type: str | None = None

    def __str__(self) -> str:
        """返回人类可读错误文本。"""

        return self.message


class FcApiSemanticV2Client:
    """无 GPU 本地侧 Semantic API v2 WebSocket client。"""

    def __init__(
        self,
        *,
        base_url: str,
        insecure: bool = False,
        queue_timeout_sec: float = 120.0,
        event_timeout_sec: float = 180.0,
        close_timeout_sec: float = 5.0,
    ) -> None:
        """初始化客户端。

        参数:
            base_url: Gateway HTTP(S) 根地址。
            insecure: 是否关闭 WSS 证书校验。
            queue_timeout_sec: 等待 Worker 分配的超时秒数。
            event_timeout_sec: 等待单个 API 事件的超时秒数。
            close_timeout_sec: 发送 close 后等待关闭事件的超时秒数。
        """

        self.base_url = base_url
        self.insecure = insecure
        self.queue_timeout_sec = queue_timeout_sec
        self.event_timeout_sec = event_timeout_sec
        self.close_timeout_sec = close_timeout_sec

    async def evaluate(
        self,
        scenario: FcApiTrainingDataScenario,
    ) -> FcApiSemanticSessionResult:
        """执行一条完整 TrainingData Semantic API 会话。

        参数:
            scenario: 已构造的 session.init、Unit 输入和工具调度。

        返回:
            全量双向 history、resume parser history 和会话身份。

        异常:
            FcApiSemanticV2ClientError: 队列、事件顺序、工具匹配或会话协议失败。
        """

        ssl_context = (
            ssl._create_unverified_context() if self.insecure else None
        )
        history: list[FcApiHistoryEntry] = []
        protocol_history: list[JsonObject] = []
        state = _ClientHistoryState(
            history=history,
            protocol_history=protocol_history,
        )
        api_to_gt_tool_call_id: dict[str, str] = {}
        gt_to_api_tool_call_id: dict[str, str] = {}
        next_expected_tool_ordinal = 0
        committed_units: list[int] = []

        try:
            async with websockets.connect(
                self._websocket_url(),
                ssl=ssl_context,
                max_size=128 * 1024 * 1024,
                open_timeout=self.queue_timeout_sec,
            ) as websocket:
                await self._wait_queue_done(websocket, state)
                await self._send(
                    websocket,
                    state,
                    scenario.session_init,
                    add_to_protocol_history=True,
                )
                session_created = await self._wait_session_created(
                    websocket,
                    state,
                )
                self._validate_session_identity(
                    event=session_created,
                    scenario=scenario,
                )

                for unit in scenario.units:
                    await self._send_due_tool_results(
                        websocket=websocket,
                        state=state,
                        schedules=scenario.tool_responses,
                        current_unit_index=unit.unit_index,
                        gt_to_api_tool_call_id=gt_to_api_tool_call_id,
                    )
                    await self._send(
                        websocket,
                        state,
                        unit.to_frame(),
                        add_to_protocol_history=True,
                    )
                    (
                        next_expected_tool_ordinal,
                        committed_event,
                    ) = await self._consume_until_committed(
                        websocket=websocket,
                        state=state,
                        target_unit_index=unit.unit_index,
                        expected_tool_calls=scenario.expected_tool_calls,
                        next_expected_tool_ordinal=(
                            next_expected_tool_ordinal
                        ),
                        api_to_gt_tool_call_id=api_to_gt_tool_call_id,
                        gt_to_api_tool_call_id=gt_to_api_tool_call_id,
                    )
                    committed_units.append(int(committed_event["unit_index"]))

                self._assert_all_tool_results_sent(
                    schedules=scenario.tool_responses,
                    gt_to_api_tool_call_id=gt_to_api_tool_call_id,
                    sent_gt_tool_call_ids=state.sent_gt_tool_call_ids,
                )
                await self._send(
                    websocket,
                    state,
                    {"type": "session.close"},
                    add_to_protocol_history=True,
                )
                await self._drain_close(websocket, state)
        except FcApiSemanticV2ClientError:
            raise
        except Exception as exc:
            raise FcApiSemanticV2ClientError(
                category=FcApiEvaluationErrorCategory.WEBSOCKET_ERROR,
                message=f"Semantic API WebSocket 失败: {type(exc).__name__}: {exc}",
            ) from exc

        return FcApiSemanticSessionResult(
            history=history,
            protocol_history=protocol_history,
            session_created=session_created,
            committed_unit_indices=committed_units,
            api_to_gt_tool_call_id=api_to_gt_tool_call_id,
        )

    def _websocket_url(self) -> str:
        """把 HTTP(S) Gateway URL 转为 realtime WebSocket URL。"""

        parsed = urlsplit(
            self.base_url.rstrip("/") + "/v1/realtime?mode=audio"
        )
        return urlunsplit(
            (
                "wss" if parsed.scheme == "https" else "ws",
                parsed.netloc,
                parsed.path,
                parsed.query,
                "",
            )
        )

    async def _wait_queue_done(
        self,
        websocket: Any,
        state: "_ClientHistoryState",
    ) -> None:
        """等待 ``session.queue_done``，同时记录全部排队事件。"""

        while True:
            event = await self._receive(
                websocket,
                state,
                timeout_sec=self.queue_timeout_sec,
                add_to_protocol_history=False,
            )
            event_type = str(event.get("type") or "")
            if event_type == "session.queue_done":
                return
            if event_type in {"error", "session.closed"}:
                raise FcApiSemanticV2ClientError(
                    category=FcApiEvaluationErrorCategory.API_PROTOCOL_ERROR,
                    message=f"Worker 排队失败: {event}",
                    event_type=event_type,
                )

    async def _wait_session_created(
        self,
        websocket: Any,
        state: "_ClientHistoryState",
    ) -> JsonObject:
        """等待 ``session.created``，保留初始化阶段全部下行事件。"""

        while True:
            event = await self._receive(
                websocket,
                state,
                timeout_sec=self.event_timeout_sec,
                add_to_protocol_history=True,
            )
            event_type = str(event.get("type") or "")
            if event_type == "session.created":
                return event
            if event_type in {"error", "session.closed"}:
                raise FcApiSemanticV2ClientError(
                    category=FcApiEvaluationErrorCategory.API_PROTOCOL_ERROR,
                    message=f"session.init 失败: {event}",
                    event_type=event_type,
                )

    async def _consume_until_committed(
        self,
        *,
        websocket: Any,
        state: "_ClientHistoryState",
        target_unit_index: int,
        expected_tool_calls: list[FcApiExpectedToolCall],
        next_expected_tool_ordinal: int,
        api_to_gt_tool_call_id: dict[str, str],
        gt_to_api_tool_call_id: dict[str, str],
    ) -> tuple[int, JsonObject]:
        """消费一个 Unit 的全部下行事件直到 committed。"""

        saw_non_spoken_end = False
        while True:
            event = await self._receive(
                websocket,
                state,
                timeout_sec=self.event_timeout_sec,
                add_to_protocol_history=True,
            )
            event_type = str(event.get("type") or "")
            if event_type == "response.non_spoken.end":
                if int(event.get("unit_index", -1)) == target_unit_index:
                    saw_non_spoken_end = True
                continue
            if event_type == "response.tool_call.done":
                next_expected_tool_ordinal = self._bind_tool_call(
                    event=event,
                    expected_tool_calls=expected_tool_calls,
                    next_expected_tool_ordinal=next_expected_tool_ordinal,
                    api_to_gt_tool_call_id=api_to_gt_tool_call_id,
                    gt_to_api_tool_call_id=gt_to_api_tool_call_id,
                )
                continue
            if event_type in {
                "error",
                "session.closed",
                "session.resume.failed",
            }:
                raise FcApiSemanticV2ClientError(
                    category=FcApiEvaluationErrorCategory.API_PROTOCOL_ERROR,
                    message=f"Unit {target_unit_index} 处理失败: {event}",
                    unit_index=target_unit_index,
                    event_type=event_type,
                )
            if event_type != "response.unit.committed":
                continue
            actual_unit = int(event.get("unit_index", -1))
            if actual_unit != target_unit_index:
                raise FcApiSemanticV2ClientError(
                    category=FcApiEvaluationErrorCategory.API_PROTOCOL_ERROR,
                    message=(
                        "committed Unit 顺序错误: "
                        f"expected={target_unit_index}, actual={actual_unit}"
                    ),
                    unit_index=actual_unit,
                    event_type=event_type,
                )
            if not saw_non_spoken_end:
                raise FcApiSemanticV2ClientError(
                    category=(
                        FcApiEvaluationErrorCategory.MISSING_NON_SPOKEN_END
                    ),
                    message=(
                        f"Unit {target_unit_index} committed 前缺少 "
                        "response.non_spoken.end"
                    ),
                    unit_index=target_unit_index,
                    event_type=event_type,
                )
            return next_expected_tool_ordinal, event

    def _bind_tool_call(
        self,
        *,
        event: JsonObject,
        expected_tool_calls: list[FcApiExpectedToolCall],
        next_expected_tool_ordinal: int,
        api_to_gt_tool_call_id: dict[str, str],
        gt_to_api_tool_call_id: dict[str, str],
    ) -> int:
        """把 API tool_call.done 按确定性顺序绑定到 GT fixed ID。"""

        if next_expected_tool_ordinal >= len(expected_tool_calls):
            raise FcApiSemanticV2ClientError(
                category=FcApiEvaluationErrorCategory.TOOL_CALL_MISMATCH,
                message=f"API 产生了 GT 中不存在的额外工具调用: {event}",
                event_type="response.tool_call.done",
            )
        expected = expected_tool_calls[next_expected_tool_ordinal]
        api_id = str(event.get("tool_call_id") or "")
        call = event.get("call")
        if not api_id or not isinstance(call, Mapping):
            raise FcApiSemanticV2ClientError(
                category=FcApiEvaluationErrorCategory.TOOL_CALL_MISMATCH,
                message=f"tool_call.done 缺少 ID 或 call: {event}",
                event_type="response.tool_call.done",
            )
        actual_arguments = call.get("arguments")
        if isinstance(actual_arguments, str):
            try:
                actual_arguments = json.loads(actual_arguments)
            except json.JSONDecodeError as exc:
                raise FcApiSemanticV2ClientError(
                    category=FcApiEvaluationErrorCategory.TOOL_CALL_MISMATCH,
                    message=f"工具参数不是合法 JSON: {actual_arguments!r}",
                    event_type="response.tool_call.done",
                ) from exc
        if (
            str(call.get("name") or "") != expected.name
            or actual_arguments != expected.arguments
        ):
            raise FcApiSemanticV2ClientError(
                category=FcApiEvaluationErrorCategory.TOOL_CALL_MISMATCH,
                message=(
                    "API 工具调用与 GT 不一致: "
                    f"expected={expected.model_dump(mode='json')}, actual={call}"
                ),
                event_type="response.tool_call.done",
            )
        api_to_gt_tool_call_id[api_id] = expected.tool_call_id
        gt_to_api_tool_call_id[expected.tool_call_id] = api_id
        return next_expected_tool_ordinal + 1

    async def _send_due_tool_results(
        self,
        *,
        websocket: Any,
        state: "_ClientHistoryState",
        schedules: list[FcApiToolResponseSchedule],
        current_unit_index: int,
        gt_to_api_tool_call_id: dict[str, str],
    ) -> None:
        """在 GT perceived_input_gate Unit 前发送已闭合工具的原始结果。"""

        for schedule in schedules:
            if (
                schedule.send_before_unit_index != current_unit_index
                or schedule.tool_call_id in state.sent_gt_tool_call_ids
            ):
                continue
            api_id = gt_to_api_tool_call_id.get(schedule.tool_call_id)
            if api_id is None:
                raise FcApiSemanticV2ClientError(
                    category=FcApiEvaluationErrorCategory.TOOL_CALL_MISMATCH,
                    message=(
                        "GT 工具结果到注入 Unit 时仍无已完成 API call: "
                        f"gt_id={schedule.tool_call_id}, "
                        f"unit={current_unit_index}"
                    ),
                    unit_index=current_unit_index,
                )
            frame: JsonObject = {
                "type": "input.tool_result",
                "tool_call_id": api_id,
                "content": schedule.content,
            }
            await self._send(
                websocket,
                state,
                frame,
                add_to_protocol_history=True,
            )
            state.sent_gt_tool_call_ids.add(schedule.tool_call_id)

    def _assert_all_tool_results_sent(
        self,
        *,
        schedules: list[FcApiToolResponseSchedule],
        gt_to_api_tool_call_id: dict[str, str],
        sent_gt_tool_call_ids: set[str],
    ) -> None:
        """确认所有 GT tool response 均已映射并上行。"""

        expected = {item.tool_call_id for item in schedules}
        missing = sorted(expected - sent_gt_tool_call_ids)
        unmapped = sorted(expected - set(gt_to_api_tool_call_id))
        if missing or unmapped:
            raise FcApiSemanticV2ClientError(
                category=FcApiEvaluationErrorCategory.TOOL_CALL_MISMATCH,
                message=(
                    "工具结果未完整发送: "
                    f"missing={missing}, unmapped={unmapped}"
                ),
            )

    def _validate_session_identity(
        self,
        *,
        event: JsonObject,
        scenario: FcApiTrainingDataScenario,
    ) -> None:
        """校验 API 返回的 model/tokenizer/Profile 身份。"""

        identity = event.get("resume")
        if not isinstance(identity, Mapping):
            identity = event
        checks = {
            "tokenizer_target": scenario.tokenizer_target,
            "checkpoint_profile_id": scenario.session_init["payload"][
                "checkpoint_profile_id"
            ],
            "model": scenario.session_init["payload"]["model"],
        }
        mismatches = {
            key: {"expected": expected, "actual": identity.get(key)}
            for key, expected in checks.items()
            if identity.get(key) not in {None, "", "unknown", expected}
        }
        actual_fingerprint = identity.get("tokenizer_fingerprint")
        if (
            isinstance(actual_fingerprint, Mapping)
            and dict(actual_fingerprint)
            != scenario.tokenizer_fingerprint
        ):
            mismatches["tokenizer_fingerprint"] = {
                "expected": scenario.tokenizer_fingerprint,
                "actual": dict(actual_fingerprint),
            }
        if mismatches:
            raise FcApiSemanticV2ClientError(
                category=(
                    FcApiEvaluationErrorCategory.PROFILE_OR_TARGET_MISMATCH
                ),
                message=f"session.created 身份不匹配: {mismatches}",
                event_type="session.created",
            )

    async def _drain_close(
        self,
        websocket: Any,
        state: "_ClientHistoryState",
    ) -> None:
        """发送 close 后尽力等待 ``session.closed``。"""

        while True:
            try:
                event = await self._receive(
                    websocket,
                    state,
                    timeout_sec=self.close_timeout_sec,
                    add_to_protocol_history=True,
                )
            except (asyncio.TimeoutError, websockets.ConnectionClosed):
                return
            if event.get("type") == "session.closed":
                return

    async def _send(
        self,
        websocket: Any,
        state: "_ClientHistoryState",
        frame: JsonObject,
        *,
        add_to_protocol_history: bool,
        wire_frame: JsonObject | None = None,
    ) -> None:
        """发送并记录一条上行帧。"""

        state.append(
            direction="up",
            event=frame,
            add_to_protocol_history=add_to_protocol_history,
            protocol_event=wire_frame,
        )
        await websocket.send(
            json.dumps(wire_frame or frame, ensure_ascii=False)
        )

    async def _receive(
        self,
        websocket: Any,
        state: "_ClientHistoryState",
        *,
        timeout_sec: float,
        add_to_protocol_history: bool,
    ) -> JsonObject:
        """接收、解析并原样记录一条下行帧。"""

        raw = await asyncio.wait_for(websocket.recv(), timeout=timeout_sec)
        event = json.loads(raw)
        if not isinstance(event, dict):
            raise FcApiSemanticV2ClientError(
                category=FcApiEvaluationErrorCategory.API_PROTOCOL_ERROR,
                message=f"API 事件必须是 JSON object: {event!r}",
            )
        state.append(
            direction="down",
            event=event,
            add_to_protocol_history=add_to_protocol_history,
        )
        return event


@dataclass
class _ClientHistoryState:
    """客户端 history 序号和 parser history 的局部状态。"""

    history: list[FcApiHistoryEntry]
    protocol_history: list[JsonObject]
    sent_gt_tool_call_ids: set[str] = field(default_factory=set)

    def append(
        self,
        *,
        direction: Literal["up", "down"],
        event: JsonObject,
        add_to_protocol_history: bool,
        protocol_event: JsonObject | None = None,
    ) -> None:
        """追加一条 history，并按需同步 resume parser history。"""

        if direction not in {"up", "down"}:
            raise ValueError(f"非法 history direction: {direction}")
        self.history.append(
            FcApiHistoryEntry(
                sequence=len(self.history),
                direction=direction,
                event=dict(event),
            )
        )
        if add_to_protocol_history:
            self.protocol_history.append(dict(protocol_event or event))
