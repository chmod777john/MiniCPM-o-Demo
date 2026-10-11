"""standalone input_event 的服务端链路：消息校验 → 运行时排队 → view 组装 → 回执。

用户定的规则（2026-10-10）：
- 落在哪个 Unit：与工具结果共用一个队列、按到达顺序，进入下一个被处理的音频 Unit；
- 回执：``response.unit.started`` 新增 ``standalone_events``，回显被该 Unit 吃掉的 event_id；
- O45 已停止维护：O45 收到 standalone 直接报错，不做续传。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from core.fc_duplex.model_adapter import O45FcDuplexModelAdapter, O5FcDuplexModelAdapter
from core.processors.unified import FcDuplexView
from core.schemas.fc_duplex import (
    FcDuplexPrefillRequest,
    FcStandaloneEvent,
    FcToolResponse,
)
from minicpm_o5_sdk import O5TokenizerID, load_builtin_tokenizer
from py_backend.fc_duplex_runtime import FcDuplexSessionRuntime


# ---------------------------------------------------------------- schema


def test_standalone_event_requires_non_empty_text_content() -> None:
    """SDK 只支持单段文本：content 必须是非空字符串，其余形态一律拒绝。"""

    assert FcStandaloneEvent(content="进度").event_id is None
    assert FcStandaloneEvent(content="进度", event_id="e1").event_id == "e1"
    for bad in ({"content": ""}, {"content": ["a"]}, {"content": {"text": "a"}}, {}):
        with pytest.raises(ValidationError):
            FcStandaloneEvent(**bad)
    with pytest.raises(ValidationError):
        FcStandaloneEvent(content="a", event_id="")
    with pytest.raises(ValidationError):  # 不收未知字段，免得客户端拼错字段名被静默吞掉
        FcStandaloneEvent(content="a", contents="a")


def test_prefill_request_rejects_both_legacy_and_ordered_event_fields() -> None:
    """input_events 是保序的统一列表；与旧的 tool_responses 同时给出就无法确定顺序。"""

    with pytest.raises(ValidationError):
        FcDuplexPrefillRequest(
            tool_responses=[FcToolResponse(call_id="c", content="x")],
            input_events=[FcStandaloneEvent(content="y")],
        )


# ---------------------------------------------------------------- view


class _CapturingModel:
    """记录 view 交给模型的 input_event 列表。"""

    def __init__(self, tokenizer_id: O5TokenizerID) -> None:
        self.fc_duplex = SimpleNamespace(protocol_tokenizer=load_builtin_tokenizer(tokenizer_id))
        self.prefill_events: list[Any] = []

    def fc_duplex_streaming_prefill(self, **kwargs: Any) -> dict[str, Any]:
        self.prefill_events.append(kwargs.get("tool_responses"))
        return {"unit": 0, "n_audio": 0, "has_event": bool(kwargs.get("tool_responses"))}


def test_view_keeps_arrival_order_after_framework_events() -> None:
    """同 Unit：先框架的 tool_started，再按到达顺序放工具结果与 standalone。"""

    model = _CapturingModel(O5TokenizerID.O5)
    view = FcDuplexView(O5FcDuplexModelAdapter(model))  # type: ignore[arg-type]
    first = view.tool_call_manager.register_tool_call({"name": "delegate_task", "arguments": {}})
    view.streaming_prefill(FcDuplexPrefillRequest())  # 下一个 Unit 注入 tool_started
    second = view.tool_call_manager.register_tool_call({"name": "web_search", "arguments": {}})

    result = view.streaming_prefill(
        FcDuplexPrefillRequest(
            input_events=[
                FcStandaloneEvent(content="已找到两篇文献", event_id="p1"),
                FcToolResponse(call_id=first, content="最终结论"),
                FcStandaloneEvent(content="还在核对价格"),
            ]
        )
    )

    sent = model.prefill_events[-1]
    assert [e["type"] for e in sent] == [
        "tool_started", "standalone_event", "tool_response", "standalone_event",
    ]
    assert sent[0]["call_id"] == second
    assert sent[1] == {"type": "standalone_event", "content": "已找到两篇文献", "event_id": "p1"}
    assert sent[3]["event_id"] is None
    assert [e["type"] for e in result.tool_events] == [e["type"] for e in sent]


def test_view_keeps_legacy_tool_responses_field_working() -> None:
    """只给旧字段 tool_responses 的调用方（离线脚本）行为不变。"""

    model = _CapturingModel(O5TokenizerID.O5)
    view = FcDuplexView(O5FcDuplexModelAdapter(model))  # type: ignore[arg-type]
    call = view.tool_call_manager.register_tool_call({"name": "get_weather", "arguments": {}})
    view.streaming_prefill(FcDuplexPrefillRequest())
    view.streaming_prefill(
        FcDuplexPrefillRequest(tool_responses=[FcToolResponse(call_id=call, content="晴")])
    )
    assert [e["type"] for e in model.prefill_events[-1]] == ["tool_response"]


def test_o45_rejects_standalone_event_explicitly() -> None:
    """O45 已停止维护：收到 standalone 要报明确原因，而不是渲染层的「缺 call_id」。"""

    model = _CapturingModel(O5TokenizerID.O45_FC)
    view = FcDuplexView(O45FcDuplexModelAdapter(model))  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="O45.*standalone_event"):
        view.streaming_prefill(
            FcDuplexPrefillRequest(input_events=[FcStandaloneEvent(content="x")])
        )
    assert model.prefill_events == []


# ---------------------------------------------------------------- runtime


class _RecordingBackend:
    """只记录 prefill 收到的 input_events。"""

    def __init__(self) -> None:
        self.prefills: list[dict[str, Any]] = []

    def fc_duplex_prefill(self, **fields: Any) -> None:
        self.prefills.append(fields)


def _runtime(backend: Any, events: list[dict[str, Any]] | None = None) -> FcDuplexSessionRuntime:
    async def send(event_type: str, **fields: Any) -> None:
        if events is not None:
            events.append({"type": event_type, **fields})

    return FcDuplexSessionRuntime(session_id="sess_sa", backend=backend, send=send)


@pytest.mark.asyncio
async def test_runtime_queues_standalone_and_tool_results_in_arrival_order() -> None:
    """两种事件共用一个队列：下一个音频 Unit 按到达顺序整体吃掉。"""

    backend = _RecordingBackend()
    runtime = _runtime(backend)
    runtime._api_to_internal["tc_1"] = "fc_call_000001"

    await runtime.push({"type": "standalone_event", "content": "进度一", "event_id": "p1"})
    await runtime.push({"type": "tool_result", "tool_call_id": "tc_1", "content": "结果"})
    await runtime.push({"type": "standalone_event", "content": "进度二"})

    queued = list(runtime._pending_input_events)
    assert [type(e).__name__ for e in queued] == [
        "FcStandaloneEvent", "FcToolResponse", "FcStandaloneEvent",
    ]
    assert queued[0].event_id == "p1" and queued[2].event_id is None
    assert queued[1].call_id == "fc_call_000001"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"type": "standalone_event"},
        {"type": "standalone_event", "content": ""},
        {"type": "standalone_event", "content": [{"kind": "text", "text": "a"}]},
        {"type": "standalone_event", "content": "a", "event_id": ""},
        {"type": "standalone_event", "contents": "a"},
    ],
)
async def test_runtime_rejects_malformed_standalone_event(payload: dict[str, Any]) -> None:
    """不合法的 standalone 直接报错，不进队列、不会被当成音频。"""

    runtime = _runtime(_RecordingBackend())
    with pytest.raises((ValueError, ValidationError)):
        await runtime.push(payload)
    assert runtime._pending_input_events == []


@pytest.mark.asyncio
async def test_unit_started_reports_standalone_events_separately() -> None:
    """回执：standalone 进 standalone_events，tool_events 保持原样不受影响。"""

    events: list[dict[str, Any]] = []
    runtime = _runtime(_RecordingBackend(), events)
    runtime._internal_to_api["fc_call_000001"] = "tc_000002"
    prefill = SimpleNamespace(
        tool_events=[
            {"type": "tool_started", "call_id": "fc_call_000001"},
            {"type": "standalone_event", "content": "进度", "event_id": "p1"},
            {"type": "tool_response", "call_id": "fc_call_000001", "content": "x"},
            {"type": "standalone_event", "content": "进度二", "event_id": None},
        ]
    )
    await runtime._emit_unit_input_events(prefill, unit_index=3, input_id="u3")

    assert events == [{
        "type": "response.unit.started",
        "input_id": "u3",
        "unit_index": 3,
        "tool_events": [
            {"type": "tool_started", "tool_call_id": "tc_000002"},
            {"type": "tool_result", "tool_call_id": "tc_000002"},
        ],
        "standalone_events": [{"event_id": "p1"}, {"event_id": None}],
    }]
