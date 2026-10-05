"""模型协议违例：抛 ModelProtocolError（带违例类型 + 模型原文 context），
服务端以 reason=model_protocol_error 关 Session，与 backend_error 分开。

背景（2026-10-05）：违例以前是裸 RuntimeError → backend_error，评测侧分不清是模型错
还是基础设施错，只能当故障反复重跑，碰巧过一次就按那次算分 —— 分数虚高。
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from core.fc_duplex.model_adapter import O45FcDuplexModelAdapter
from core.fc_duplex.protocol_errors import PROTOCOL_VIOLATIONS, ModelProtocolError
from core.fc_duplex.system_input import FcSystemContentInput
from core.processors.unified import FcDuplexView
from core.schemas.fc_duplex import FcDuplexPrepareRequest, FcNonSpokenGenerateRequest
from py_backend.server import _fatal_reason

# 复用现有 View 测试里的 CPU fake model（不加载权重）
from tests.test_fc_duplex_view_streaming_text import _FakeFcModel


def _view(model: _FakeFcModel) -> FcDuplexView:
    adapter = O45FcDuplexModelAdapter(model)
    assert adapter.drops_unclassified_non_spoken_tokens is False   # O45 默认：违例即报错
    view = FcDuplexView(adapter)
    view.prepare(FcDuplexPrepareRequest(system=FcSystemContentInput()))
    return view


def _ids(model: _FakeFcModel, *parts: Any) -> list[int]:
    """parts 里字符串按普通文本编码，其余视为 special 名字（special_token_id_map 的键）。"""
    tok = model.fc_duplex.protocol_tokenizer
    out: list[int] = []
    for p in parts:
        if isinstance(p, str) and p.startswith("<") and p.endswith(">"):
            out.append(tok.special_token_id_map[p[1:-1]])
        else:
            out.extend(tok.encode_ordinary(p))
    return out


def test_ordinary_before_opener_raises_with_model_text_context() -> None:
    model = _FakeFcModel()
    model.non_spoken_results = [
        # 第 1 步：合法的 think 段，进入「最近生成」窗口
        {"token_ids": _ids(model, "<think_start>", "先查一下订单", "<think_end>", "<no_action>")},
        # 第 2 步：没开 opener 就吐普通文本 → 违例
        {"token_ids": _ids(model, "订单号是", "<no_action>")},
    ]
    view = _view(model)
    view.streaming_non_spoken_generate(FcNonSpokenGenerateRequest())

    with pytest.raises(ModelProtocolError) as ei:
        view.streaming_non_spoken_generate(FcNonSpokenGenerateRequest())

    err = ei.value
    assert err.code == "ordinary_before_opener"
    assert err.track == "non_spoken"
    ctx = err.context
    # 出错那一步：违例 token 被 ⟦⟧ 标出，之后的 token 也在
    assert "⟦" in ctx["step_text"] and "⟧" in ctx["step_text"]
    assert "no_action" in ctx["step_text"]
    # 跨 unit 的最近原文：能看到上一步合法的 think 内容
    assert "先查一下订单" in ctx["recent_text"]
    assert ctx["recent_text"].rstrip().endswith("⟧")
    assert "[non_spoken]" in ctx["recent_text_all_tracks"]
    assert "context_render_error" not in ctx


def test_end_token_without_stream_is_protocol_error() -> None:
    model = _FakeFcModel()
    model.non_spoken_results = [{"token_ids": _ids(model, "<think_end>")}]
    view = _view(model)
    with pytest.raises(ModelProtocolError) as ei:
        view.streaming_non_spoken_generate(FcNonSpokenGenerateRequest())
    assert ei.value.code == "end_without_stream"
    # 渲染的是模型实际吐出的 token 文本（</think>），不是语义键名
    assert ei.value.context["step_text"] == "⟦</think>⟧"


def test_stream_opened_before_previous_closed_is_protocol_error() -> None:
    model = _FakeFcModel()
    model.non_spoken_results = [{"token_ids": _ids(
        model, "<think_start>", "想一想", "<think_start>")}]
    view = _view(model)
    with pytest.raises(ModelProtocolError) as ei:
        view.streaming_non_spoken_generate(FcNonSpokenGenerateRequest())
    assert ei.value.code == "stream_opened_before_close"
    assert ei.value.context["active_stream"]["emitted_text_tail"] == "想一想"


def test_server_maps_protocol_error_to_distinct_reason_with_full_diagnostic() -> None:
    err = ModelProtocolError("ordinary_before_opener", "boom", track="non_spoken",
                             token_id=1, context={"step_text": "⟦x⟧"})
    reason, diag = _fatal_reason(err)
    assert reason == "model_protocol_error"
    assert diag["violation"] == "ordinary_before_opener"
    assert diag["violation_desc"] == PROTOCOL_VIOLATIONS["ordinary_before_opener"]
    assert diag["context"] == {"step_text": "⟦x⟧"}
    assert "boom" in diag["message"]           # 老字段保留，老客户端照常能读
    json.dumps(diag, ensure_ascii=False)       # 要能进 session.closed 帧

    reason, diag = _fatal_reason(RuntimeError("CUDA out of memory"))
    assert reason == "backend_error"
    assert diag == {"message": "CUDA out of memory"}


def test_protocol_error_is_still_runtime_error_for_old_callers() -> None:
    """继承 RuntimeError：没升级的调用方 `except RuntimeError` 照样接得住。"""
    assert issubclass(ModelProtocolError, RuntimeError)
