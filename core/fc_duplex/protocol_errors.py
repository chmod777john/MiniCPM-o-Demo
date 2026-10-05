"""模型输出违反 FC 流协议时抛出的异常。

🔴 **这是被评模型自身的能力缺陷，不是服务故障。** 与 OOM、CUDA 错误、连接断开等
基础设施故障必须区分开：服务端据此以 ``reason="model_protocol_error"``（而不是
``backend_error``）关闭 Session；评测客户端据此把该样本判为「模型失败、只跑一次」，
而不是当成可重试的故障反复重跑（2026-10-05 用户明确的口径）。

本模块刻意不依赖推理代码，server / gateway 侧可以轻量 import。
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# 违例类型 → 一句话说明。code 是稳定的机器可读标识，下游（评测、统计）按它聚合。
PROTOCOL_VIOLATIONS: Dict[str, str] = {
    "ordinary_before_opener": "流 opener（think/tool_call/speak）之前生成了普通文本 token",
    "special_without_semantic_key": "生成了无法归类的非普通 token（无 semantic key）",
    "stream_opened_before_close": "上一个流尚未关闭就打开了新流",
    "aggregate_kind_changed": "同一段 non_spoken 内 opener 种类发生了变化",
    "end_without_stream": "生成了结束 token，但没有与之匹配的已打开流",
}


class ModelProtocolError(RuntimeError):
    """模型生成的 token 序列违反 FC 流协议。

    Attributes:
        code: ``PROTOCOL_VIOLATIONS`` 的键。
        track: ``spoken`` / ``non_spoken``。
        token_id: 触发违例的那个 token。
        context: 定位用的原始上下文（模型最近生成的原文、当前流状态等），
            由 ``diagnostic()`` 原样带进 ``session.closed`` 事件。
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        track: str,
        token_id: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(f"[model_protocol_error:{code}] {message}")
        self.code = code
        self.detail = message
        self.track = track
        self.token_id = token_id
        self.context = context or {}

    def diagnostic(self) -> Dict[str, Any]:
        """``session.closed.diagnostic`` 的内容。``message`` 保留旧字段名，兼容老客户端。"""

        return {
            "message": str(self),
            "violation": self.code,
            "violation_desc": PROTOCOL_VIOLATIONS.get(self.code, ""),
            "detail": self.detail,
            "track": self.track,
            "token_id": self.token_id,
            "context": self.context,
        }
