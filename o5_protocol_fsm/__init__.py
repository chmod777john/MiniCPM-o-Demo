"""O5 双工 FC 协议状态机（只依赖 minicpm_o5_sdk，可独立验证）。

- :mod:`.fsm`：状态机本体，符号层面（普通 token 统一为 ORDINARY，prefill 段各压成一个符号）
- :mod:`.vocab`：符号 ⇄ 真实 token id，推理侧建 mask、验证侧回放 / 展开
- :mod:`.explorer`：逐步走状态机的网页（``python -m o5_protocol_fsm.explorer.server``）
"""

from .fsm import (
    Abstract,
    Actor,
    FsmConfig,
    O5ProtocolFsm,
    O5ProtocolViolation,
    StepOptions,
    Track,
)

__all__ = [
    "Abstract",
    "Actor",
    "FsmConfig",
    "O5ProtocolFsm",
    "O5ProtocolViolation",
    "StepOptions",
    "Track",
]
