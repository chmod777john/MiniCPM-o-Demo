"""O5 模型对外包装（MiniCPMO.fc_duplex_*）与内层 FcDuplexCapability 的参数一致性回归。

真实故障（2026-10-05）：621daf4 给内层 ``FcDuplexCapability.streaming_spoken_generate``
加了 ``tts_pad_protocol``，backend 也开始传它，但外层包装
``MiniCPMO.fc_duplex_streaming_spoken_generate`` 没跟上，每个 Session 第一个 unit 就
``unexpected keyword argument 'tts_pad_protocol'``，全部 O5 评测 100% 失败。其它单测用的是
``**_`` 签名的假模型，测不出来，所以这里直接对真实类做检查。
"""

from __future__ import annotations

import inspect

import pytest

from modeling.o5.modeling_minicpmo_unified import FcDuplexCapability, MiniCPMO

PAIRS = [
    ("fc_duplex_streaming_spoken_generate", "streaming_spoken_generate"),
    ("fc_duplex_streaming_non_spoken_generate", "streaming_non_spoken_generate"),
]


def _params(fn) -> dict[str, inspect.Parameter]:
    return {k: v for k, v in inspect.signature(fn).parameters.items() if k != "self"}


@pytest.mark.parametrize("outer,inner", PAIRS)
def test_wrapper_accepts_every_inner_parameter(outer: str, inner: str) -> None:
    """外层包装必须接受内层的全部参数，且默认值一致。"""

    outer_p = _params(getattr(MiniCPMO, outer))
    inner_p = _params(getattr(FcDuplexCapability, inner))
    missing = sorted(set(inner_p) - set(outer_p))
    assert not missing, f"MiniCPMO.{outer} 缺少内层参数 {missing}"
    for name, p in inner_p.items():
        assert outer_p[name].default == p.default, (
            f"{name} 默认值不一致：外层 {outer_p[name].default!r} vs 内层 {p.default!r}"
        )


def test_spoken_wrapper_forwards_tts_pad_protocol() -> None:
    """外层包装要把 tts_pad_protocol 原样传给内层，而不是吞掉。"""

    calls: dict = {}

    class _Cap:
        def streaming_spoken_generate(self, **kwargs):
            calls.update(kwargs)
            return {}

    obj = MiniCPMO.__new__(MiniCPMO)
    object.__setattr__(obj, "_require_fc_duplex", lambda: _Cap())
    MiniCPMO.fc_duplex_streaming_spoken_generate(
        obj, max_tokens=3, decode_mode="greedy", tts_pad_protocol="revised"
    )
    assert calls == {"max_tokens": 3, "decode_mode": "greedy", "tts_pad_protocol": "revised"}
