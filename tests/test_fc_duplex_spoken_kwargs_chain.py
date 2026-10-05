"""spoken_generate 参数从 view 一路到真实模型签名的贯通性回归（O5 + O45）。

背景（2026-10-05）：``tts_pad_protocol`` 先后两次在调用链上漏掉一层——O5 外层包装
``MiniCPMO.fc_duplex_streaming_spoken_generate`` 没加（a8cc956 修，期间全部 O5 评测 100% 失败），
O45 的两层模型签名也都没有。上层 ``core/processors/unified.py`` 对所有模型无条件传它，
而单测里的假模型都是 ``**_`` 签名，看不出来。

这里不起模型：拿 view 实际发出的 kwargs，经**真实 adapter**，``bind`` 到**真实模型类**的签名上。
链条上任何一层再漏参数，这条就红。
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from core.fc_duplex.model_adapter import O45FcDuplexModelAdapter, O5FcDuplexModelAdapter
from modeling.o5.modeling_minicpmo_unified import MiniCPMO as O5MiniCPMO

# core/processors/unified.py 里 streaming_spoken_generate 实际传给 adapter 的参数
VIEW_KWARGS = {"max_tokens": 24, "decode_mode": "greedy", "tts_pad_protocol": "legacy",
               "decoding_policy": "raise_protocol_error"}
NS_KWARGS = {"decode_mode": "greedy", "max_tokens": 1, "close_reason": None,
             "decoding_policy": "raise_protocol_error"}


def _recording_model(real_cls: type) -> tuple[Any, dict]:
    """一个按真实类签名校验入参的假模型：签名对不上就抛 TypeError，和线上一样。"""

    sp = inspect.signature(real_cls.fc_duplex_streaming_spoken_generate)
    ns = inspect.signature(real_cls.fc_duplex_streaming_non_spoken_generate)
    seen: dict = {}

    class _Model:
        def fc_duplex_streaming_spoken_generate(self, **kwargs: Any) -> dict:
            bound = sp.bind(self, **kwargs)  # 多给 / 写错参数名在这里炸
            seen.update({k: v for k, v in bound.arguments.items() if k != "self"})
            return {}

        def fc_duplex_streaming_non_spoken_generate(self, **kwargs: Any) -> dict:
            bound = ns.bind(self, **kwargs)
            seen.update({"ns_" + k: v for k, v in bound.arguments.items() if k != "self"})
            return {}

    return _Model(), seen


def _o45_model_cls() -> type:
    mod = pytest.importorskip("modeling.o45.modeling_minicpmo_unified")
    return mod.MiniCPMO


def test_o5_chain_accepts_and_forwards_tts_pad_protocol() -> None:
    model, seen = _recording_model(O5MiniCPMO)
    O5FcDuplexModelAdapter(model).streaming_spoken_generate(**VIEW_KWARGS)
    assert seen["tts_pad_protocol"] == "legacy"


@pytest.mark.parametrize("value", ["legacy", "revised"])
def test_o5_chain_passes_both_protocols(value: str) -> None:
    model, seen = _recording_model(O5MiniCPMO)
    O5FcDuplexModelAdapter(model).streaming_spoken_generate(**{**VIEW_KWARGS, "tts_pad_protocol": value})
    assert seen["tts_pad_protocol"] == value


def test_o45_chain_drops_tts_pad_protocol() -> None:
    """O45 模型签名里没有这个参数：adapter 必须吃掉，否则 O45 部署第一个 unit 就崩。"""

    model, seen = _recording_model(_o45_model_cls())
    O45FcDuplexModelAdapter(model).streaming_spoken_generate(**VIEW_KWARGS)
    assert "tts_pad_protocol" not in seen and "decoding_policy" not in seen
    assert seen == {"max_tokens": 24, "decode_mode": "greedy"}


def test_o45_chain_rejects_revised() -> None:
    """O45 只有 legacy 一种 pad 形态，要求 revised 必须明确报错而不是静默忽略。"""

    model, _ = _recording_model(_o45_model_cls())
    with pytest.raises(ValueError, match="O45 模型不支持"):
        O45FcDuplexModelAdapter(model).streaming_spoken_generate(
            **{**VIEW_KWARGS, "tts_pad_protocol": "revised"}
        )


@pytest.mark.parametrize("policy", ["raise_protocol_error", "constrained_decoding"])
def test_o5_chain_passes_decoding_policy_both_tracks(policy: str) -> None:
    model, seen = _recording_model(O5MiniCPMO)
    adapter = O5FcDuplexModelAdapter(model)
    adapter.streaming_spoken_generate(**{**VIEW_KWARGS, "decoding_policy": policy})
    adapter.streaming_non_spoken_generate(**{**NS_KWARGS, "decoding_policy": policy})
    assert seen["decoding_policy"] == policy and seen["ns_decoding_policy"] == policy


def test_o45_chain_drops_decoding_policy_non_spoken() -> None:
    model, seen = _recording_model(_o45_model_cls())
    O45FcDuplexModelAdapter(model).streaming_non_spoken_generate(**NS_KWARGS)
    assert "ns_decoding_policy" not in seen


def test_o45_chain_rejects_constrained() -> None:
    model, _ = _recording_model(_o45_model_cls())
    with pytest.raises(ValueError, match="decoding_policy"):
        O45FcDuplexModelAdapter(model).streaming_non_spoken_generate(
            **{**NS_KWARGS, "decoding_policy": "constrained_decoding"})
