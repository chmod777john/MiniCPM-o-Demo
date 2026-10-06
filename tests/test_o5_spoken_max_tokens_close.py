"""spoken 撞 max_tokens 时 runtime 怎么关 slot（modeling/o5 的 streaming_spoken_generate）。

背景（2026-10-06）：旧兜底在撞 max_tokens 时直接写 ``AI_SPOKEN_SLOT_END``，没有终止符 ——
这不是 SDK 的训练形态。协议状态机因此判失配，constrained_decoding 下本 Session 余下部分退回
不约束，随后模型采出协议违例（cd1005 tau 11 个 run 共 6 次，全部发生在这种失配之后）。
现在 runtime 先补 ``<|spoken_slot_eos|>`` 再关 slot，并打 warning。

不起模型：真实的 FcDuplexCapability 方法 + 真实 SDK tokenizer / registry，decoder 换成按脚本
吐 logits 的假实现；最后把 runtime 写进 KV 的 ``output_ids`` 交给协议状态机校验。
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch

from minicpm_o5_sdk.protocols.duplex.special_tokens import O5SpecialTokenKey as K
from minicpm_o5_sdk.protocols.duplex.special_tokens import O5SpecialTokenRegistry

from modeling.o5.modeling_minicpmo_unified import FcDuplexCapability, _ConstrainedDecoding

VOCAB = 248168


@pytest.fixture(scope="module")
def proto():
    from minicpm_o5_sdk import load_builtin_o5_tokenizer

    tok = load_builtin_o5_tokenizer()
    return SimpleNamespace(tok=tok, reg=O5SpecialTokenRegistry.from_tokenizer(tok))


def _logits(tid: int) -> torch.Tensor:
    x = torch.zeros(1, VOCAB)
    x[0, tid] = 100.0
    return x


class _ScriptedDecoder:
    """每喂一个 token 就按脚本给出下一步的 logits；脚本用完后一直吐最后一个。"""

    def __init__(self, script: list[int]):
        self.script = list(script)

    def embed_token(self, tid):
        return tid

    def embed_tokens(self, ids):
        return ids

    def feed(self, emb, return_logits=False):
        nxt = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        return (_logits(nxt), None) if return_logits else (None, None)


def _cap(proto, first: int, script: list[int]) -> FcDuplexCapability:
    cap = FcDuplexCapability.__new__(FcDuplexCapability)
    cap.K = K
    cap._registry = proto.reg
    cap._sdk_tokenizer = proto.tok
    cap.id2name = {proto.reg.get(k).token_id: k for k in K}
    cap._ensure_protocol = lambda: None
    cap._log_decision_logits = lambda **_: None
    cap.generate_audio = False
    cap.temperature = 1.0
    cap._constrained = None
    cap._current_unit_info = {}
    cap.decoder = _ScriptedDecoder(script)
    sid = lambda k: proto.reg.get(k).token_id  # noqa: E731
    sys_ids = [sid(K.IM_START)] + proto.tok.encode_ordinary("system\nhi") + [sid(K.IM_END)]
    audio = [sid(K.USER_AUDIO_SLOT_START), sid(K.AUDIO_PLACEHOLDER), sid(K.USER_AUDIO_SLOT_END)]
    cap.output_ids = sys_ids + [sid(K.UNIT_START)] + audio + [sid(K.AI_SPOKEN_SLOT_START)]
    cap._spoken_slot_open = True
    cap._spoken_logits = _logits(first)
    return cap


def _fsm_accepts_and_continues(proto, cap) -> dict:
    """runtime 写完 spoken slot 后，接着开 non-spoken：状态机必须仍然同步（没有 disabled_reason）。"""
    cd = cap._constrained or _ConstrainedDecoding(cap, "legacy")
    ns = proto.reg.get(K.AI_NON_SPOKEN_SLOT_START).token_id
    cd.step(cap.output_ids + [ns], _logits(proto.reg.get(K.NO_ACTION).token_id))
    return cd.report()


@pytest.mark.parametrize("policy", ["raise_protocol_error", "constrained_decoding"])
def test_reaching_max_tokens_appends_slot_eos_then_closes(proto, policy, caplog):
    word = proto.tok.encode_ordinary("red")[0]
    speak = proto.reg.get(K.SPEAK).token_id
    cap = _cap(proto, first=speak, script=[word])             # speak 之后一直说，永不收尾
    with caplog.at_level(logging.WARNING):
        out = cap.streaming_spoken_generate(max_tokens=24, decoding_policy=policy)
    assert out["spoken_generation_reached_max_tokens"] is True
    assert len(out["spoken_ids"]) == 24 and out["spoken_turn_eos"] is False
    assert cap.output_ids[-3:] == [word, proto.reg.get(K.SPOKEN_SLOT_EOS).token_id,
                                   proto.reg.get(K.AI_SPOKEN_SLOT_END).token_id]
    assert cap._current_unit_info["spoken_slot_eos_appended_by_runtime"] is True
    assert any("撞 max_tokens=24" in r.getMessage() for r in caplog.records)
    assert _fsm_accepts_and_continues(proto, cap)["disabled_reason"] is None


def test_turn_eos_path_unchanged(proto, caplog):
    word = proto.tok.encode_ordinary("ok")[0]
    speak, teos = proto.reg.get(K.SPEAK).token_id, proto.reg.get(K.SPOKEN_TURN_EOS).token_id
    cap = _cap(proto, first=speak, script=[word, teos])
    with caplog.at_level(logging.WARNING):
        out = cap.streaming_spoken_generate(max_tokens=24, decoding_policy="constrained_decoding")
    assert out["spoken_turn_eos"] is True and out["spoken_generation_reached_max_tokens"] is False
    assert cap.output_ids[-3:] == [teos, proto.reg.get(K.SPOKEN_SLOT_EOS).token_id,
                                   proto.reg.get(K.AI_SPOKEN_SLOT_END).token_id]
    assert cap._current_unit_info["spoken_slot_eos_appended_by_runtime"] is False
    assert not any("撞 max_tokens" in r.getMessage() for r in caplog.records)
    assert _fsm_accepts_and_continues(proto, cap)["disabled_reason"] is None


def test_listen_path_unchanged(proto):
    cap = _cap(proto, first=proto.reg.get(K.LISTEN).token_id, script=[0])
    out = cap.streaming_spoken_generate(max_tokens=24, decoding_policy="constrained_decoding")
    assert out["is_listen"] and not out["spoken_generation_reached_max_tokens"]
    assert cap.output_ids[-2:] == [proto.reg.get(K.LISTEN).token_id,
                                   proto.reg.get(K.AI_SPOKEN_SLOT_END).token_id]
    assert _fsm_accepts_and_continues(proto, cap)["disabled_reason"] is None
