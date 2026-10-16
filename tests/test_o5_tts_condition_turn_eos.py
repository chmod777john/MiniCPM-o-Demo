"""轮末 TTS condition 与训练对齐：收 spoken_turn_eos 的 token/hidden，并在 audio_bos 前补 text_eos。

背景（2026-10-16）：推理侧只收「正文 + tts_pad」，把 spoken_turn_eos 当普通 special 丢掉，
轮末也没有独立的 text_eos。训练侧两处都有：

- minicpm-o `multimodal_common/dataset/o5_agent_duplex/tts.py`
  ``condition_special_ids = {spoken_turn_eos, tts_pad}``；
- Megatron-LM-huawei `components/tts_backbone/minicpm_tts_bicodec/modeling_minicpmtts.py:1341-1354`
  ``interleaved_forward`` 给一轮话的最后一个 unit 加 text_eos，再给每个 unit 加 audio_bos。
  我们的训练入口固定 ``--interleaved`` + ``TTS_FORCE_STREAMING=1``，必走这条路径。

不起模型：真实的 FcDuplexCapability 方法 + 真实 SDK tokenizer / registry，decoder 与 tts 换成假实现。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from minicpm_o5_sdk.protocols.duplex.special_tokens import O5SpecialTokenKey as K
from minicpm_o5_sdk.protocols.duplex.special_tokens import O5SpecialTokenRegistry

from modeling.o5.modeling_minicpmo_unified import FcDuplexCapability

VOCAB = 248168
HIDDEN = 4


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
    def __init__(self, script: list[int]):
        self.script = list(script)

    def embed_token(self, tid):
        return tid

    def embed_tokens(self, ids):
        return ids

    def feed(self, emb, return_logits=False):
        nxt = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        # hidden 恒为 0：测试只核对 condition 里 token 的身份与顺序
        return (_logits(nxt), torch.zeros(1, 1, HIDDEN)) if return_logits else (None, None)


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
    cap.output_ids = [sid(K.UNIT_START), sid(K.AI_SPOKEN_SLOT_START)]
    cap._spoken_slot_open = True
    cap._spoken_logits = _logits(first)
    return cap


def _collect(proto, first: int, script: list[int]) -> tuple[list[int], bool]:
    """跑一个 spoken slot，返回 (进了 TTS condition 的 token ids, end_of_turn)。"""
    cap = _cap(proto, first=first, script=script)
    seen: dict = {}

    def _record(results, end_of_turn):
        seen["ids"] = [r[0] for r in results]
        seen["eot"] = end_of_turn
        return {"audio_waveform": None, "audio_sample_rate": None, "n_tts_tokens": 0,
                "cost_tts_prep": 0.0, "cost_tts": 0.0, "cost_token2wav": 0.0}

    cap._spoken_audio = _record  # type: ignore[method-assign]
    cap.streaming_spoken_generate(max_tokens=24, tts_pad_protocol="revised",
                                  decoding_policy="raise_protocol_error")
    return seen["ids"], seen["eot"]


def test_turn_eos_enters_tts_condition_after_the_payload(proto):
    """正文 + turn_eos：turn_eos 排在正文之后进 condition；speak / slot_eos 不进。"""
    word = proto.tok.encode_ordinary("ok")[0]
    speak = proto.reg.get(K.SPEAK).token_id
    teos = proto.reg.get(K.SPOKEN_TURN_EOS).token_id

    ids, eot = _collect(proto, first=speak, script=[word, teos])
    assert ids == [word, teos] and eot is True


def test_terminal_only_unit_still_has_a_condition_token(proto):
    """合法的 `speak → spoken_turn_eos`（本 unit 只结束上一段讲话、没有新正文）。

    修复前 results 为空，condition 退化成只剩一个 audio_bos。
    """
    speak = proto.reg.get(K.SPEAK).token_id
    teos = proto.reg.get(K.SPOKEN_TURN_EOS).token_id

    ids, eot = _collect(proto, first=speak, script=[teos])
    assert ids == [teos] and eot is True


def test_slot_eos_and_listen_do_not_enter_condition(proto):
    """turn 未结束的 unit（slot_eos 收尾）与 listen：condition 不收这两个 special。"""
    word = proto.tok.encode_ordinary("ok")[0]
    speak = proto.reg.get(K.SPEAK).token_id
    seos = proto.reg.get(K.SPOKEN_SLOT_EOS).token_id

    ids, eot = _collect(proto, first=speak, script=[word, seos])
    assert ids == [word] and eot is False

    ids, eot = _collect(proto, first=proto.reg.get(K.LISTEN).token_id, script=[0])
    assert ids == [] and eot is False


def test_tts_pad_still_enters_condition(proto):
    """2026-10-05 的 pad 出声修复不受影响。"""
    speak = proto.reg.get(K.SPEAK).token_id
    pad = proto.reg.get(K.TTS_PAD).token_id
    seos = proto.reg.get(K.SPOKEN_SLOT_EOS).token_id

    ids, eot = _collect(proto, first=speak, script=[pad, seos])
    assert ids == [pad] and eot is False


def test_legacy_env_restores_the_old_selection(proto, monkeypatch):
    """A/B 开关：O5_TTS_CONDITION_LEGACY=1 回到旧口径（turn_eos 被丢掉）。"""
    monkeypatch.setenv("O5_TTS_CONDITION_LEGACY", "1")
    speak = proto.reg.get(K.SPEAK).token_id
    teos = proto.reg.get(K.SPOKEN_TURN_EOS).token_id
    word = proto.tok.encode_ordinary("ok")[0]

    ids, eot = _collect(proto, first=speak, script=[word, teos])
    assert ids == [word] and eot is True


# ---------------------------------------------------------------- _tts_condition 的拼接顺序


def _fake_tts() -> SimpleNamespace:
    """emb_text 把 token id 铺成常量行，于是可以从 condition 直接读回 token 序列。"""

    def emb_text(ids: torch.Tensor) -> torch.Tensor:
        return ids.to(torch.float32).unsqueeze(-1).repeat(1, HIDDEN)

    emb_text.weight = SimpleNamespace(device=torch.device("cpu"))  # type: ignore[attr-defined]
    return SimpleNamespace(
        emb_text=emb_text,
        projector_semantic=lambda h: torch.zeros_like(h),
        projector_semantic_norm=None,
        audio_bos_token_id=248100,
        config=SimpleNamespace(text_eos_token_id=248101, normalize_projected_hidden=False),
    )


def _condition_tokens(results, end_of_turn: bool) -> list[int]:
    cap = FcDuplexCapability.__new__(FcDuplexCapability)
    cap.model = SimpleNamespace(tts=_fake_tts())
    out = cap._tts_condition(results, end_of_turn)
    assert out.shape[0] == 1 and out.shape[-1] == HIDDEN
    return [int(round(float(row[0]))) for row in out[0]]


def test_end_of_turn_appends_text_eos_before_audio_bos(proto):
    word = proto.tok.encode_ordinary("ok")[0]
    teos = proto.reg.get(K.SPOKEN_TURN_EOS).token_id
    results = [(word, torch.zeros(1, 1, HIDDEN), True), (teos, torch.zeros(1, 1, HIDDEN), True)]

    # 一轮话的最后一个 unit：… + text_eos + audio_bos
    assert _condition_tokens(results, True) == [word, teos, 248101, 248100]
    # turn 中间的 unit：只有 audio_bos
    assert _condition_tokens(results, False) == [word, teos, 248100]


def test_empty_results_keeps_audio_bos_and_adds_text_eos_only_at_turn_end():
    assert _condition_tokens([], False) == [248100]
    assert _condition_tokens([], True) == [248101, 248100]


def test_legacy_env_drops_the_text_eos(monkeypatch):
    monkeypatch.setenv("O5_TTS_CONDITION_LEGACY", "1")
    results = [(123, torch.zeros(1, 1, HIDDEN), True)]
    assert _condition_tokens(results, True) == [123, 248100]
