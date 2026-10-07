"""decoding_policy=constrained_decoding 的驱动（modeling/o5 的 _ConstrainedDecoding）离线测试。

不起模型：用真实 SDK tokenizer / registry 和**真实词表大小**的 logits（CPU），手工拼出推理侧
``output_ids`` 的样子，验证：
- 合法集合唯一时直接返回 token、不采样；
- 否则 mask 之后 argmax 一定落在合法集合里 —— 即使非法 token 的 logit 最高；
- 复现 2026-10-05 tau / FDB 实测的两类违例，确认被挡住。
"""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
import torch

from minicpm_o5_sdk.protocols.duplex.special_tokens import O5SpecialTokenKey as K
from minicpm_o5_sdk.protocols.duplex.special_tokens import O5SpecialTokenRegistry

from modeling.o5.modeling_minicpmo_unified import _ConstrainedDecoding


@pytest.fixture(scope="module")
def cap():
    from minicpm_o5_sdk import load_builtin_o5_tokenizer

    tok = load_builtin_o5_tokenizer()
    return SimpleNamespace(_sdk_tokenizer=tok, _registry=O5SpecialTokenRegistry.from_tokenizer(tok))


@pytest.fixture(scope="module")
def vocab_size() -> int:
    return 248168


def _id(cap, key) -> int:
    return cap._registry.get(key).token_id


def _prefix(cap) -> list[int]:
    """system 预填 + 一个 unit 的 prefill，停在 spoken slot 首步之前。"""
    sys_ids = [_id(cap, K.IM_START)] + cap._sdk_tokenizer.encode_ordinary("system\nhi") + [_id(cap, K.IM_END)]
    audio = [_id(cap, K.USER_AUDIO_SLOT_START), _id(cap, K.AUDIO_PLACEHOLDER), _id(cap, K.USER_AUDIO_SLOT_END)]
    return sys_ids + [_id(cap, K.UNIT_START)] + audio + [_id(cap, K.AI_SPOKEN_SLOT_START)]


def _close_unit_listen(cap) -> list[int]:
    """listen + 空动作，把一个 unit 走完，停在下一个 unit 的 spoken 首步前。"""
    return [_id(cap, K.LISTEN), _id(cap, K.AI_SPOKEN_SLOT_END), _id(cap, K.AI_NON_SPOKEN_SLOT_START),
            _id(cap, K.NO_ACTION), _id(cap, K.AI_NON_SPOKEN_SLOT_END), _id(cap, K.UNIT_END),
            _id(cap, K.UNIT_START), _id(cap, K.AI_SPOKEN_SLOT_START)]


def _logits_favoring(size: int, token_id: int) -> torch.Tensor:
    logits = torch.zeros(1, size)
    logits[0, token_id] = 100.0
    return logits


def test_spoken_first_step_masks_everything_but_listen_speak(cap, vocab_size):
    cd = _ConstrainedDecoding(cap, "legacy")
    ids = _prefix(cap)
    ordinary = cap._sdk_tokenizer.encode_ordinary("spoken")[0]
    forced, masked = cd.step(ids, _logits_favoring(vocab_size, ordinary))
    assert forced is None
    assert int(torch.argmax(masked[0])) in {_id(cap, K.LISTEN), _id(cap, K.SPEAK)}
    finite = torch.isfinite(masked[0]).nonzero().flatten().tolist()
    assert sorted(finite) == sorted([_id(cap, K.LISTEN), _id(cap, K.SPEAK)])


def test_blocks_ordinary_before_opener(cap, vocab_size):
    """实测违例 ordinary_before_opener：non-spoken 首步模型想吐普通词（实为想 listen 采偏了）。"""
    cd = _ConstrainedDecoding(cap, "legacy")
    ids = _prefix(cap) + [_id(cap, K.LISTEN), _id(cap, K.AI_SPOKEN_SLOT_END), _id(cap, K.AI_NON_SPOKEN_SLOT_START)]
    ordinary = cap._sdk_tokenizer.encode_ordinary("spoken")[0]
    forced, masked = cd.step(ids, _logits_favoring(vocab_size, ordinary))
    assert forced is None
    assert int(torch.argmax(masked[0])) in {_id(cap, K.NO_ACTION), _id(cap, K.THINK_START), _id(cap, K.TOOL_CALL_START)}


def test_blocks_reopening_think_while_continued_think_open(cap, vocab_size):
    """实测主导违例 stream_opened_before_close（think_16 -> think）。"""
    cd = _ConstrainedDecoding(cap, "legacy")
    word = cap._sdk_tokenizer.encode_ordinary("hmm")[0]
    ids = (_prefix(cap) + [_id(cap, K.LISTEN), _id(cap, K.AI_SPOKEN_SLOT_END), _id(cap, K.AI_NON_SPOKEN_SLOT_START),
                           _id(cap, K.THINK_START), word, _id(cap, K.NON_SPOKEN_BUDGET_REACHED),
                           _id(cap, K.AI_NON_SPOKEN_SLOT_END), _id(cap, K.UNIT_END),
                           _id(cap, K.UNIT_START), _id(cap, K.AI_SPOKEN_SLOT_START), _id(cap, K.LISTEN),
                           _id(cap, K.AI_SPOKEN_SLOT_END), _id(cap, K.AI_NON_SPOKEN_SLOT_START)])
    forced, masked = cd.step(ids, _logits_favoring(vocab_size, _id(cap, K.THINK_START)))
    assert forced is None
    top = int(torch.argmax(masked[0]))
    assert top != _id(cap, K.THINK_START)
    assert torch.isfinite(masked[0, _id(cap, K.THINK_END)])


def test_unique_token_is_forced_without_sampling(cap, vocab_size):
    """revised 下 turn 开着时 spoken 首步只能 speak：直接返回，不看 logits。"""
    cd = _ConstrainedDecoding(cap, "revised")
    word = cap._sdk_tokenizer.encode_ordinary("hi")[0]
    ids = (_prefix(cap) + [_id(cap, K.SPEAK), word, _id(cap, K.SPOKEN_SLOT_EOS), _id(cap, K.AI_SPOKEN_SLOT_END),
                           _id(cap, K.AI_NON_SPOKEN_SLOT_START), _id(cap, K.NO_ACTION),
                           _id(cap, K.AI_NON_SPOKEN_SLOT_END), _id(cap, K.UNIT_END),
                           _id(cap, K.UNIT_START), _id(cap, K.AI_SPOKEN_SLOT_START)])
    forced, _ = cd.step(ids, _logits_favoring(vocab_size, _id(cap, K.LISTEN)))
    assert forced == _id(cap, K.SPEAK)
    assert cd.report()["forced_steps"] == 1


def test_desync_raises_infra_error_with_context(cap, vocab_size):
    """推理侧写了状态机不认的东西 = 框架 bug：抛 ConstrainedDecodingError（服务端按 backend_error 关、
    评测侧重试），不再悄悄退回不约束（用户 2026-10-07 定）。诊断要能进 session.closed 帧。"""
    import json

    from core.fc_duplex.protocol_errors import ConstrainedDecodingError

    cd = _ConstrainedDecoding(cap, "legacy")
    word = cap._sdk_tokenizer.encode_ordinary("hi")[0]
    ids = _prefix(cap) + [_id(cap, K.SPEAK), word, _id(cap, K.AI_SPOKEN_SLOT_END)]   # 撞 max_tokens 直接关 slot
    with pytest.raises(ConstrainedDecodingError) as ei:
        cd.step(ids, _logits_favoring(vocab_size, word))
    diag = ei.value.diagnostic()
    assert diag["infra_kind"] == "constrained_decoding_desync"
    assert "AI_SPOKEN_SLOT_END" in diag["detail"]
    assert diag["context"]["tail_token_ids"] and diag["context"]["fsm_snapshot"]
    json.dumps(diag, ensure_ascii=False)


def test_mask_is_cached_and_cheap(cap, vocab_size):
    """同一合法集合只建一次 mask；之后每步只是一次加法。"""
    cd = _ConstrainedDecoding(cap, "legacy")
    ids = _prefix(cap)
    logits = torch.zeros(1, vocab_size)
    cd.step(ids, logits)                       # 第一次建 mask（含一次性的普通 token 表）
    t0 = time.perf_counter()
    for _ in range(200):
        cd.symbolizer.pos = len(ids)           # 不推进状态，只测每步开销
        cd.step(ids, logits)
    per_step_ms = (time.perf_counter() - t0) / 200 * 1000
    assert len(cd._masks) == 1
    assert per_step_ms < 2.0, f"每步 {per_step_ms:.3f} ms"
