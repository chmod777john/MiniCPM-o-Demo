"""o5_protocol_fsm 的独立验证（只用 SDK，不起模型、不连任何服务）。

两条主验收：
1. 训练形态全接受：SDK 规范 case × 多种 tokenize 配置（含 budget 截断续接、no-think）→ 状态机一个不拒。
2. 生成的都合法：随机游走状态机 → 展开成 token id → SDK parser 全部通过（关掉「同 slot 多段 wrapper」时；
   打开时 SDK parser 自身拒绝 case_3_03 这类规范形态，不能拿它比）。
其余是逐条规则测试，含 tau 实测的两类违例。
"""

from __future__ import annotations

import random

import pytest

import minicpm_o5_sdk as sdk
from minicpm_o5_sdk.protocols.duplex.parsing.models import O5DuplexParseRequest
from minicpm_o5_sdk.protocols.duplex.parsing.parser import O5DuplexParser
from minicpm_o5_sdk.protocols.duplex.special_tokens import O5SpecialTokenKey as K
from minicpm_o5_sdk.tokenizers.base import O5TokenizerID

from o5_protocol_fsm import Abstract, Actor, FsmConfig, O5ProtocolFsm, O5ProtocolViolation, Track
from o5_protocol_fsm.vocab import O5ProtocolVocab, UnknownToken

ORD = Abstract.ORDINARY
BUDGET = K.NON_SPOKEN_BUDGET_REACHED

TOKENIZE_CONFIGS = {
    "default": {},
    "no_think": {"prepend_empty_think_before_action": True},
    "budget1": {"non_spoken_budget_while_listening_override": 1, "non_spoken_budget_while_speaking_override": 1},
    "budget2": {"non_spoken_budget_while_listening_override": 2, "non_spoken_budget_while_speaking_override": 2},
    "budget3": {"non_spoken_budget_while_listening_override": 3, "non_spoken_budget_while_speaking_override": 3},
    "budget3_no_think": {
        "prepend_empty_think_before_action": True,
        "non_spoken_budget_while_listening_override": 3,
        "non_spoken_budget_while_speaking_override": 3,
    },
}


@pytest.fixture(scope="module")
def vocab() -> O5ProtocolVocab:
    return O5ProtocolVocab.builtin()


def _autofill(f: O5ProtocolFsm) -> None:
    while (forced := f.options().forced) is not None:
        f.advance(*forced)


def _drive(f: O5ProtocolFsm, *decisions) -> O5ProtocolFsm:
    """依次做决策；每个决策前后把唯一合法的步骤自动填上。"""
    _autofill(f)
    for d in decisions:
        f.advance(d)
        _autofill(f)
    return f


def _fresh(*decisions, multi: bool = True) -> O5ProtocolFsm:
    return _drive(O5ProtocolFsm(FsmConfig(allow_multi_wrapper_per_slot=multi)), *decisions)


def _code(f: O5ProtocolFsm, sym, actor=None) -> str:
    with pytest.raises(O5ProtocolViolation) as e:
        f.advance(sym, actor)
    return e.value.code


# ---------------------------------------------------------------------- 主验收 1：训练形态全接受


@pytest.mark.parametrize("config_name", list(TOKENIZE_CONFIGS))
def test_sdk_canonical_cases_all_accepted(vocab, config_name):
    n = 0
    for case_id in sdk.list_case_ids():
        for variant in sdk.list_case_variants(case_id):
            td = sdk.load_case(case_id, variant_id=variant.variant_id)
            try:
                ids = td.tokenize(vocab.tokenizer, **TOKENIZE_CONFIGS[config_name]).tokenized_data.input_ids
            except ValueError:
                continue  # no-think 不适用于含 think 的 case（SDK 自己拒绝 tokenize）
            f = O5ProtocolFsm()
            f.feed(vocab.symbols_from_token_ids(ids))
            assert f.can_end_session(), f"{case_id.value}/{variant.variant_id} 结束时状态 {f.snapshot()}"
            n += 1
    assert n >= 10


def _case_with_silence(lead: float = 0.0, trail: float = 0.0, mid_gap: float = 0.0):
    """case_1_01 的 AI 语音改静音分布：开头 / 结尾补零，或在第 8 个字后插停顿；alignment 同步平移。"""
    import torch
    from minicpm_o5_sdk import O5Alignment, O5WordInterval
    from minicpm_o5_sdk.protocols.duplex.training_data.lazy_media import O5LazyAudio

    sr = 16000
    td = sdk.load_case("case_1_01")
    seg = td.tracks.ai_spoken.segments[0]
    wav = seg.audio.get_tensor()
    words = list(seg.alignment.word_intervals)
    cut_t = words[7].end_sec
    if mid_gap:
        wav = torch.cat([wav[: round(cut_t * sr)], torch.zeros(round(mid_gap * sr)), wav[round(cut_t * sr):]])
    wav = torch.cat([torch.zeros(round(lead * sr)), wav, torch.zeros(round(trail * sr))]).float()

    def shift(t: float) -> float:
        return round(t + lead + (mid_gap if mid_gap and t >= cut_t - 1e-9 else 0.0), 4)

    new_seg = seg.model_copy(update={
        "audio": O5LazyAudio(wav.numel() / sr, lambda: wav),
        "alignment": O5Alignment(word_intervals=[
            O5WordInterval(text=w.text, start_sec=shift(w.start_sec), end_sec=shift(w.end_sec)) for w in words]),
    })
    spoken = td.tracks.ai_spoken.model_copy(update={"segments": [new_seg]})
    return td.model_copy(update={"tracks": td.tracks.model_copy(update={"ai_spoken": spoken})})


@pytest.mark.parametrize("kw", [{"mid_gap": 2.5}, {"trail": 2.5}, {"trail": 4.0}])
def test_sdk_tts_pad_forms_accepted(vocab, kw):
    """序列化器产出 tts_pad 的正常场景：turn 中间停顿、turn 末尾静音。
    后者的 turn 以 terminal-only「speak + spoken_turn_eos + spoken_slot_eos」收尾，tts_pad 后从不接 turn_eos。"""
    ids = _case_with_silence(**kw).tokenize(vocab.tokenizer).tokenized_data.input_ids
    syms = vocab.symbols_from_token_ids(ids)
    assert K.TTS_PAD in syms
    for i, s in enumerate(syms):
        if s == K.TTS_PAD:
            assert syms[i + 1] == K.AI_SPOKEN_SLOT_END
    O5ProtocolFsm().feed(syms)


def test_turn_leading_tts_pad_rejected_like_sdk_replay(vocab):
    """AI 语音开头静音 > unit_sec + prelook（1.3s）时，SDK 序列化器会在 turn 开头直接出 tts_pad，
    但 SDK 自己的 replay 拒绝它（「孤立 <|tts_pad|>」）。状态机与 replay 一致：tts_pad 必须先有 speak 打开 turn。"""
    td = _case_with_silence(lead=2.5)
    ids = td.tokenize(vocab.tokenizer).tokenized_data.input_ids
    parser = O5DuplexParser(O5TokenizerID.O5, tokenizer=vocab.tokenizer)
    with pytest.raises(Exception, match="unsupported_replay_pattern"):
        parser.parse(O5DuplexParseRequest(sequence=list(ids), unit_policy=td.unit_policy))
    with pytest.raises(O5ProtocolViolation) as e:
        O5ProtocolFsm().feed(vocab.symbols_from_token_ids(ids))
    assert e.value.code == "tts_pad_without_open_turn"


# ---------------------------------------------------------------------- 主验收 2：生成的都合法


def _random_walk(rng: random.Random, n_units: int, multi: bool, p_budget: float) -> list:
    f = O5ProtocolFsm(FsmConfig(allow_multi_wrapper_per_slot=multi))
    out = []
    while True:
        if f.can_end_session() and f.snapshot()["unit_index"] >= n_units - 1:
            return out
        opts = f.options()
        budget_ok = opts.framework and f.snapshot()["unit_index"] < n_units - 1
        if opts.track == Track.NON_SPOKEN and budget_ok and rng.random() < p_budget:
            sym, actor = BUDGET, Actor.FRAMEWORK
        elif opts.model:
            # 普通 token 给高权重，否则游走到不了「正文中」的状态
            pool = sorted(opts.model, key=lambda s: s.value) + ([ORD] * 3 if ORD in opts.model else [])
            sym, actor = rng.choice(pool), Actor.MODEL
        else:
            (sym,) = opts.framework
            actor = Actor.FRAMEWORK
        f.advance(sym, actor)
        out.append(sym)


def test_random_walks_parse_with_sdk_parser(vocab):
    parser = O5DuplexParser(O5TokenizerID.O5, tokenizer=vocab.tokenizer)
    unit_policy = sdk.load_case(sdk.list_case_ids()[0]).unit_policy
    audio_slot = [vocab.key_to_id[k] for k in (K.USER_AUDIO_SLOT_START, K.AUDIO_PLACEHOLDER, K.USER_AUDIO_SLOT_END)]
    rng = random.Random(20261005)
    for i in range(300):
        syms = _random_walk(rng, n_units=rng.randint(1, 8), multi=False, p_budget=0.25)
        ids = vocab.token_ids_from_symbols(syms, unit_prefill_ids=audio_slot if i % 2 else [])
        parser.validate(O5DuplexParseRequest(sequence=ids, unit_policy=unit_policy))


def test_never_offers_untrained_or_framework_only_tokens():
    never_model = {K.NON_SPOKEN_HOLD, K.NON_SPOKEN_ABORT, BUDGET, K.AI_SPOKEN_SLOT_END,
                   K.AI_NON_SPOKEN_SLOT_END, K.UNIT_END, K.UNIT_START, K.IM_START, K.IM_END}
    rng = random.Random(7)
    for _ in range(200):
        f = O5ProtocolFsm()
        for _step in range(400):
            opts = f.options()
            assert not (opts.model & never_model), (f.snapshot(), opts.model)
            assert K.NON_SPOKEN_HOLD not in opts.framework and K.NON_SPOKEN_ABORT not in opts.framework
            if opts.track == Track.NON_SPOKEN and opts.framework and rng.random() < 0.2:
                f.advance(BUDGET, Actor.FRAMEWORK)
            elif opts.model:
                f.advance(rng.choice(sorted(opts.model, key=lambda s: s.value)), Actor.MODEL)
            else:
                (sym,) = opts.framework
                f.advance(sym, Actor.FRAMEWORK)


# ---------------------------------------------------------------------- spoken 规则


def test_spoken_first_step_listen_or_speak():
    f = _fresh()
    assert f.options().track == Track.SPOKEN
    assert f.options().model == {K.LISTEN, K.SPEAK}
    assert f.options().framework == frozenset()


def test_open_turn_bans_listen_and_allows_tts_pad():
    # unit0: speak + 正文 + slot_eos（turn 延续）→ unit1 spoken 首步
    f = _fresh(K.SPEAK, ORD, K.SPOKEN_SLOT_EOS, K.NO_ACTION)
    assert f.options().model == {K.SPEAK, K.TTS_PAD}
    assert _code(f, K.LISTEN) == "listen_during_open_turn"


def test_tts_pad_requires_open_turn():
    assert _code(_fresh(), K.TTS_PAD) == "tts_pad_without_open_turn"


def test_tts_pad_keeps_turn_open_until_turn_eos():
    f = _fresh(K.SPEAK, ORD, K.SPOKEN_SLOT_EOS, K.NO_ACTION, K.TTS_PAD, K.NO_ACTION)
    assert f.options().model == {K.SPEAK, K.TTS_PAD}
    f = _drive(f, K.SPEAK, K.SPOKEN_TURN_EOS, K.NO_ACTION)  # terminal-only：speak + turn_eos
    assert f.options().model == {K.LISTEN, K.SPEAK}


def test_speak_cannot_close_slot_without_text():
    f = _fresh(K.SPEAK)
    assert f.options().model == {ORD, K.SPOKEN_TURN_EOS}
    assert _code(f, K.SPOKEN_SLOT_EOS) == "not_allowed"


def test_after_turn_eos_only_slot_eos():
    f = _fresh(K.SPEAK, ORD, K.SPOKEN_TURN_EOS)
    # 唯一合法 → 被自动填上，已经进到 non-spoken
    assert f.options().track == Track.NON_SPOKEN


# ---------------------------------------------------------------------- non-spoken 规则


def test_non_spoken_first_step_no_bare_text():
    """tau / FDB 实测的 ordinary_before_opener：本该 listen，采成了语义相近的普通词。"""
    f = _fresh(K.LISTEN)
    assert f.options().model == {K.NO_ACTION, K.THINK_START, K.TOOL_CALL_START}
    assert f.options().framework == frozenset()  # 首步不给 budget（budget ≥ 1）
    assert _code(f, ORD) == "ordinary_before_opener"


def test_continued_think_bans_new_opener():
    """tau 实测的主导违例 think_16 -> think：think 被 budget 截到下个 unit，模型又开 think。"""
    f = _fresh(K.LISTEN, K.THINK_START, ORD, BUDGET, K.LISTEN)
    assert f.options().model == {ORD, K.THINK_END}
    assert _code(f, K.THINK_START) == "stream_opened_before_close"
    assert _code(f, K.NO_ACTION) == "not_allowed"


def test_empty_think_allowed_empty_tool_call_not():
    f = _fresh(K.LISTEN, K.THINK_START)
    assert K.THINK_END in f.options().model
    f = _fresh(K.LISTEN, K.TOOL_CALL_START)
    assert f.options().model == {ORD}


def test_empty_tool_call_banned_across_budget_continuation():
    f = _fresh(K.LISTEN, K.TOOL_CALL_START, BUDGET, K.LISTEN)
    assert f.options().model == {ORD}


def test_multi_wrapper_switch():
    f = _fresh(K.LISTEN, K.THINK_START, ORD, K.THINK_END)
    assert f.options().model == {K.NON_SPOKEN_EOS, K.THINK_START, K.TOOL_CALL_START}
    f = _fresh(K.LISTEN, K.THINK_START, ORD, K.THINK_END, multi=False)
    assert f.options().model == {K.NON_SPOKEN_EOS}
    assert _code(f, K.TOOL_CALL_START) == "aggregate_kind_changed"


def test_eos_owed_after_budget_at_wrapper_close():
    f = _fresh(K.LISTEN, K.THINK_START, ORD, K.THINK_END, BUDGET, K.LISTEN)
    assert f.options().model == {K.NON_SPOKEN_EOS, K.THINK_START, K.TOOL_CALL_START}
    assert _code(f, K.NO_ACTION) == "not_allowed"
    f = _fresh(K.LISTEN, K.THINK_START, ORD, K.THINK_END, BUDGET, K.LISTEN, multi=False)
    assert f.options().track == Track.SPOKEN  # 关掉多段时欠的 EOS 是唯一选项，已被自动补上、进到下个 unit


def test_budget_offered_only_after_first_token():
    f = _fresh(K.LISTEN, K.THINK_START)
    assert f.options().framework == {BUDGET}


def test_framework_tokens_cannot_come_from_model():
    f = _fresh(K.LISTEN, K.THINK_START, ORD)
    assert _code(f, BUDGET, Actor.MODEL) == "framework_token_by_model"


# ---------------------------------------------------------------------- 骨架 / 结束 / 原子性


def test_session_end_only_between_units_without_open_wrapper():
    f = _fresh(K.LISTEN, K.NO_ACTION)
    # 自动填到了下一个 unit 的 spoken 首步；回放一个停在 unit 之间的状态
    g = O5ProtocolFsm()
    for sym in (Abstract.SYSTEM_PREFILL, K.UNIT_START, Abstract.UNIT_PREFILL, K.AI_SPOKEN_SLOT_START, K.LISTEN,
                K.AI_SPOKEN_SLOT_END, K.AI_NON_SPOKEN_SLOT_START, K.THINK_START, ORD, BUDGET,
                K.AI_NON_SPOKEN_SLOT_END, K.UNIT_END):
        g.advance(sym)
    assert not g.can_end_session()  # think 还没闭合
    assert _code(g, Abstract.SESSION_END) == "session_end_not_allowed"
    assert not f.can_end_session()


def test_failed_advance_leaves_state_unchanged():
    f = _fresh(K.LISTEN)
    before = f.snapshot()
    with pytest.raises(O5ProtocolViolation):
        f.advance(ORD)
    assert f.snapshot() == before


def test_unknown_control_token_is_violation(vocab):
    eot = vocab.tokenizer.encode_special("<|endoftext|>") if hasattr(vocab.tokenizer, "encode_special") else None
    sym = UnknownToken(eot[0] if eot else 10**9)
    f = _fresh(K.LISTEN, K.THINK_START)
    assert _code(f, sym) == "special_without_semantic_key"


def test_allowed_ids_never_open_ordinary_where_banned(vocab):
    f = _fresh()
    allow_ordinary, ids = vocab.allowed_ids(f.options())
    assert allow_ordinary is False
    assert set(ids) == {vocab.key_to_id[K.LISTEN], vocab.key_to_id[K.SPEAK]}
