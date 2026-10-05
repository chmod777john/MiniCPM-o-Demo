"""O5 双工 FC 协议状态机：序列走到任一位置时，下一步允许哪些 token。

只依赖 ``minicpm_o5_sdk``（取 ``O5SpecialTokenKey``）。推理侧两种 ``decoding_policy`` 共用这一份定义：

- ``constrained_decoding``：用 :meth:`O5ProtocolFsm.options` 构造 logits mask，模型采不到非法 token；
- ``raise_protocol_error``：不干预采样，用 :meth:`O5ProtocolFsm.advance` 校验，非法即抛
  :class:`O5ProtocolViolation`。

两种 policy 读同一张表，所以「mask 放行了、校验却拒」这种不一致在定义上不可能发生。

**合法 = 训练里出现过的形态**（权威是 SDK 序列化器 ``tokenization/slots/*.py`` 及其规范 case），
不是 SDK parser 能解析的形态：parser 在几处比训练宽（接受 non-spoken 裸文本、turn 中途 listen 等），
又在一处比训练严（拒绝同一 slot 内多段 wrapper，连 SDK 自家规范 ``case_3_03`` 都会被它拒）。

抽象（状态机只管协议合法性，不管语义）：

- 普通 token 不区分具体 id，统一记作 :attr:`Abstract.ORDINARY`；
- system prompt 整段记作 :attr:`Abstract.SYSTEM_PREFILL`；每个 unit 的 user_video / user_audio /
  input_event 整段记作 :attr:`Abstract.UNIT_PREFILL`（内部不涉及状态切换，整段由框架 prefill）；
- ``NON_SPOKEN_BUDGET_REACHED`` 由推理框架写入。状态机只回答「此处框架可以写它」，
  什么时候真的写由调用方决定（推理框架按 budget 计数；验证页面里由人点）。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import FrozenSet, Iterable, Literal, Optional, Union

from minicpm_o5_sdk.protocols.duplex.special_tokens import O5SpecialTokenKey as K


class Abstract(str, Enum):
    """状态机里不对应单个协议 special token 的符号。"""

    ORDINARY = "ORDINARY"
    SYSTEM_PREFILL = "SYSTEM_PREFILL"
    UNIT_PREFILL = "UNIT_PREFILL"
    SESSION_END = "SESSION_END"


Symbol = Union[K, Abstract]


class Actor(str, Enum):
    """一个 token 由谁产生。"""

    MODEL = "model"
    FRAMEWORK = "framework"


class Track(str, Enum):
    """当前这一步属于哪段。``frame`` = 框架写的骨架步（unit / slot 边界、prefill）。"""

    FRAME = "frame"
    SPOKEN = "spoken"
    NON_SPOKEN = "non_spoken"


class Phase(str, Enum):
    BEFORE_SYSTEM = "before_system"
    BETWEEN_UNITS = "between_units"
    UNIT_OPEN = "unit_open"                  # 已写 <unit>，等 UNIT_PREFILL
    PREFILLED = "prefilled"                  # 等 <ai_spoken_slot>
    SPOKEN = "spoken"                        # 模型在 spoken slot 内解码
    SPOKEN_DONE = "spoken_done"              # 等 </ai_spoken_slot>
    SPOKEN_CLOSED = "spoken_closed"          # 等 <ai_non_spoken_slot>
    NON_SPOKEN = "non_spoken"                # 模型在 non-spoken slot 内解码
    NON_SPOKEN_DONE = "non_spoken_done"      # 等 </ai_non_spoken_slot>
    NON_SPOKEN_CLOSED = "non_spoken_closed"  # 等 </unit>
    ENDED = "ended"


class SpokenStep(str, Enum):
    FIRST = "first"                  # slot 首步
    SPEAK_NO_TEXT = "speak_no_text"  # 刚出 <|speak|>，还没正文
    SPEAK_TEXT = "speak_text"        # <|speak|> 之后已有正文
    AFTER_TURN_EOS = "after_turn_eos"
    AFTER_PAD = "after_pad"          # 刚出 <|tts_pad|>（speak_first 形态；只能再出 slot_eos）


Wrapper = Optional[str]  # None / "think" / "tool_call"

_OPENERS = {K.THINK_START: "think", K.TOOL_CALL_START: "tool_call"}
_CLOSERS = {"think": K.THINK_END, "tool_call": K.TOOL_CALL_END}

# 框架在每个骨架阶段写的唯一 token
_FRAME_NEXT = {
    Phase.BEFORE_SYSTEM: Abstract.SYSTEM_PREFILL,
    Phase.BETWEEN_UNITS: K.UNIT_START,
    Phase.UNIT_OPEN: Abstract.UNIT_PREFILL,
    Phase.PREFILLED: K.AI_SPOKEN_SLOT_START,
    Phase.SPOKEN_DONE: K.AI_SPOKEN_SLOT_END,
    Phase.SPOKEN_CLOSED: K.AI_NON_SPOKEN_SLOT_START,
    Phase.NON_SPOKEN_DONE: K.AI_NON_SPOKEN_SLOT_END,
    Phase.NON_SPOKEN_CLOSED: K.UNIT_END,
}


@dataclass(frozen=True)
class FsmConfig:
    """协议上尚待拍板、因此做成开关的规则。

    allow_multi_wrapper_per_slot: 同一 non-spoken slot 内一段 wrapper 闭合后能否再开一段
        （``<think>…</think><tool_call>…</tool_call>``）。SDK 序列化器与规范 case 都会产出
        （case_3_03 并行查询、no-think 模式的「空 think + 动作」），但 SDK parser 不接受，
        当前推理服务也会把它报成 ``aggregate_kind_changed``。默认按训练形态允许。
    pad_form: ``<|tts_pad|>`` 格的形态，**取决于 ckpt 是用哪版 SDK 的数据训的**：

        - ``speak_first``（默认，SDK **≥ 0.0.6a4**）：``<|speak|> <|tts_pad|> <|spoken_slot_eos|>``。
          slot 首步恒为 ``listen`` / ``speak`` 二选一；pad 是 speak 之后与正文并列的内容分支，
          且**不允许出现在 turn 的第一个 unit**（那等于开口却整段不发声）。
        - ``bare``（SDK ≤ 0.0.6a3 训出来的存量 ckpt）：``<|tts_pad|>`` 独占整个 slot，
          前面没有 ``<|speak|>``，出了就直接关 slot。

        🔴 **给存量 ckpt 跑约束解码必须用 ``bare``**。实测 A iter8000 的 tau 录制里，
        40 个会话有 21 个出现模型自己吐的 ``response.spoken.end reason=tts_pad``，
        单个会话最多 21 次 —— 拿 ``speak_first`` 去卡它们会大面积误判协议错误。
    """

    allow_multi_wrapper_per_slot: bool = True
    pad_form: Literal["speak_first", "bare"] = "speak_first"


@dataclass(frozen=True)
class StepOptions:
    """下一步的合法集合。``model`` 可能含 :attr:`Abstract.ORDINARY`（= 整个普通词表）。"""

    track: Track
    model: FrozenSet[Symbol]
    framework: FrozenSet[Symbol]

    @property
    def all(self) -> FrozenSet[Symbol]:
        return self.model | self.framework

    @property
    def forced(self) -> Optional[tuple[Symbol, Actor]]:
        """合法集合只有一个具体 token 时返回 (它, 由谁产生)，否则 None。

        唯一选项是 ORDINARY 时也返回 None：它是整个普通词表，具体哪个 token 仍要模型去选。
        """
        if len(self.model) + len(self.framework) != 1:
            return None
        if self.model:
            sym = next(iter(self.model))
            return None if sym == Abstract.ORDINARY else (sym, Actor.MODEL)
        return next(iter(self.framework)), Actor.FRAMEWORK


class O5ProtocolViolation(ValueError):
    """采到 / 写入了当前状态下不合法的符号。``code`` 与推理服务的违例类型对齐（能对上时）。"""

    def __init__(self, code: str, message: str, *, symbol: object, options: StepOptions, state: dict):
        super().__init__(message)
        self.code = code
        self.symbol = symbol
        self.options = options
        self.state = state


@dataclass
class _SlotLocal:
    """non-spoken slot 内的局部状态（slot 开始时重置）。"""

    first: bool = True
    cur: Wrapper = None          # 本 slot 内当前打开的 wrapper（含从上个 unit 延续来的）
    closed: int = 0              # 本 slot 已闭合的 wrapper 数
    n_decode: int = 0            # 本 slot 已产生的模型 token 数


@dataclass
class _State:
    phase: Phase = Phase.BEFORE_SYSTEM
    unit_index: int = -1
    # —— 跨 unit 状态 ——
    turn_open: bool = False      # T：spoken turn 已被 <|speak|> 打开、还没 <|spoken_turn_eos|>
    wrapper: Wrapper = None      # W：non-spoken wrapper 被 budget 截断、延续到下一个 unit
    wrapper_payload: int = 0     # W 已累计的正文 token 数（跨 unit 计，用于禁止空 tool_call）
    eos_owed: bool = False       # E：上个 unit 在 wrapper 刚闭合时被 budget 截断，EOS 欠到本 unit
    # —— slot 局部 ——
    spoken: SpokenStep = SpokenStep.FIRST
    turn_open_at_slot_start: bool = False   # 进入本 spoken slot 之前 turn 是否已经开着
                                            # （speak_first 下用它禁止 turn 首个 unit 出 pad）
    ns: _SlotLocal = field(default_factory=_SlotLocal)


class O5ProtocolFsm:
    """O5 双工 FC 协议状态机。一个 Session 一个实例；:meth:`advance` 逐符号推进。"""

    def __init__(self, config: FsmConfig | None = None) -> None:
        self.config = config or FsmConfig()
        self._s = _State()

    # ------------------------------------------------------------------ 查询

    def options(self) -> StepOptions:
        """当前位置下一步的合法集合。纯函数：同一状态反复调用结果相同。"""
        s = self._s
        if s.phase in _FRAME_NEXT:
            return StepOptions(Track.FRAME, frozenset(), frozenset({_FRAME_NEXT[s.phase]}))
        if s.phase == Phase.SPOKEN:
            return StepOptions(Track.SPOKEN, self._spoken_model_options(), frozenset())
        if s.phase == Phase.NON_SPOKEN:
            budget = frozenset({K.NON_SPOKEN_BUDGET_REACHED}) if s.ns.n_decode >= 1 else frozenset()
            return StepOptions(Track.NON_SPOKEN, self._non_spoken_model_options(), budget)
        return StepOptions(Track.FRAME, frozenset(), frozenset())  # ENDED

    def can_end_session(self) -> bool:
        """Session 能否在此结束：只能落在 unit 之间，且没有被 budget 截断、仍未闭合的 wrapper。"""
        return self._s.phase == Phase.BETWEEN_UNITS and self._s.wrapper is None and self._s.unit_index >= 0

    def snapshot(self) -> dict:
        """当前状态（JSON 可序列化），给日志 / 验证页面看。"""
        s = self._s
        d = {
            "phase": s.phase.value,
            "unit_index": s.unit_index,
            "turn_open": s.turn_open,
            "wrapper": s.wrapper,
            "wrapper_payload": s.wrapper_payload,
            "eos_owed": s.eos_owed,
            "pad_form": self.config.pad_form,
        }
        if s.phase == Phase.SPOKEN:
            d["turn_open_at_slot_start"] = s.turn_open_at_slot_start
        if s.phase == Phase.SPOKEN:
            d["spoken_step"] = s.spoken.value
        if s.phase == Phase.NON_SPOKEN:
            d["slot"] = asdict(s.ns)
        d["label"] = self.state_label()
        return d

    def state_label(self) -> str:
        """当前状态的一句人话解释（为什么下一步只能是这些）。"""
        s = self._s
        if s.phase == Phase.SPOKEN:
            bare = self.config.pad_form == "bare"
            if s.spoken == SpokenStep.FIRST:
                if s.turn_open:
                    if bare:
                        return "spoken 首步 · turn 已打开未结束：继续说（speak），或本 unit 无新正文（tts_pad）；不能 listen"
                    return "spoken 首步 · turn 已打开未结束：只能 speak（pad 要排在 speak 之后）；不能 listen"
                return "spoken 首步 · 没有打开的 turn：listen 或 speak 二选一"
            if s.spoken == SpokenStep.SPEAK_NO_TEXT:
                if not bare and s.turn_open_at_slot_start:
                    return ("刚出 speak、还没正文：写正文 / 本 unit 无新正文（tts_pad）/ 结束本 turn；"
                            "不能空着关 slot")
                if not bare:
                    return ("刚出 speak、还没正文：写正文，或直接结束本 turn；"
                            "**本 unit 是这个 turn 的第一格，不能发 tts_pad**（开口却整段不发声）")
                return "刚出 speak、还没正文：写正文，或直接结束本 turn（spoken_turn_eos）；不能空着关 slot"
            if s.spoken == SpokenStep.SPEAK_TEXT:
                return "speak 正文中：继续写正文 / 结束本 turn / 只结束本 unit 的 slot（turn 延续到下个 unit）"
            if s.spoken == SpokenStep.AFTER_PAD:
                return "刚出 tts_pad：只能 spoken_slot_eos（pad 与 turn_eos 永不同格）"
            return "刚出 spoken_turn_eos：只能 spoken_slot_eos"
        if s.phase == Phase.NON_SPOKEN:
            ns = s.ns
            if ns.cur is not None:
                how = "从上个 unit 延续来的" if ns.first and s.wrapper else "打开的"
                return f"non-spoken · {how} {ns.cur} 中：写正文或闭合它；不能开新流、不能 no_action / eos"
            if ns.first and s.eos_owed:
                return "non-spoken 首步 · 上个 unit 欠一个 EOS（wrapper 刚闭合就被 budget 截断）：补 non_spoken_eos" + (
                    "，或开下一段 wrapper" if self.config.allow_multi_wrapper_per_slot else "")
            if ns.first:
                return "non-spoken 首步 · 没有打开的流：no_action / 开 think / 开 tool_call；不能直接写普通文本"
            return "non-spoken · wrapper 刚闭合：结束本 slot（non_spoken_eos）" + (
                "，或再开一段 wrapper" if self.config.allow_multi_wrapper_per_slot else "")
        return {
            Phase.BEFORE_SYSTEM: "Session 开始：框架 prefill system",
            Phase.BETWEEN_UNITS: "unit 之间：框架开下一个 unit（或结束 Session）",
            Phase.UNIT_OPEN: "框架 prefill 本 unit 的 user_video / user_audio / input_event",
            Phase.PREFILLED: "框架打开 spoken slot",
            Phase.SPOKEN_DONE: "框架关闭 spoken slot",
            Phase.SPOKEN_CLOSED: "框架打开 non-spoken slot",
            Phase.NON_SPOKEN_DONE: "框架关闭 non-spoken slot",
            Phase.NON_SPOKEN_CLOSED: "框架关闭 unit",
            Phase.ENDED: "Session 已结束",
        }[s.phase]

    # ------------------------------------------------------------------ 推进

    def advance(self, symbol: object, actor: Actor | None = None) -> None:
        """推进一个符号。不合法时抛 :class:`O5ProtocolViolation`，状态不变。

        actor: 给出时额外要求该符号由这一方产生（模型不能写框架的 token，反之亦然）；
            None 表示不区分（例如回放一段 token 序列时）。
        """
        opts = self.options()
        if symbol == Abstract.SESSION_END:
            if not self.can_end_session():
                raise self._violation("session_end_not_allowed", symbol, opts,
                                      "此处不能结束 Session（不在 unit 之间，或仍有未闭合的 wrapper）")
            self._s.phase = Phase.ENDED
            return
        pool = opts.all if actor is None else (opts.model if actor == Actor.MODEL else opts.framework)
        if symbol not in pool:
            raise self._violation(self._classify(symbol, opts, actor), symbol, opts)
        s = self._s
        if s.phase in _FRAME_NEXT:
            self._advance_frame(symbol)
        elif s.phase == Phase.SPOKEN:
            self._advance_spoken(symbol)
        else:
            self._advance_non_spoken(symbol)

    def feed(self, symbols: Iterable[object]) -> None:
        for sym in symbols:
            self.advance(sym)

    # ------------------------------------------------------------------ 规则

    def _spoken_model_options(self) -> FrozenSet[Symbol]:
        s = self._s
        bare = self.config.pad_form == "bare"
        if s.spoken == SpokenStep.FIRST:
            if bare:
                # 存量形态：pad 独占 slot，所以它和 listen/speak 一样是首步的决策之一；
                # turn 开着时不能 listen（listen 会让未收尾的 turn 断掉）
                return frozenset({K.SPEAK, K.TTS_PAD}) if s.turn_open else frozenset({K.LISTEN, K.SPEAK})
            # speak_first：首步恒为二选一，pad 推迟到 speak 之后
            return frozenset({K.SPEAK} if s.turn_open else {K.LISTEN, K.SPEAK})
        if s.spoken == SpokenStep.SPEAK_NO_TEXT:
            # 训练里 speak 之后要么有正文，要么是 turn 末尾的 terminal-only（speak + turn_eos）
            base = {Abstract.ORDINARY, K.SPOKEN_TURN_EOS}
            # speak_first 下 pad 是这里的第三个分支，但**不允许出现在 turn 的第一个 unit**：
            # 那等于开口却整段不发声，不如下一个 unit 再开口。
            if not bare and s.turn_open_at_slot_start:
                base.add(K.TTS_PAD)
            return frozenset(base)
        if s.spoken == SpokenStep.SPEAK_TEXT:
            return frozenset({Abstract.ORDINARY, K.SPOKEN_TURN_EOS, K.SPOKEN_SLOT_EOS})
        # AFTER_TURN_EOS / AFTER_PAD 都只能关 slot
        return frozenset({K.SPOKEN_SLOT_EOS})

    def _non_spoken_model_options(self) -> FrozenSet[Symbol]:
        s, ns = self._s, self._s.ns
        openers = frozenset(_OPENERS) if self.config.allow_multi_wrapper_per_slot else frozenset()
        if ns.cur is not None:
            if ns.cur == "tool_call" and s.wrapper_payload == 0:
                return frozenset({Abstract.ORDINARY})  # 训练里没有空 tool_call
            return frozenset({Abstract.ORDINARY, _CLOSERS[ns.cur]})  # 空 think 合法（no-think 模式的空操作）
        if ns.first:
            if s.eos_owed:
                return frozenset({K.NON_SPOKEN_EOS}) | openers
            return frozenset({K.NO_ACTION, K.THINK_START, K.TOOL_CALL_START})
        return frozenset({K.NON_SPOKEN_EOS}) | openers  # 本 slot 已闭合过 wrapper

    def _advance_frame(self, symbol: object) -> None:
        s = self._s
        if symbol == Abstract.SYSTEM_PREFILL:
            s.phase = Phase.BETWEEN_UNITS
        elif symbol == K.UNIT_START:
            s.unit_index += 1
            s.phase = Phase.UNIT_OPEN
        elif symbol == Abstract.UNIT_PREFILL:
            s.phase = Phase.PREFILLED
        elif symbol == K.AI_SPOKEN_SLOT_START:
            s.spoken = SpokenStep.FIRST
            s.turn_open_at_slot_start = s.turn_open
            s.phase = Phase.SPOKEN
        elif symbol == K.AI_SPOKEN_SLOT_END:
            s.phase = Phase.SPOKEN_CLOSED
        elif symbol == K.AI_NON_SPOKEN_SLOT_START:
            s.ns = _SlotLocal(cur=s.wrapper)
            s.phase = Phase.NON_SPOKEN
        elif symbol == K.AI_NON_SPOKEN_SLOT_END:
            s.phase = Phase.NON_SPOKEN_CLOSED
        elif symbol == K.UNIT_END:
            s.phase = Phase.BETWEEN_UNITS

    def _advance_spoken(self, symbol: object) -> None:
        s = self._s
        if symbol == K.TTS_PAD and self.config.pad_form != "bare":
            s.spoken = SpokenStep.AFTER_PAD      # speak_first：pad 之后还要一个 slot_eos
        elif symbol in (K.LISTEN, K.TTS_PAD, K.SPOKEN_SLOT_EOS):
            s.phase = Phase.SPOKEN_DONE
        elif symbol == K.SPEAK:
            s.turn_open = True
            s.spoken = SpokenStep.SPEAK_NO_TEXT
        elif symbol == Abstract.ORDINARY:
            s.spoken = SpokenStep.SPEAK_TEXT
        elif symbol == K.SPOKEN_TURN_EOS:
            s.turn_open = False
            s.spoken = SpokenStep.AFTER_TURN_EOS

    def _advance_non_spoken(self, symbol: object) -> None:
        s, ns = self._s, self._s.ns
        if symbol == K.NON_SPOKEN_BUDGET_REACHED:
            s.wrapper = ns.cur                     # 截在 wrapper 中间 → 下个 unit 续写
            s.eos_owed = ns.cur is None            # 截在 wrapper 刚闭合处 → 下个 unit 补 EOS
            if ns.cur is None:
                s.wrapper_payload = 0
            s.phase = Phase.NON_SPOKEN_DONE
            return
        ns.first = False
        ns.n_decode += 1
        if symbol in (K.NO_ACTION, K.NON_SPOKEN_EOS):
            s.wrapper, s.wrapper_payload, s.eos_owed = None, 0, False
            s.phase = Phase.NON_SPOKEN_DONE
        elif symbol in _OPENERS:
            ns.cur = _OPENERS[symbol]
            s.wrapper_payload = 0
            s.eos_owed = False
        elif symbol in (K.THINK_END, K.TOOL_CALL_END):
            ns.cur = None
            ns.closed += 1
            s.wrapper, s.wrapper_payload, s.eos_owed = None, 0, False
        elif symbol == Abstract.ORDINARY:
            s.wrapper_payload += 1

    # ------------------------------------------------------------------ 违例分类

    def _classify(self, symbol: object, opts: StepOptions, actor: Actor | None) -> str:
        s = self._s
        if actor == Actor.MODEL and symbol in opts.framework:
            return "framework_token_by_model"
        if actor == Actor.FRAMEWORK and symbol in opts.model:
            return "model_token_by_framework"
        if not isinstance(symbol, (K, Abstract)):
            return "special_without_semantic_key"
        if s.phase == Phase.SPOKEN:
            if symbol == K.LISTEN and s.turn_open:
                return "listen_during_open_turn"
            if symbol == K.TTS_PAD:
                if self.config.pad_form == "bare":
                    if not s.turn_open:
                        return "tts_pad_without_open_turn"
                else:
                    if s.spoken == SpokenStep.FIRST:
                        return "tts_pad_without_speak"          # 新形态下 pad 必须排在 speak 之后
                    if not s.turn_open_at_slot_start:
                        return "tts_pad_at_turn_start"          # turn 第一格不许 pad
        if s.phase == Phase.NON_SPOKEN:
            ns = s.ns
            if symbol == Abstract.ORDINARY and ns.cur is None:
                return "ordinary_before_opener"
            if symbol in _OPENERS and ns.cur is not None:
                return "stream_opened_before_close"
            if symbol in _OPENERS and ns.closed and not self.config.allow_multi_wrapper_per_slot:
                return "aggregate_kind_changed"
            if symbol in (K.THINK_END, K.TOOL_CALL_END) and (ns.cur is None or symbol != _CLOSERS[ns.cur]):
                return "end_without_stream"
        return "not_allowed"

    def _violation(self, code: str, symbol: object, opts: StepOptions, message: str | None = None):
        name = getattr(symbol, "name", repr(symbol))
        allowed = sorted(getattr(x, "name", str(x)) for x in opts.all)
        msg = message or f"{code}: {self.state_label()} —— 得到 {name}，允许 {allowed}"
        return O5ProtocolViolation(code, msg, symbol=symbol, options=opts, state=self.snapshot())
