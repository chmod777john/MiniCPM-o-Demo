"""状态机符号 ⇄ 真实 token id。只依赖 ``minicpm_o5_sdk``。

- 推理侧：:meth:`O5ProtocolVocab.allowed_ids` 把 :class:`StepOptions` 翻成「普通词表放不放行 + 哪些
  special id 放行」，调用方据此建 logits mask（``StepOptions`` 可哈希，按它缓存 mask 即可）。
- 验证侧：:meth:`symbols_from_token_ids` 把一段完整 token 序列折成状态机符号（prefill 段压成一个符号）；
  :meth:`token_ids_from_symbols` 反过来，把状态机生成的符号序列展开成 token id 交给 SDK parser。
"""

from __future__ import annotations

from typing import Iterable, Optional

from minicpm_o5_sdk.protocols.duplex.special_tokens import O5SpecialTokenKey as K
from minicpm_o5_sdk.protocols.duplex.special_tokens import O5SpecialTokenRegistry

from .fsm import Abstract, StepOptions, Symbol


class UnknownToken:
    """不属于 duplex 协议的 control token（如 ``<|endoftext|>``）。状态机里永远非法。"""

    def __init__(self, token_id: int) -> None:
        self.token_id = token_id
        self.name = f"UNKNOWN_CONTROL({token_id})"

    def __repr__(self) -> str:
        return self.name


class O5ProtocolVocab:
    def __init__(self, tokenizer, registry: O5SpecialTokenRegistry | None = None) -> None:
        self.tokenizer = tokenizer
        self.registry = registry or O5SpecialTokenRegistry.from_tokenizer(tokenizer)
        self.key_to_id: dict[K, int] = {k: t.token_id for k, t in self.registry.tokens.items()}
        self.id_to_key: dict[int, K] = {v: k for k, v in self.key_to_id.items()}
        self.display: dict[K, str] = {k: t.display_name for k, t in self.registry.tokens.items()}

    @classmethod
    def builtin(cls) -> "O5ProtocolVocab":
        from minicpm_o5_sdk.tokenizers import load_builtin_tokenizer
        from minicpm_o5_sdk.tokenizers.base import O5TokenizerID

        return cls(load_builtin_tokenizer(O5TokenizerID.O5))

    # ------------------------------------------------------------------ id → 符号

    def is_ordinary(self, token_id: int) -> bool:
        return bool(self.tokenizer.is_ordinary_token_id(int(token_id)))

    def symbol_of(self, token_id: int) -> Symbol | UnknownToken:
        token_id = int(token_id)
        if token_id in self.id_to_key:
            return self.id_to_key[token_id]
        if self.is_ordinary(token_id):
            return Abstract.ORDINARY
        return UnknownToken(token_id)

    def allowed_ids(self, options: StepOptions) -> tuple[bool, tuple[int, ...]]:
        """(普通词表是否整体放行, 放行的 special id)。其余 id —— 含所有非协议 control —— 一律 ban。"""
        allow_ordinary = Abstract.ORDINARY in options.model
        ids = tuple(sorted(self.key_to_id[s] for s in options.model if isinstance(s, K)))
        return allow_ordinary, ids

    def symbols_from_token_ids(self, token_ids: Iterable[int]) -> list[Symbol | UnknownToken]:
        """完整序列 → 状态机符号。首个 ``<unit>`` 之前整段 = SYSTEM_PREFILL；
        每个 unit 从 ``<unit>`` 之后到 ``<ai_spoken_slot>`` 之前整段（可以为空）= UNIT_PREFILL。"""
        out: list[Symbol | UnknownToken] = []
        unit_start, spoken_start = self.key_to_id[K.UNIT_START], self.key_to_id[K.AI_SPOKEN_SLOT_START]
        ids = [int(t) for t in token_ids]
        i = 0
        if ids and ids[0] != unit_start:
            out.append(Abstract.SYSTEM_PREFILL)
            while i < len(ids) and ids[i] != unit_start:
                i += 1
        while i < len(ids):
            t = ids[i]
            out.append(self.symbol_of(t))
            i += 1
            if t == unit_start:
                out.append(Abstract.UNIT_PREFILL)
                while i < len(ids) and ids[i] != spoken_start:
                    i += 1
        return out

    # ------------------------------------------------------------------ 符号 → id

    def token_ids_from_symbols(
        self,
        symbols: Iterable[Symbol],
        *,
        ordinary_ids: Optional[list[int]] = None,
        system_ids: Optional[list[int]] = None,
        unit_prefill_ids: Optional[list[int]] = None,
    ) -> list[int]:
        """状态机符号 → token id（给 SDK parser 校验用）。ORDINARY 轮流取 ``ordinary_ids``。"""
        ordinary_ids = ordinary_ids or self.tokenizer.encode_ordinary("hello world")
        system_ids = system_ids if system_ids is not None else (
            [self.key_to_id[K.IM_START]] + self.tokenizer.encode_ordinary("system\nhi") + [self.key_to_id[K.IM_END]]
        )
        unit_prefill_ids = unit_prefill_ids if unit_prefill_ids is not None else []
        out: list[int] = []
        n_ord = 0
        for sym in symbols:
            if sym == Abstract.ORDINARY:
                out.append(ordinary_ids[n_ord % len(ordinary_ids)])
                n_ord += 1
            elif sym == Abstract.SYSTEM_PREFILL:
                out.extend(system_ids)
            elif sym == Abstract.UNIT_PREFILL:
                out.extend(unit_prefill_ids)
            elif sym == Abstract.SESSION_END:
                continue
            else:
                out.append(self.key_to_id[sym])
        return out

    def label(self, sym: object) -> str:
        if isinstance(sym, K):
            return self.display.get(sym, sym.name)
        return {
            Abstract.ORDINARY: "‹普通 token›",
            Abstract.SYSTEM_PREFILL: "⟨system prefill⟩",
            Abstract.UNIT_PREFILL: "⟨unit prefill⟩",
            Abstract.SESSION_END: "⟨session end⟩",
        }.get(sym, getattr(sym, "name", str(sym)))


class IncrementalSymbolizer:
    """把一条**不断追加**的 token id 序列，增量翻成状态机符号并推进状态机。

    推理侧所有写入（system 预填、prefill、模型采样、框架写的 slot / unit 边界、budget）都会进
    ``output_ids``，所以每次解码前调用 :meth:`sync` 把新增部分喂进状态机，状态机就始终和
    真实写进 KV 的序列一致。prefill 压缩规则与 :meth:`O5ProtocolVocab.symbols_from_token_ids` 相同。
    """

    def __init__(self, vocab: O5ProtocolVocab, fsm) -> None:
        self.vocab = vocab
        self.fsm = fsm
        self.pos = 0
        self._unit_start = vocab.key_to_id[K.UNIT_START]
        self._spoken_start = vocab.key_to_id[K.AI_SPOKEN_SLOT_START]
        self._system_done = False
        self._in_unit_prefill = False

    def sync(self, output_ids: list[int]) -> None:
        """喂入 ``output_ids[self.pos:]``。状态机拒绝时抛 O5ProtocolViolation，pos 停在出错处。"""
        n = len(output_ids)
        while self.pos < n:
            t = int(output_ids[self.pos])
            if not self._system_done:
                if t != self._unit_start:
                    self.pos += 1                      # system 预填段整段压成一个符号
                    continue
                self.fsm.advance(Abstract.SYSTEM_PREFILL)
                self._system_done = True
            if self._in_unit_prefill:
                if t != self._spoken_start:
                    self.pos += 1                      # user_video / user_audio / input_event 整段
                    continue
                self.fsm.advance(Abstract.UNIT_PREFILL)
                self._in_unit_prefill = False
            self.fsm.advance(self.vocab.symbol_of(t))
            self.pos += 1
            if t == self._unit_start:
                self._in_unit_prefill = True
