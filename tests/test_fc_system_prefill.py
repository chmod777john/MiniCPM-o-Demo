"""FC Semantic Realtime API v3 system prefill 的纯 CPU 顺序测试。

测试使用精确的 SDK schema，并用最小 fake tokenizer/registry/serializer 隔离模型与
GPU，验证 text/audio/tools 编排和 embedding 一次性拼接契约。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pytest
import torch

from core.fc_duplex.system_prefill import (
    FcAudioPrefillPart,
    FcSystemPrefillPlan,
    FcTokenIdsPrefillPart,
    build_fc_system_prefill_plan,
    materialize_fc_system_embeddings,
)
from minicpm_o5_sdk import (
    O5LazyAudio,
    O5SystemContent,
    O5SystemTextSegment,
    OpenAIFunctionDefinition,
    OpenAIToolDefinition,
)
from minicpm_o5_sdk.protocols.duplex.special_tokens import (
    O5SpecialTokenKey,
    O5SpecialTokenRegistry,
)
from minicpm_o5_sdk.protocols.duplex.training_data.system import (
    O5SystemAudioSegment,
)
from minicpm_o5_sdk.tokenizers.base import (
    MiniCPMO5TokenizerAdapter,
    TokenizerToken,
)
from minicpm_o5_sdk.tool_serializer import (
    AbstractO5ToolSerializer,
    ToolSystemBlock,
)


class _FakeTokenizer:
    """把每段 ordinary 文本映射为一个稳定 token，并记录原始调用。"""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._token_ids: dict[str, int] = {}

    def encode_ordinary_with_offsets(self, text: str) -> list[TokenizerToken]:
        """记录文本并返回单 token 编码。"""

        self.calls.append(text)
        token_id = self._token_ids.setdefault(text, 10 + len(self._token_ids))
        return [TokenizerToken(token_id=token_id, start_char=0, end_char=len(text))]


@dataclass(frozen=True)
class _FakeResolvedToken:
    """registry.get 测试返回值。"""

    token_id: int


class _FakeRegistry:
    """只实现 system prefill 使用的四个 special token。"""

    _TOKEN_IDS = {
        O5SpecialTokenKey.IM_START: 1,
        O5SpecialTokenKey.AUDIO_START: 2,
        O5SpecialTokenKey.AUDIO_END: 3,
        O5SpecialTokenKey.IM_END: 4,
    }

    def get(self, key: O5SpecialTokenKey) -> _FakeResolvedToken:
        """按语义 key 返回固定 token id。"""

        return _FakeResolvedToken(token_id=self._TOKEN_IDS[key])


class _FakeToolSerializer:
    """返回可观察顺序的三段工具 system block。"""

    def __init__(self) -> None:
        self.calls: list[list[OpenAIToolDefinition]] = []

    def render_tool_system_block(
        self,
        tools: list[OpenAIToolDefinition],
        validate: bool = True,
    ) -> ToolSystemBlock:
        """记录工具并返回固定三段文本。"""

        assert validate is True
        self.calls.append(tools)
        return ToolSystemBlock(
            preamble="<tool-preamble>",
            definitions="<tool-definitions>",
            guidelines="<tool-guidelines>",
        )


def _audio_segment(sample_value: float) -> O5SystemAudioSegment:
    """构造一个采样点的合法 SDK lazy audio segment。"""

    return O5SystemAudioSegment(
        audio=O5LazyAudio(
            duration_sec=1 / 16000,
            get_tensor_fn=lambda: torch.tensor(
                [sample_value],
                dtype=torch.float32,
            ),
        )
    )


def _build_plan(
    system: O5SystemContent,
) -> tuple[FcSystemPrefillPlan, _FakeTokenizer, _FakeToolSerializer]:
    """使用精确 SDK 参数类型调用计划构造器。"""

    tokenizer = _FakeTokenizer()
    serializer = _FakeToolSerializer()
    plan = build_fc_system_prefill_plan(
        system=system,
        tokenizer=cast(MiniCPMO5TokenizerAdapter, tokenizer),
        registry=cast(O5SpecialTokenRegistry, _FakeRegistry()),
        tool_serializer=cast(AbstractO5ToolSerializer, serializer),
    )
    return plan, tokenizer, serializer


def test_text_audio_text_preserves_order_and_merges_token_chunks() -> None:
    """text-audio-text 必须保序，audio 两侧相邻 token 各自合并。"""

    audio = _audio_segment(0.25)
    plan, tokenizer, _ = _build_plan(
        O5SystemContent(
            segments=[
                O5SystemTextSegment(text="before"),
                audio,
                O5SystemTextSegment(text="after"),
            ]
        )
    )

    assert tokenizer.calls == ["system\n", "before", "after"]
    assert plan.audio_segment_count == 1
    assert plan.audit_token_ids == [1, 10, 11, 2, 3, 12, 4]
    assert plan.parts == [
        FcTokenIdsPrefillPart(token_ids=[1, 10, 11, 2]),
        FcAudioPrefillPart(segment=audio),
        FcTokenIdsPrefillPart(token_ids=[3, 12, 4]),
    ]


def test_two_audio_segments_keep_one_merged_boundary_token_part() -> None:
    """连续两段 audio 中间只能有合并后的 AUDIO_END+AUDIO_START token part。"""

    first_audio = _audio_segment(0.1)
    second_audio = _audio_segment(0.2)
    plan, _, _ = _build_plan(
        O5SystemContent(segments=[first_audio, second_audio])
    )

    assert plan.audio_segment_count == 2
    assert plan.audit_token_ids == [1, 10, 2, 3, 2, 3, 4]
    assert plan.parts == [
        FcTokenIdsPrefillPart(token_ids=[1, 10, 2]),
        FcAudioPrefillPart(segment=first_audio),
        FcTokenIdsPrefillPart(token_ids=[3, 2]),
        FcAudioPrefillPart(segment=second_audio),
        FcTokenIdsPrefillPart(token_ids=[3, 4]),
    ]


def test_tools_follow_segments_without_automatic_newline() -> None:
    """工具三段紧随用户文本编码，计划不得自动补换行。"""

    tool = OpenAIToolDefinition(
        function=OpenAIFunctionDefinition(
            name="lookup",
            parameters={"type": "object", "properties": {}},
        )
    )
    plan, tokenizer, serializer = _build_plan(
        O5SystemContent(
            segments=[O5SystemTextSegment(text="system-body")],
            tools=[tool],
        )
    )

    assert tokenizer.calls == [
        "system\n",
        "system-body",
        "<tool-preamble>",
        "<tool-definitions>",
        "<tool-guidelines>",
    ]
    assert "\n" not in tokenizer.calls[1:]
    assert serializer.calls == [[tool]]
    assert plan.audit_token_ids == [1, 10, 11, 12, 13, 14, 4]
    assert plan.parts == [
        FcTokenIdsPrefillPart(
            token_ids=[1, 10, 11, 12, 13, 14, 4]
        )
    ]


def test_materializer_embeds_in_order_and_concatenates_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """materializer 必须依次物化 parts，最后只调用一次 torch.cat。"""

    audio = _audio_segment(0.5)
    plan = FcSystemPrefillPlan(
        parts=[
            FcTokenIdsPrefillPart(token_ids=[1, 2]),
            FcAudioPrefillPart(segment=audio),
            FcTokenIdsPrefillPart(token_ids=[3]),
        ],
        audit_token_ids=[1, 2, 3],
        audio_segment_count=1,
    )
    events: list[str] = []
    cat_calls: list[int] = []
    original_cat = torch.cat

    def embed_token_ids(token_ids: list[int]) -> torch.Tensor:
        """记录 token part 并返回两维 CPU embedding。"""

        events.append(f"tokens:{token_ids}")
        return torch.tensor(
            [[float(token_id), -float(token_id)] for token_id in token_ids]
        )

    def embed_audio(segment: O5SystemAudioSegment) -> torch.Tensor:
        """记录 audio part 并返回两维 CPU embedding。"""

        events.append(f"audio:{float(segment.audio.get_tensor()[0])}")
        return torch.tensor([[9.0, -9.0]])

    def count_cat(
        tensors: list[torch.Tensor],
        dim: int = 0,
    ) -> torch.Tensor:
        """记录最终拼接次数并委托给真实 torch.cat。"""

        cat_calls.append(dim)
        return original_cat(tensors, dim=dim)

    monkeypatch.setattr(torch, "cat", count_cat)
    embeddings = materialize_fc_system_embeddings(
        plan=plan,
        embed_token_ids=embed_token_ids,
        embed_audio=embed_audio,
    )

    assert events == ["tokens:[1, 2]", "audio:0.5", "tokens:[3]"]
    assert cat_calls == [0]
    assert embeddings.tolist() == [
        [1.0, -1.0],
        [2.0, -2.0],
        [9.0, -9.0],
        [3.0, -3.0],
    ]
