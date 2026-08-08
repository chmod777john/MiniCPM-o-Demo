"""把 SDK 的 O5 system 语义内容编排为 Demo 本地的一次性模型预填充。

本模块只负责推理侧的两步转换：先把有序 text/audio segments 与工具定义编排成
token/audio parts，再把所有 parts 物化并拼成一个完整 embedding tensor。协议数据
结构、tokenizer、special-token registry 和工具序列化均直接复用 minicpm_o5_sdk。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypeAlias, TypedDict

import torch

from minicpm_o5_sdk import O5SystemContent, O5SystemTextSegment
from minicpm_o5_sdk.protocols.duplex.special_tokens import (
    O5SpecialTokenKey,
    O5SpecialTokenRegistry,
)
from minicpm_o5_sdk.protocols.duplex.training_data.system import O5SystemAudioSegment
from minicpm_o5_sdk.tokenizers.base import MiniCPMO5TokenizerAdapter
from minicpm_o5_sdk.tool_serializer import AbstractO5ToolSerializer


@dataclass(frozen=True)
class FcTokenIdsPrefillPart:
    """一段可由语言模型 embedding table 直接物化的连续 token ids。"""

    token_ids: list[int]


@dataclass(frozen=True)
class FcAudioPrefillPart:
    """一段需要通过 MiniCPM-o 音频编码器物化的 system audio。"""

    segment: O5SystemAudioSegment


FcSystemPrefillPart: TypeAlias = FcTokenIdsPrefillPart | FcAudioPrefillPart
"""FC system prefill 中按模型输入顺序排列的 token/audio part。"""


@dataclass(frozen=True)
class FcSystemPrefillPlan:
    """可审计、可延迟物化的 FC system prefill 计划。"""

    parts: list[FcSystemPrefillPart]
    audit_token_ids: list[int]
    audio_segment_count: int


class FcEmbeddingResizeInfo(TypedDict):
    """FC prepare 期间 embedding table resize 的固定结果。"""

    old_vocab: int
    new_vocab: int
    resized: bool
    need: int


class FcModelPrepareResult(TypedDict):
    """Modeling capability 返回给共享 View 的固定 prepare 结果。"""

    prefill_ids: list[int]
    resize_info: FcEmbeddingResizeInfo
    output_render: str
    generate_audio: bool
    tts_prompt_audio_path: str | None
    has_system_audio: bool


def _encode_ordinary(
    tokenizer: MiniCPMO5TokenizerAdapter,
    text: str,
) -> list[int]:
    """使用 SDK ordinary tokenizer 编码文本，不允许隐式插入 control token。"""

    if not text:
        return []
    return [
        token.token_id
        for token in tokenizer.encode_ordinary_with_offsets(text)
    ]


def build_fc_system_prefill_plan(
    system: O5SystemContent,
    tokenizer: MiniCPMO5TokenizerAdapter,
    registry: O5SpecialTokenRegistry,
    tool_serializer: AbstractO5ToolSerializer,
) -> FcSystemPrefillPlan:
    """按 O5 system 原始顺序构造 Demo 本地推理计划。

    参数:
        system: SDK 定义的 system text/audio segments 与工具定义。
        tokenizer: 当前模型 target 对应的 SDK ordinary tokenizer adapter。
        registry: 当前 tokenizer target 对应的 SDK special-token registry。
        tool_serializer: SDK 工具序列化器。

    返回:
        包含有序 token/audio parts、纯 token 审计序列和音频段数量的计划。
    """

    parts: list[FcSystemPrefillPart] = []
    audit_token_ids: list[int] = []

    def append_token_ids(token_ids: list[int]) -> None:
        """追加 token，并把相邻 token part 合并为一个连续块。"""

        if not token_ids:
            return
        audit_token_ids.extend(token_ids)
        if parts and isinstance(parts[-1], FcTokenIdsPrefillPart):
            previous = parts[-1]
            parts[-1] = FcTokenIdsPrefillPart(
                token_ids=[*previous.token_ids, *token_ids]
            )
            return
        parts.append(FcTokenIdsPrefillPart(token_ids=list(token_ids)))

    append_token_ids(
        [
            registry.get(O5SpecialTokenKey.IM_START).token_id,
            *_encode_ordinary(tokenizer, "system\n"),
        ]
    )

    audio_segment_count = 0
    for segment in system.segments:
        if isinstance(segment, O5SystemTextSegment):
            append_token_ids(_encode_ordinary(tokenizer, segment.text))
            continue
        if isinstance(segment, O5SystemAudioSegment):
            append_token_ids(
                [registry.get(O5SpecialTokenKey.AUDIO_START).token_id]
            )
            parts.append(FcAudioPrefillPart(segment=segment))
            append_token_ids(
                [registry.get(O5SpecialTokenKey.AUDIO_END).token_id]
            )
            audio_segment_count += 1
            continue
        raise TypeError(
            "system.segments 仅支持 O5SystemTextSegment/O5SystemAudioSegment，"
            f"实际类型={type(segment).__name__}"
        )

    if system.tools:
        tool_block = tool_serializer.render_tool_system_block(system.tools)
        append_token_ids(_encode_ordinary(tokenizer, tool_block.preamble))
        append_token_ids(_encode_ordinary(tokenizer, tool_block.definitions))
        append_token_ids(_encode_ordinary(tokenizer, tool_block.guidelines))

    append_token_ids([registry.get(O5SpecialTokenKey.IM_END).token_id])
    return FcSystemPrefillPlan(
        parts=parts,
        audit_token_ids=audit_token_ids,
        audio_segment_count=audio_segment_count,
    )


def materialize_fc_system_embeddings(
    plan: FcSystemPrefillPlan,
    embed_token_ids: Callable[[list[int]], torch.Tensor],
    embed_audio: Callable[[O5SystemAudioSegment], torch.Tensor],
) -> torch.Tensor:
    """按计划顺序物化所有 parts，并且只执行一次最终拼接。

    参数:
        plan: `build_fc_system_prefill_plan` 生成的有序计划。
        embed_token_ids: 把连续 token ids 转为 `[长度, 隐藏维]` tensor 的回调。
        embed_audio: 把一个 system audio segment 转为同布局 tensor 的回调。

    返回:
        可直接一次传给 `StreamDecoder.feed` 的完整二维 embedding tensor。
    """

    embedding_parts: list[torch.Tensor] = []
    for part in plan.parts:
        if isinstance(part, FcTokenIdsPrefillPart):
            embedding_parts.append(embed_token_ids(part.token_ids))
        else:
            embedding_parts.append(embed_audio(part.segment))
    if not embedding_parts:
        raise ValueError("FC system prefill plan 不得为空")
    return torch.cat(embedding_parts, dim=0)
