"""FC Semantic Realtime API v3 的 system 与独立 TTS 音频输入。

本模块只负责把显式、可校验的 wire DTO 转成 MiniCPMO5 SDK 的
``O5SystemContent``。音频路径在建模时读取元数据，真实波形保持延迟加载；加载时不做
重采样或声道折叠，以便 16 kHz、单声道、float32 invariant 失败时立即报错。
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import librosa
import numpy as np
import soundfile as sf
import torch
from minicpm_o5_sdk import (
    O5LazyAudio,
    O5SystemContent,
    OpenAIToolDefinition,
)
from minicpm_o5_sdk.protocols.duplex.training_data import (
    O5SystemAudioSegment,
    O5SystemTextSegment,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator


class FcAudioPathInput(BaseModel):
    """服务端可读取的绝对音频路径。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Literal["path"] = "path"
    file_path: str = Field(min_length=1)

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, value: str) -> str:
        """校验音频路径是已存在、服务端可读的绝对普通文件。

        参数:
            value: API 请求中的音频文件路径。

        返回:
            规范化后的绝对路径字符串。
        """

        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError("file_path 必须是服务端可读取的绝对路径")
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"file_path 不是普通文件: {resolved}")
        return str(resolved)


class FcSystemTextInput(BaseModel):
    """v3 system 中保持原顺序的一段文本。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["text"] = "text"
    text: str


class FcSystemAudioInput(BaseModel):
    """v3 system 中保持原顺序的一段路径音频。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["audio"] = "audio"
    audio: FcAudioPathInput


FcSystemSegmentInput = Annotated[
    FcSystemTextInput | FcSystemAudioInput,
    Field(discriminator="kind"),
]


class FcSystemContentInput(BaseModel):
    """v3 wire system：有序多模态 segments 与嵌套工具定义。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    segments: list[FcSystemSegmentInput] = Field(default_factory=list)
    tools: list[OpenAIToolDefinition] = Field(default_factory=list)


def _load_o5_audio_tensor(file_path: str) -> torch.Tensor:
    """按 O5 invariant 延迟读取一段音频。

    参数:
        file_path: 已校验的 16 kHz 单声道音频绝对路径。

    返回:
        一维 ``torch.float32`` 音频 Tensor。
    """

    samples, sample_rate = librosa.load(
        file_path,
        sr=None,
        mono=False,
        dtype=np.float32,
    )
    if sample_rate != 16_000:
        raise ValueError(
            f"system audio 必须是 16 kHz，实际 sample_rate={sample_rate}: {file_path}"
        )
    if samples.ndim != 1:
        raise ValueError(
            f"system audio 必须是单声道，实际 shape={samples.shape}: {file_path}"
        )
    if samples.dtype != np.float32:
        raise ValueError(
            f"system audio 必须解码为 float32，实际 dtype={samples.dtype}: {file_path}"
        )
    return torch.from_numpy(np.ascontiguousarray(samples))


def _materialize_audio_segment(
    segment: FcSystemAudioInput,
) -> O5SystemAudioSegment:
    """把路径 DTO 转成携带可靠时长元数据的 SDK 延迟音频段。

    参数:
        segment: 已通过 wire schema 校验的 system 音频段。

    返回:
        可由 SDK 在真正需要波形时加载的 system 音频段。
    """

    file_path = segment.audio.file_path
    info = sf.info(file_path)
    if info.samplerate != 16_000:
        raise ValueError(
            f"system audio 必须是 16 kHz，实际 sample_rate={info.samplerate}: "
            f"{file_path}"
        )
    if info.channels != 1:
        raise ValueError(
            f"system audio 必须是单声道，实际 channels={info.channels}: {file_path}"
        )
    if info.frames <= 0:
        raise ValueError(f"system audio 不能为空: {file_path}")
    duration_sec = info.frames / info.samplerate
    lazy_audio = O5LazyAudio(
        duration_sec=duration_sec,
        get_tensor_fn=lambda path=file_path: _load_o5_audio_tensor(path),
        file_path=file_path,
    )
    return O5SystemAudioSegment(audio=lazy_audio)


def materialize_o5_system_content(
    system_input: FcSystemContentInput,
) -> O5SystemContent:
    """把 v3 wire system 无损转换为 SDK system content。

    参数:
        system_input: 保持 text/audio 原始顺序并包含嵌套工具定义的 wire DTO。

    返回:
        可直接传给 ``MiniCPMO.fc_duplex_prepare`` 的 SDK system content。
    """

    segments: list[O5SystemTextSegment | O5SystemAudioSegment] = []
    for segment in system_input.segments:
        if isinstance(segment, FcSystemTextInput):
            segments.append(O5SystemTextSegment(text=segment.text))
        else:
            segments.append(_materialize_audio_segment(segment))
    return O5SystemContent(
        segments=segments,
        tools=list(system_input.tools),
    )


def project_o5_system_content_input(
    system_content: O5SystemContent,
    *,
    data_root: Path,
) -> FcSystemContentInput:
    """把 SDK system content 投影为服务端路径型 v3 DTO。

    参数:
        system_content: 已从 TrainingData 加载的 SDK system 内容。
        data_root: 相对媒体路径的解析根目录。

    返回:
        保持 segment 顺序、可直接交给 FC View 的强类型 v3 输入。
    """

    projected: list[FcSystemTextInput | FcSystemAudioInput] = []
    for segment in system_content.segments:
        if isinstance(segment, O5SystemTextSegment):
            projected.append(FcSystemTextInput(text=segment.text))
            continue
        if not isinstance(segment, O5SystemAudioSegment):
            raise TypeError(
                "system.segments 仅支持 O5SystemTextSegment/O5SystemAudioSegment，"
                f"实际类型={type(segment).__name__}"
            )
        file_path = segment.audio.file_path
        if not file_path:
            raise ValueError("TrainingData system audio 缺少 file_path")
        candidate = Path(file_path)
        resolved = candidate if candidate.is_absolute() else data_root / candidate
        projected.append(
            FcSystemAudioInput(
                audio=FcAudioPathInput(file_path=str(resolved.resolve(strict=True)))
            )
        )
    return FcSystemContentInput(
        segments=projected,
        tools=list(system_content.tools),
    )
