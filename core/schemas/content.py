"""Conversation roles and multimodal content schemas."""

from __future__ import annotations

from enum import Enum
from typing import List, Literal, Union

from pydantic import BaseModel, Field, field_validator


class Role(str, Enum):
    """MiniCPM-o conversation message role."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class TTSMode(str, Enum):
    """TTS mode used by model-facing request schemas."""

    DEFAULT = "default"
    AUDIO_ASSISTANT = "audio_assistant"
    OMNI = "omni"
    AUDIO_ROLEPLAY = "audio_roleplay"
    VOICE_CLONING = "voice_cloning"


class ContentType(str, Enum):
    """Multimodal content type discriminator."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


class TextContent(BaseModel):
    """Text content in a conversation message."""

    type: Literal["text"] = "text"
    text: str = Field(..., description="文本内容")


class ImageContent(BaseModel):
    """Base64-encoded image input."""

    type: Literal["image"] = "image"
    data: str = Field(..., description="Base64 编码的图像数据")


class AudioContent(BaseModel):
    """Base64-encoded 16kHz mono PCM float32 audio input."""

    type: Literal["audio"] = "audio"
    data: str = Field(
        ...,
        description="Base64 编码的 PCM 数据（float32，16kHz，mono）",
    )
    sample_rate: int = Field(16000, description="采样率（必须为 16000）")

    @field_validator("sample_rate")
    @classmethod
    def check_sample_rate(cls, v: int) -> int:
        """Validate the model-required input audio sample rate."""

        if v != 16000:
            raise ValueError(f"采样率必须为 16000，当前为 {v}")
        return v


class VideoContent(BaseModel):
    """Base64-encoded video input."""

    type: Literal["video"] = "video"
    data: str = Field(..., description="Base64 编码的视频文件数据")
    stack_frames: int = Field(
        1,
        ge=1,
        description="高刷帧率模式帧数，1=标准（每秒 1 帧）",
    )


ContentItem = Union[TextContent, ImageContent, AudioContent, VideoContent]
"""Multimodal content item accepted by :class:`Message`."""


class Message(BaseModel):
    """One conversation message."""

    role: Role = Field(..., description="消息角色")
    content: Union[str, List[ContentItem]] = Field(
        ...,
        description="消息内容（字符串或多模态列表）",
    )

    @field_validator("content", mode="before")
    @classmethod
    def normalize_content(cls, v):
        """Keep string content in compact form."""

        return v


__all__ = [
    "Role",
    "TTSMode",
    "ContentType",
    "TextContent",
    "ImageContent",
    "AudioContent",
    "VideoContent",
    "ContentItem",
    "Message",
]
