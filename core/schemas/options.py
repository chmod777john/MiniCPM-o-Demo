"""Reusable option and parameter schemas.

These are value objects that may appear in multiple request lifecycles. Keep
them small and avoid adding transport, queueing, or runtime state here.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator

from core.schemas.content import TTSMode


class TTSSamplingParams(BaseModel):
    """TTS audio-token sampling parameters."""

    top_p: float = Field(
        0.85,
        ge=0.0,
        le=1.0,
        description="Top-P 采样阈值，控制采样多样性",
    )
    min_p: float = Field(
        0.01,
        ge=0.0,
        le=1.0,
        description="最小概率阈值",
    )
    top_k: int = Field(
        25,
        ge=0,
        description="Top-K 采样数量",
    )
    repetition_penalty: float = Field(
        1.05,
        ge=1.0,
        description="重复惩罚系数",
    )
    temperature: float = Field(
        0.8,
        ge=0.0,
        le=2.0,
        description="采样温度",
    )
    win_size: int = Field(
        16,
        ge=1,
        description="重复检测滑动窗口大小",
    )
    tau_r: float = Field(
        0.1,
        ge=0.0,
        description="温度调节参数",
    )


class TTSConfig(BaseModel):
    """Request-level TTS output configuration."""

    enabled: bool = Field(False, description="是否启用 TTS 输出")
    # Legacy-mode; removal-candidate.
    mode: TTSMode = Field(
        TTSMode.AUDIO_ASSISTANT,
        description="TTS 模式（推荐 AUDIO_ASSISTANT）",
    )
    ref_audio_path: Optional[str] = Field(
        None,
        description="参考音频路径（16kHz mono WAV）",
    )
    ref_audio_data: Optional[str] = Field(
        None,
        description="参考音频 Base64 数据",
    )
    ref_audio_max_ms: Optional[int] = Field(
        None,
        ge=1000,
        description="参考音频最大使用长度（毫秒），建议 3000-10000",
    )
    output_path: Optional[str] = Field(
        None,
        description="输出音频保存路径",
    )
    language: str = Field(
        "en",
        description="语言（'en' 英语 / 'zh' 中文）",
    )
    sampling: TTSSamplingParams = Field(
        default_factory=TTSSamplingParams,
        description="TTS 采样参数",
    )

    @model_validator(mode="after")
    def check_ref_audio_when_enabled(self) -> "TTSConfig":
        """Processor/runtime layers may fill the default reference audio."""

        return self


class ImageConfig(BaseModel):
    """Image preprocessing options."""

    max_slice_nums: Optional[int] = Field(
        None,
        ge=1,
        le=16,
        description="HD 图像最大切片数（None 为自动，1-16）",
    )
    use_image_id: bool = Field(
        False,
        description="是否使用图像 ID（多图像场景）",
    )


class GenerationConfig(BaseModel):
    """Turn-based/streaming LLM text generation parameters."""

    max_new_tokens: int = Field(
        512,
        ge=1,
        le=4096,
        description="最大生成 token 数",
    )
    min_new_tokens: int = Field(
        0,
        ge=0,
        description="最小生成 token 数",
    )
    do_sample: bool = Field(
        True,
        description="是否采样（False 为贪婪解码）",
    )
    temperature: float = Field(
        0.7,
        ge=0.0,
        le=2.0,
        description="采样温度",
    )
    top_p: float = Field(
        0.8,
        ge=0.0,
        le=1.0,
        description="Top-P 采样",
    )
    top_k: int = Field(
        100,
        ge=0,
        description="Top-K 采样（0 禁用）",
    )
    length_penalty: float = Field(
        1.1,
        ge=0.1,
        le=5.0,
        description="长度惩罚系数。>1.0 抑制 EOS token 使输出更长更详细，=1.0 不惩罚，<1.0 鼓励更早结束",
    )
    max_inp_length: int = Field(
        8192,
        ge=1,
        description="最大输入长度（token 数）",
    )


__all__ = [
    "TTSSamplingParams",
    "TTSConfig",
    "ImageConfig",
    "GenerationConfig",
]
