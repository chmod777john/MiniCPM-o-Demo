"""InferenceBackend protocol parameter schemas.

This module organizes backend-facing types by lifecycle:

- session init parameters
- unary request/response payloads
- push input/control messages
- pull events, metrics, and errors

It deliberately reuses public API value objects from `content.py` and
`options.py` instead of redefining message, generation, vision, and TTS leaves.
Deployment-time resources such as model paths, devices, ports, and GPU layers
belong to service config/factory code, not to this backend protocol.
"""

from __future__ import annotations

import time
from typing import Annotated, Any, Dict, List, Literal, Optional, TypeAlias, Union
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from core.schemas.content import ContentItem, Message
from core.schemas.duplex import DuplexConfig
from core.schemas.options import GenerationConfig, ImageConfig, TTSConfig


def _new_id() -> str:
    return uuid4().hex


def _now_ms() -> int:
    return int(time.time() * 1000)


class BackendSchemaModel(BaseModel):
    """Base class for backend protocol models."""

    model_config = ConfigDict(extra="forbid")


BackendMode: TypeAlias = Literal["turn_based", "full_duplex"]
BackendMessageType: TypeAlias = Literal["input", "control", "close"]


class TurnBasedBackendInit(BackendSchemaModel):
    mode: Literal["turn_based"] = "turn_based"
    system_prompt: Optional[str] = None
    chat_vocoder: Optional[Literal["token2wav", "cosyvoice2"]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FullDuplexBackendInit(BackendSchemaModel):
    mode: Literal["full_duplex"] = "full_duplex"
    system_prompt: Optional[str] = None
    ref_audio_path: Optional[str] = None
    tts_ref_audio_path: Optional[str] = None
    duplex: DuplexConfig = Field(default_factory=DuplexConfig)
    pause_timeout_s: Optional[float] = Field(None, ge=0.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)


BackendInitParams: TypeAlias = Annotated[
    Union[TurnBasedBackendInit, FullDuplexBackendInit],
    Field(discriminator="mode"),
]


class TurnBasedUnaryRequest(BackendSchemaModel):
    """One-shot turn-based request carried by `backend.unary(...)`."""

    type: Literal["turn_based"] = "turn_based"
    request_id: str = Field(default_factory=_new_id)
    timestamp_ms: int = Field(default_factory=_now_ms)
    messages: List[Message] = Field(..., min_length=1)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    use_tts_template: bool = False
    omni_mode: bool = False
    enable_thinking: bool = False
    return_prompt: bool = False
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TurnBasedUnaryUsage(BackendSchemaModel):
    """Token usage summary for a turn-based unary result."""

    input_tokens: Optional[int] = Field(None, ge=0)
    output_tokens: Optional[int] = Field(None, ge=0)
    total_tokens: Optional[int] = Field(None, ge=0)
    cached_tokens: Optional[int] = Field(None, ge=0)


class BackendMetrics(BackendSchemaModel):
    backend: Optional[str] = None
    kv_cache_length: Optional[int] = Field(None, ge=0)
    n_past_max: Optional[int] = Field(None, ge=0)
    prefill_ms: Optional[float] = Field(None, ge=0.0)
    generate_ms: Optional[float] = Field(None, ge=0.0)
    wall_clock_ms: Optional[float] = Field(None, ge=0.0)
    cost_llm_ms: Optional[float] = Field(None, ge=0.0)
    cost_tts_prep_ms: Optional[float] = Field(None, ge=0.0)
    cost_tts_ms: Optional[float] = Field(None, ge=0.0)
    cost_token2wav_ms: Optional[float] = Field(None, ge=0.0)
    n_tokens: Optional[int] = Field(None, ge=0)
    n_tts_tokens: Optional[int] = Field(None, ge=0)
    vision_slices: Optional[int] = Field(None, ge=0)
    vision_tokens: Optional[int] = Field(None, ge=0)


class TurnBasedUnaryResult(BackendSchemaModel):
    type: Literal["turn_based"] = "turn_based"
    request_id: str
    text: str = ""
    audio_data: Optional[str] = None
    audio_sample_rate: Optional[int] = Field(None, ge=1)
    prompt: Optional[str] = None
    usage: Optional[TurnBasedUnaryUsage] = None
    metrics: Optional[BackendMetrics] = None


UnaryRequest: TypeAlias = TurnBasedUnaryRequest
UnaryResult: TypeAlias = TurnBasedUnaryResult


class InputHints(BackendSchemaModel):
    force_listen: Optional[bool] = None
    max_slice_nums: Optional[int] = Field(None, ge=1)


class TurnBasedStreamInput(BackendSchemaModel):
    """Turn-based streaming input carried by `push(input)`.

    Use `unary(TurnBasedUnaryRequest)` for one-shot turn-based responses.
    Use this shape only when turn-based output is consumed through `pull()`.
    """

    mode: Literal["turn_based"] = "turn_based"
    input_id: str = Field(default_factory=_new_id)
    timestamp_ms: int = Field(default_factory=_now_ms)
    messages: List[Message] = Field(..., min_length=1)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    image: ImageConfig = Field(default_factory=ImageConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    use_tts_template: bool = False
    omni_mode: bool = False
    enable_thinking: bool = False
    return_prompt: bool = False
    commit: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FullDuplexStreamInput(BackendSchemaModel):
    """Full-duplex streaming input carried by `push(input)`."""

    mode: Literal["full_duplex"] = "full_duplex"
    input_id: str = Field(default_factory=_new_id)
    timestamp_ms: int = Field(default_factory=_now_ms)
    role: Literal["user", "system"] = "user"
    content: List[ContentItem] = Field(default_factory=list)
    hints: Optional[InputHints] = None


InputBundle: TypeAlias = Annotated[
    Union[TurnBasedStreamInput, FullDuplexStreamInput],
    Field(discriminator="mode"),
]


class BackendControl(BackendSchemaModel):
    command_id: str = Field(default_factory=_new_id)
    type: Literal[
        "backend.pause",
        "backend.resume",
        "metrics.get",
        "backend.close",
    ]
    payload: Dict[str, Any] = Field(default_factory=dict)


class ClosePayload(BackendSchemaModel):
    reason: Optional[str] = None


class BackendInputMessage(BackendSchemaModel):
    message_id: str = Field(default_factory=_new_id)
    type: Literal["input"] = "input"
    timestamp_ms: int = Field(default_factory=_now_ms)
    payload: InputBundle


class BackendControlMessage(BackendSchemaModel):
    message_id: str = Field(default_factory=_new_id)
    type: Literal["control"] = "control"
    timestamp_ms: int = Field(default_factory=_now_ms)
    payload: BackendControl


class BackendCloseMessage(BackendSchemaModel):
    message_id: str = Field(default_factory=_new_id)
    type: Literal["close"] = "close"
    timestamp_ms: int = Field(default_factory=_now_ms)
    payload: ClosePayload = Field(default_factory=ClosePayload)


BackendMessage: TypeAlias = Annotated[
    Union[BackendInputMessage, BackendControlMessage, BackendCloseMessage],
    Field(discriminator="type"),
]


BackendEventType: TypeAlias = Literal[
    "backend.initialized",
    "backend.state",
    "backend.closed",
    "input.accepted",
    "input.rejected",
    "input.committed",
    "response.started",
    "response.text.delta",
    "response.audio.delta",
    "response.listen",
    "response.speak",
    "response.done",
    "metrics.snapshot",
    "metrics.frame",
    "metrics.response",
    "error",
]


class BackendEvent(BackendSchemaModel):
    version: Literal["backend.class.v1"] = "backend.class.v1"
    event_id: str = Field(default_factory=_new_id)
    type: BackendEventType
    message_id: Optional[str] = None
    response_id: Optional[str] = None
    input_id: Optional[str] = None
    timestamp_ms: int = Field(default_factory=_now_ms)
    payload: Dict[str, Any] = Field(default_factory=dict)


class BackendError(BackendSchemaModel):
    code: Literal[
        "invalid_input",
        "invalid_state",
        "unsupported",
        "context_full",
        "backend_busy",
        "backend_error",
        "engine_error",
    ]
    message: str
    scope: Literal["input", "response", "backend"]
    terminal: bool = False


__all__ = [
    "BackendMode",
    "BackendMessageType",
    "TurnBasedBackendInit",
    "FullDuplexBackendInit",
    "BackendInitParams",
    "TurnBasedUnaryRequest",
    "TurnBasedUnaryUsage",
    "TurnBasedUnaryResult",
    "BackendMetrics",
    "UnaryRequest",
    "UnaryResult",
    "InputHints",
    "TurnBasedStreamInput",
    "FullDuplexStreamInput",
    "InputBundle",
    "BackendControl",
    "ClosePayload",
    "BackendInputMessage",
    "BackendControlMessage",
    "BackendCloseMessage",
    "BackendMessage",
    "BackendEventType",
    "BackendEvent",
    "BackendError",
]
