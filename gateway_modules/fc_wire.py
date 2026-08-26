"""Torch-free FC wire DTOs used by the Gateway.

The backend has a richer ``core.fc_duplex.system_input`` module which turns
these values into model-side SDK objects and lazy audio tensors.  The Gateway
only validates and serializes the public wire shape, so importing that backend
module here would unnecessarily pull in Torch and the model implementation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class OpenAIFunctionDefinition(BaseModel):
    """Torch-free OpenAI function definition used at the wire boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    description: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    strict: bool | None = None


class OpenAIToolDefinition(BaseModel):
    """Torch-free OpenAI function tool definition."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["function"] = "function"
    function: OpenAIFunctionDefinition


class FcAudioPathInput(BaseModel):
    """A server-readable absolute audio path."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: Literal["path"] = "path"
    file_path: str = Field(min_length=1)

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, value: str) -> str:
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError("file_path must be an absolute server-readable path")
        resolved = path.resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"file_path is not a regular file: {resolved}")
        return str(resolved)


class FcSystemTextInput(BaseModel):
    """An ordered text segment in the v3 system content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["text"] = "text"
    text: str


class FcSystemAudioInput(BaseModel):
    """An ordered path-backed audio segment in the v3 system content."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["audio"] = "audio"
    audio: FcAudioPathInput


FcSystemSegmentInput = Annotated[
    FcSystemTextInput | FcSystemAudioInput,
    Field(discriminator="kind"),
]


class FcSystemContentInput(BaseModel):
    """Torch-free v3 system content wire DTO."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    segments: list[FcSystemSegmentInput] = Field(default_factory=list)
    tools: list[OpenAIToolDefinition] = Field(default_factory=list)


__all__ = [
    "FcAudioPathInput",
    "FcSystemAudioInput",
    "FcSystemContentInput",
    "FcSystemTextInput",
    "OpenAIFunctionDefinition",
    "OpenAIToolDefinition",
]
