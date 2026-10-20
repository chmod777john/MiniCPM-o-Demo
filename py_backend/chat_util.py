"""Chat request/message parsing utilities for the backend protocol server.

These helpers parse a `chat.request` protocol packet into a structured request,
and translate frontend raw messages into the model message format. They are
transport-agnostic and reused by the backend protocol server before inference.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from core.schemas.common import (
    AudioContent,
    ContentItem,
    ImageContent,
    Message,
    Role,
    TextContent,
    VideoContent,
)


class ChatRequestError(ValueError):
    pass


@dataclass
class WorkerChatRequest:
    messages: list
    streaming: bool
    max_new_tokens: int
    do_sample: bool
    max_slice_nums: Optional[int]
    generate_audio: bool
    tts_ref_audio: Optional[np.ndarray]
    use_tts_template: bool
    omni_mode: bool
    enable_thinking: bool
    generation_options: dict = field(default_factory=dict)
    return_completion_metadata: bool = False


def parse_worker_chat_request_message(msg: Dict[str, Any]) -> WorkerChatRequest:
    """Parse a `chat.request` protocol message into a structured request."""

    if msg.get("type") != "chat.request":
        raise ChatRequestError("expected chat.request message")

    payload = msg.get("payload") or {}
    if not isinstance(payload, dict):
        raise ChatRequestError("chat.request payload must be an object")

    generation = payload.get("generation") or {}
    if not isinstance(generation, dict):
        raise ChatRequestError("chat.request generation must be an object")

    image = payload.get("image") or {}
    if not isinstance(image, dict):
        raise ChatRequestError("chat.request image must be an object")

    tts = payload.get("tts") or {}
    if not isinstance(tts, dict):
        raise ChatRequestError("chat.request tts must be an object")

    generation_options = {}
    for name, lower, upper in (("temperature", 0.0, 2.0), ("top_p", 0.0, 1.0)):
        if generation.get(name) is not None:
            value = generation[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not lower <= value <= upper:
                raise ChatRequestError(f"generation.{name} is out of range")
            if name == "top_p" and value == 0:
                raise ChatRequestError("generation.top_p must be positive")
            generation_options[name] = float(value)
    do_sample = bool(generation.get("do_sample", True))
    if generation_options.get("temperature") == 0:
        do_sample = False
        # HF greedy generation does not consume a sampling temperature of zero.
        generation_options.pop("temperature")

    if payload.get("streaming", True) and payload.get("return_completion_metadata"):
        raise ChatRequestError("Completion metadata currently requires streaming=false")

    max_slice_nums = None
    if image.get("max_slice_nums") is not None:
        max_slice_nums = int(image["max_slice_nums"])

    generate_audio = bool(tts.get("enabled", False))
    if generate_audio and bool(payload.get("enable_thinking", False)):
        raise ChatRequestError("Speech chat requires enable_thinking=false; thinking is currently text-only")
    tts_ref_audio = None
    ref_b64 = tts.get("ref_audio_data")
    if generate_audio and ref_b64:
        tts_ref_audio = np.frombuffer(base64.b64decode(ref_b64), dtype=np.float32)

    if int(generation.get("max_new_tokens", 256)) < 1:
        raise ChatRequestError("max_new_tokens must be positive")
    if not payload.get("messages"):
        raise ChatRequestError("messages must be nonempty")
    return WorkerChatRequest(
        messages=payload.get("messages", []),
        streaming=bool(payload.get("streaming", True)),
        max_new_tokens=int(generation.get("max_new_tokens", 256)),
        do_sample=do_sample,
        max_slice_nums=max_slice_nums,
        generate_audio=generate_audio,
        tts_ref_audio=tts_ref_audio,
        use_tts_template=bool(payload.get("use_tts_template", False) or generate_audio),
        omni_mode=bool(payload.get("omni_mode", False)),
        enable_thinking=bool(payload.get("enable_thinking", False)),
        generation_options=generation_options,
        return_completion_metadata=bool(payload.get("return_completion_metadata", False)),
    )


def parse_raw_messages(raw_messages: List[dict]) -> List[Message]:
    """Parse frontend raw messages into schema messages."""

    messages: List[Message] = []
    for raw_message in raw_messages:
        role = Role(raw_message["role"])
        content = raw_message["content"]
        if isinstance(content, list):
            content_items: List[ContentItem] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text" and item.get("text"):
                    content_items.append(TextContent(text=item["text"]))
                elif item.get("type") == "audio" and item.get("data"):
                    content_items.append(AudioContent(data=item["data"]))
                elif item.get("type") == "image" and item.get("data"):
                    content_items.append(ImageContent(data=item["data"]))
                elif item.get("type") == "video" and item.get("data"):
                    content_items.append(VideoContent(
                        data=item["data"],
                        stack_frames=item.get("stack_frames", 1),
                    ))
            if content_items:
                messages.append(Message(role=role, content=content_items))
        else:
            messages.append(Message(role=role, content=content))
    return messages


def convert_to_model_msgs(schema_messages: List[Message]) -> list:
    """Convert schema messages into the current model message format."""

    from core.processors.base import MiniCPMOProcessorMixin

    mixin = MiniCPMOProcessorMixin()
    model_msgs = []
    for message in schema_messages:
        content = mixin._convert_content_to_model_format(message.content)
        if len(content) == 1 and isinstance(content[0], str):
            content = content[0]
        model_msgs.append({"role": message.role.value, "content": content})
    return model_msgs
