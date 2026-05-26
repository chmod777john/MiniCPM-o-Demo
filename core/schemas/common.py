"""Backward-compatible common schema exports.

Historically this module defined conversation content, TTS options, image
options, and LLM generation options in one file. The concrete definitions now
live in narrower modules:

- `core.schemas.content`: roles, multimodal content, and messages.
- `core.schemas.options`: reusable request option/value objects.

Keep importing from `core.schemas.common` working for existing code.
"""

from core.schemas.content import (
    AudioContent,
    ContentItem,
    ContentType,
    ImageContent,
    Message,
    Role,
    TTSMode,
    TextContent,
    VideoContent,
)
from core.schemas.options import (
    GenerationConfig,
    ImageConfig,
    TTSConfig,
    TTSSamplingParams,
)

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
    "TTSSamplingParams",
    "TTSConfig",
    "ImageConfig",
    "GenerationConfig",
]
