"""MiniCPMO45 Schema 模块

本模块提供所有模式的输入/输出 Schema 定义。

模块组织：
=========

- content.py: 消息角色、多模态内容、Message
- options.py: 可复用参数（GenerationConfig, TTSConfig, TTSSamplingParams 等）
- common.py: 兼容旧入口，重导出 content/options
- chat.py: 单工对话（ChatRequest, ChatResponse）
- streaming.py: 流式对话（StreamingRequest, StreamingChunk 等）
- duplex.py: 双工对话（DuplexConfig, DuplexGenerateResult 等）
- backend.py: InferenceBackend 协议参数

快速入门：
=========

```python
# 导入所有常用类型
from core.schemas import (
    # 通用
    Message, Role,
    TextContent, ImageContent, AudioContent, VideoContent,
    TTSConfig, TTSMode,
    GenerationConfig,
    
    # 单工
    ChatRequest, ChatResponse,
    
    # 流式
    StreamingRequest, StreamingChunk,
    
    # 双工
    DuplexConfig, DuplexGenerateResult,
)
```

三种模式对比：
=============

| 模式 | 特点 | 适用场景 |
|------|------|----------|
| Chat | 一问一答，完整返回 | 简单问答、批量推理 |
| Streaming | 流式返回 | 实时 TTS、长文本 |
| Duplex | 全双工实时 | 语音助手、打断对话 |
"""

# 通用内容类型
from core.schemas.content import (
    # 枚举
    Role,
    TTSMode,
    ContentType,
    
    # 内容类型
    TextContent,
    ImageContent,
    AudioContent,
    VideoContent,
    ContentItem,
    
    # 消息
    Message,
)

# 通用参数类型
from core.schemas.options import (
    # 配置
    TTSSamplingParams,
    TTSConfig,
    ImageConfig,
    GenerationConfig,
)

# 单工对话
from core.schemas.chat import (
    ChatRequest,
    ChatResponse,
)

# 流式对话
from core.schemas.streaming import (
    StreamingConfig,
    StreamingRequest,
    StreamingChunk,
    StreamingResponse,
    RollbackResult,
)

# 双工对话
from core.schemas.duplex import (
    DuplexConfig,
    DuplexPrepareRequest,
    DuplexPrefillRequest,
    DuplexGenerateResult,
    DuplexOfflineInput,
    DuplexChunkResult,
    DuplexOfflineOutput,
)

# Backend 协议
from core.schemas.backend import (
    BackendControl,
    BackendError,
    BackendEvent,
    BackendEventType,
    BackendInitParams,
    BackendCloseMessage,
    BackendControlMessage,
    BackendInputMessage,
    BackendMessage,
    BackendMessageType,
    BackendMetrics,
    BackendMode,
    ClosePayload,
    FullDuplexBackendInit,
    FullDuplexStreamInput,
    InputBundle,
    InputHints,
    TurnBasedBackendInit,
    TurnBasedStreamInput,
    TurnBasedUnaryUsage,
    TurnBasedUnaryRequest,
    TurnBasedUnaryResult,
    UnaryRequest,
    UnaryResult,
)

__all__ = [
    # 通用 - 枚举
    "Role",
    "TTSMode",
    "ContentType",
    
    # 通用 - 内容类型
    "TextContent",
    "ImageContent",
    "AudioContent",
    "VideoContent",
    "ContentItem",
    
    # 通用 - 消息
    "Message",
    
    # 通用 - 配置
    "TTSSamplingParams",
    "TTSConfig",
    "ImageConfig",
    "GenerationConfig",
    
    # 单工对话
    "ChatRequest",
    "ChatResponse",
    
    # 流式对话
    "StreamingConfig",
    "StreamingRequest",
    "StreamingChunk",
    "StreamingResponse",
    "RollbackResult",
    
    # 双工对话
    "DuplexConfig",
    "DuplexPrepareRequest",
    "DuplexPrefillRequest",
    "DuplexGenerateResult",
    "DuplexOfflineInput",
    "DuplexChunkResult",
    "DuplexOfflineOutput",

    # Backend 协议
    "BackendControl",
    "BackendError",
    "BackendEvent",
    "BackendEventType",
    "BackendInitParams",
    "BackendCloseMessage",
    "BackendControlMessage",
    "BackendInputMessage",
    "BackendMessage",
    "BackendMessageType",
    "BackendMetrics",
    "BackendMode",
    "ClosePayload",
    "FullDuplexBackendInit",
    "FullDuplexStreamInput",
    "InputBundle",
    "InputHints",
    "TurnBasedBackendInit",
    "TurnBasedStreamInput",
    "TurnBasedUnaryUsage",
    "TurnBasedUnaryRequest",
    "TurnBasedUnaryResult",
    "UnaryRequest",
    "UnaryResult",
]
