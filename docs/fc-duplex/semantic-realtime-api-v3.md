# FC Semantic Realtime API v3

v3 是破坏性协议，`session.init.payload.protocol_version` 必须严格等于字符串 `"3"`。
服务端不接受 v2 prepare 字段或别名。

## Session 初始化

```json
{
  "type": "session.init",
  "payload": {
    "mode": "full_duplex",
    "fc_duplex": true,
    "protocol_version": "3",
    "system": {
      "segments": [
        {"kind": "text", "text": "你是一个语音助手。"},
        {
          "kind": "audio",
          "audio": {
            "source": "path",
            "file_path": "/absolute/server-readable/system.wav"
          }
        },
        {"kind": "text", "text": "保持简洁。"}
      ],
      "tools": [
        {
          "type": "function",
          "function": {
            "name": "lookup",
            "parameters": {"type": "object", "properties": {}}
          }
        }
      ]
    },
    "tts_prompt_audio": {
      "source": "path",
      "file_path": "/absolute/server-readable/tts-prompt.wav"
    },
    "generate_audio": true
  }
}
```

`system.segments` 保持 SDK 的原始 text/audio 顺序，不能拼接文本或把 system audio
提升为顶层字段。`system.tools` 是 system 的组成部分。

`tts_prompt_audio` 与 system audio 相互独立；服务端不会从 system audio 推断 TTS
prompt。当前 canonical v3 只接受服务端可读取的绝对路径，不接受 base64。

以下旧字段会直接报错：

```text
system_prompt
instructions
tools
ref_audio_path
ref_audio_base64
prompt_wav_path
tts_ref
tts_ref_audio
tts_ref_audio_path
```

其余 UnitPolicy、input/output event 与 stateless resume 语义沿用既有 Semantic
Realtime 状态机；resume identity 中 system audio 与 TTS prompt 分别使用
`system_audio_sha256` 和 `tts_prompt_audio_sha256`。
