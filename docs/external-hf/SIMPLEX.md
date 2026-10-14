# 桌面与移动端单工：外部 HF

## 边界

完整历史 `messages` → Gateway `/v1/realtime?mode=chat` → Worker/backend → ChatView → 外部 HF `chat(...)`。

- `stream=False`：HF 完整多模态预处理、LLM 生成、可选完整 TTS；facade 只把 HF WAV 输出转换成 Demo 响应。
- `stream=True, generate_audio=False`：HF `chat` → 原生文字 streamer。
- `stream=True, generate_audio=True`：HF 同一套完整历史预处理和 canvas 位置编码 → HF 已有 `streaming_generate` 的文字分块/TTS/Token2wav，输出 `(waveform, text)`。Demo 只编码 PCM 并发送事件。
- API 原来的 prepare/generate 分段保留；external 模式的 prepare 只暂存完整请求。不会将历史逐条送入半双工 `streaming_prefill`。
- 每次请求完整 prefill，不复用上轮 KV；普通双工与 FC 路径不改。没有复制 vendored modeling 到 adapter。accel 未修改。

## 桌面请求映射

| 页面/API 字段 | HF 含义 |
|---|---|
| `messages` | 用户显式提供的完整 system/user/assistant 历史；不额外注入 system 消息 |
| `streaming` | `chat(stream=...)` |
| `generation.max_new_tokens` | 文字生成上限；流式 TTS 最后一个 token 也取得 hidden，且不再多采一个 token |
| `generation.do_sample` | `do_sample`；修复旧 parser 丢字段、View 错传 `sampling` |
| `generation.length_penalty` | EOS logit 偏好；正值除、负值乘，不是 beam-search 长度归一化 |
| `image.max_slice_nums` | HF processor 图像切片；桌面 HD 图片为 2，视频为 1 |
| `omni_mode` | HF 多模态内容拼接策略 |
| `tts.enabled` / `use_tts_template` | 是否生成语音 / 使用 TTS prompt；语音自动启用 template |
| `tts.ref_audio_data` | 独立 16kHz mono float32 PCM reference；不把它当作用户输入 |
| `tts.mode` | 页面音频助手/视频 Omni 的产品标签；HF 实际由 `omni_mode`、显式 messages 和独立 reference 决定 |
| `enable_thinking` | 文字单工可开；当前语音单工要求 false，桌面默认即 false |

页面的 max-KV 是旧增量半双工选项，单工 chat 不发送它。HF 单工默认输入上限仍为 8192；本次没有把它扩成 128k 单工验收。

## 生命周期

HF 文字流 iterator 负责 producer 线程，生成异常传给消费方；关闭 iterator 会停止并 join。API 单工关闭请求先标记取消并等待生成退出，然后释放会话。语音在 chunk 边界取消；非流请求等待当前生成结束再释放模型。尚未声称所有进程故障都能自动恢复。

## 已验收（2026-10-14）

- 双 A100 TP、allmerge `iter_300.pt`，project venv `o5-3drope-fc-20261013/.venv`，accel `6e33931`。
- Job 1217986：12 项真实 Gateway → Worker → backend → HF API 请求全部通过：文字流/非流、语音流/非流、完整历史、HD图片、音频输入、视频输入+语音流/非流、长语音、1-token上限、文字 thinking。
- 文本/语音取消后立即再请求通过；取消到 closed 实测约 0.040s / 0.321s（这两个样本，不是通用上界）。
- 单工 → 普通双工 2 units → 单工文字和语音回归通过；双工两个 unit 均为 listen，不代表 speaking 性能验收。
- 50组 CPU cap/EOS/hidden/四轴位置边界；HF参数筛选；Demo完整历史/选项/PCM/一次性请求契约通过。
- `tests/test_external_hf_chat_contract.py` 是 CPU 检查；HF 仓库 `tests/test_chat_{options,stream_boundaries}.py` 是另外两项。
- 本次没有浏览器/真机 UI 操作验收，也没有声学主观评分、与旧 main 单工输出对齐或全面精度性能评测。

原始事件/音频/日志、运行入口：`/user/weihongliang/analysis-runs/20261014-hf-simplex/`。
精简验收数据：`docs/external-hf/evidence/20261014-simplex/`。
