# 外部 HF 单工独立只读 review

日期：2026-10-14 UTC。Reviewer：`simplex_final_review`。只读源码与已有验收产物，没有修改 HF/Demo/accel 源码，没有启动 GPU、服务或新实验。

## 结论

在本次明确范围内，没有发现需要阻断合入的实质性缺陷。完整历史单工的编排已归属于外部 HF；Demo 新增部分为请求暂存、参数传递、事件/音频格式转换和会话生命周期管理，没有把 LLM 采样或 TTS 推理循环复制到 Demo。

这是一轮 diff 及调用链 review，加上对现有功能验收记录的核对，不是新增完整精度、主观音质、性能或故障注入验收。

## 检查版本

- HF：`/user/weihongliang/worktrees/minicpm-o-5-fix-3d-rope-o5fc-20261013-hf-chat-20261014`，`335bfdd..5cccbf73e8ca7bc01b0377eff299891e9dd45248`。
- Demo：`/user/weihongliang/worktrees/o5-fc-dev-external-hf-20261013-hf-chat-20261014`，重点 `24d02d68..1b761e12` 的 Python 路径，以及当前 `08c22855e618e8b979d1b11a496ee9551146fd18` 的移动取消补充。
- Demo `5cb93278` 为收尾文档补充；本 review 的实现版本仍为 `08c22855`。
- 最终环境按 `final/environment.json`：torch `2.11.0+cu128`、transformers `5.13.0`、FA `2.8.3`、FLA `0.5.1`、o5-sdk `0.0.6`、minicpmo-utils `1.0.7`；未引用旧交接中的 torch2.8。
- Accel 未改：`6e33931`。仅补读图输出 clone 的边界，确认本次没有把 graph 生命周期修正塞进 canonical。
- 最终真实 API evidence 的 source manifest 是 HF `5cccbf7` / Demo `47cb5315` / accel `6e33931`，Job `1218128`。Demo 后续 `08c22855` 为移动端客户端取消逻辑，不能把此前 API 测试声称为真实手机对该 commit 的验收。

## 具体核对

### 1. 完整历史与多模态输入

- Demo `core/processors/unified.py:229` 的 external 分支只保存完整 `msgs` 和选项；不调用旧 `non_streaming_prefill`，也不逐条调用 `streaming_prefill`。
- 同文件 `streaming_generate` 一次消费 `_chat_inputs`，调用外部 `chat(stream=True)`；消费后清空，错误 session 或重复生成会报错。
- 非流 `_chat_impl` 不为 external 模型注入第二份 TTS/system prompt；独立 reference 通过 `tts_ref_audio` 传递。
- HF `chat` 继续使用原完整历史模板和 processor，再通过同一个 `generate` 计算 vision/audio embeddings 以及 canvas position IDs。语音流仅在计算完这些共同输入后分流到 `_stream_chat_audio`。
- Facade 的 `chat` 不改 inference 实现；非流时把 HF 生成的 WAV 读回，临时文件在正常与异常路径清理。流式直接返回 HF iterator。

### 2. 四轴 position 与 token-hidden 边界

- HF `modeling_minicpmo.py:1183` 计算完整 `(4,B,S)` canvas positions，与原非流路径共同使用；`prompt_inputs` 将输入 IDs/embeddings/positions 一起传给原 `streaming_generate`。
- `utils.py:402` 首次完整 prefill 使用完整 position tensor；后续一步消费一个 token，使用末端 position 加 token 序号。
- HF `modeling_minicpmo.py:2684` 跨 chunk 从完整 prompt 末位加累计生成 token 数继续，恰好对应前一 chunk 留下、尚未 forward 的尾 token；没有回退到纯 KV 序列长度的位置假设。
- `finish_on_limit` 最后额外 forward 已采样 token，收集其 hidden，不继续采样；与 `yield_chunk_token_ids` 的首/中/末 chunk 拼接匹配。
- immediate EOS 的空 hidden 改为合法 `(B,0,H)` tensor；TTS 已有 `generate_with_buffer` 会补 Text EOS/Audio BOS，不会将空 condition 直接送入空序列 attention。
- 已有 `test_chat_stream_boundaries.py` 覆盖 50 组 cap/EOS 组合，包括第一次 EOS、跨 chunk EOS、cap=1/10/11/12 等，验证 token ID 与 hidden 对应、cap、四轴位置递增。该测试使用 fake model，不能替代真实数值精度 gate。

### 3. 线程、取消、异常和模型释放

- HF `_decode_stream` 把 producer 线程放在 consumer iterator 生命周期内：开始消费才启动；producer 异常写入 errors 并发送 streamer 结束；消费完抛出原异常；`finally` 标记取消并 join。
- Demo `ChatView.streaming_generate` finally 关闭 HF iterator；backend 生成线程 finally 关闭 View iterator。
- backend `close` 标记 `_chat_cancel` 并等待 `_chat_generation_done`；`_stream_turn_based` finally await executor task，之后外层才设置 done。正常 close 不会因为固定 5 秒超时把仍在工作的模型释放给另一 session。
- `_push_turn_based` 把 parser/prefill/generate 包在统一 finally 内，解析失败也恢复 done；非流完成后检查 `closed`，关闭期间不再向客户端发布完成数据。
- 此设计是协作式取消：语音等 chunk 边界，非流等当前生成完成。已有 raw-disconnect 较慢回收记录与这一点一致，不能承诺任意时刻的即时打断。
- 移动端 `08c22855` 在 session 已创建且 socket OPEN 时发送 `session.close`，用户界面立即停止，等 ACK 或 1.5 秒关闭兜底；late events 不进入已完成 UI，queued/connecting 直接关闭。该补充没有改后端。

### 4. TTS/LLM 状态生命周期与既有功能

- `_stream_chat_audio` 开始和 finally 均清空 LLM/audio/turn 状态；TTS reference base cache 按既有约定保留。明确新 reference 时重新初始化，实际流开始时从 reference base clone vocoder stream/hift state。
- API 提供 reference 或配置默认 reference 的路径均可到达 HF；默认缺失时允许 HF 用 system/输入音频 fallback 或已有初始化 reference，这一行为已在 HF 文档明示。
- `prompt_inputs` 是末尾可选参数；旧 `streaming_generate` 调用仍走原 BOS/cache 分支；没有改变原 positional ABI。
- 共享 HF 方法确实有两项语义修正，不能笼统称“没有任何共享改动”：TTS semantic normalization 遵循 checkpoint 的 L2/LayerNorm 配置；TTS RoPE 兼容 `rope_theta` 与 `rope_parameters`。前者调用 HF 已有 `_normalize_projected`，后者仅在 reindex 真需要而两处都缺失时拒绝。
- ordinary/FC capability 的调度代码未在本次 diff 中修改；本次保留了 accel 独立边界。

## 已有证据核对

`analysis-runs/20261014-hf-simplex/final/`：

- `probe-summary.json`：12 项均 pass，覆盖文字/语音流与非流、完整 history 回忆、HD图、输入音频、视频+语音、长语音、1-token cap、文字 thinking。
- `mobile-api-summary.json`：mobile helper 实际音频/视频 payload 各 2 units 成功；均 listen，不可外推 speaking 性能或真实窗口裁剪。
- `cancel-summary.json`：文本/语音 cancel-to-closed 约 0.052s/0.311s；后续新请求成功。
- `boundary-summary.json`：speech+thinking 与 max_new_tokens=0 明确拒绝，拒绝后新请求成功；raw socket disconnect 后可恢复，但样例从断开到下一请求完成约 3.386s/8.668s。
- Demo contract、HF options/边界、移动 CPU 行为测试的覆盖与 docs 描述相符。

## 仍需准确描述的限制（非本轮发现的阻断缺陷）

1. 没有真实手机/浏览器操作验证、主观音质评分，也没有与旧 main 或 native/accelerated 全量精度对齐；实际 API 的 PCM 非空与 finite 不能证明音质。
2. 目前单工 max input 默认 8192，不包含 128k 单工声明。
3. 当前 streaming speech 限定单 conversation、非 thinking、有 TTS template；teacher forcing/batched speech stream 明确拒绝。
4. Producer 的普通异常传播做了源码核对，但本轮 evidence 中的错误样例是请求校验失败，并非直接对 LLM producer 故障注入；不要称 producer 故障注入已通过。
5. `min_new_tokens` 在新 speech-stream 路径被移除，桌面/API本轮未暴露该选项；本次不能声称所有原生 HF generate kwargs 均在语音流中实现。
6. 取消等待生成安全退出；GPU/内核永久挂起、进程终止等故障不是当前正常关闭保证的范围。
7. 此处没有新增 FC 完整回归；既有 FC 的验收仍以其原报告为准。

## 收尾

独立 review 已完成。只创建此报告，没有代码修改、后台进程或占用资源；最终合入/对用户验收表述由 root 负责。
