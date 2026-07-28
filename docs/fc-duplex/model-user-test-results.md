# O45/O5 FC 用户测试结论

本文件是统一 FC Demo 的用户测试结论单一记录入口。训练血缘和权重路径仍由 Training
模块维护；这里只记录实际部署、用户体验和后端证据。

2026-07-28 的根因复核见
[`o5-no-response-root-cause-2026-07-28.md`](o5-no-response-root-cause-2026-07-28.md)。
用户验收失败仍按原始事实保留，但最后4个 O5 Session 实际在首次 TTS/Token2Wav
prepare 完成前就被关闭，不能继续解释为“4个 checkpoint 均已完成 generation 且选择沉默”。

状态定义：

- `normal`：用户确认主要交互正常，不再重复部署。
- `failed`：用户确认关键能力不可用，需要保留证据后关闭。
- `pending`：转换已完成，尚未交给用户测试。

## 已完成测试

### O45 Full4850 LLM+TTS Step400

```text
status: normal
profile: o45_fc_board_tts_full4850_sdk005_step400
training_job: 624224
deploy_job: 629212
url: https://47.95.219.248:7003/fc_board
closed: 2026-07-28
```

用户结论：

- O45 整体表现正常。
- 不需要再次部署测试。

代表性后端证据：

- 实时 Session 能产生 spoken text。
- 能识别并调用 `display_object_on_board(name="小狗")` 和
  `display_object_on_board(name="小猫")`。

### O5 pre-REF Full4850 LLM+TTS Step400

```text
status: failed
profile: o5_fc_board_tts_full4850_sdk005_step400
training_job: 625368
deploy_job: 629157
url: https://47.95.219.248:7040/fc_board
closed: 2026-07-27
```

用户结论：

- 不输出 AI spoken text。
- 该模型已经汇报，不再继续占用服务。

补充边界：

- 该 checkpoint 属于 REF-AUDIO-001 修复前历史矩阵。
- 历史 agent 自研 DCP→PT 不再使用；canonical PT 已另行生成并登记。

### O5 pre-REF MVP100 LLM-only Step100

```text
status: failed
profile: o5_fc_board_mvp100_sdk005_step100
training_job: 625365
deploy_job: 629593
url: https://47.95.219.248:7040/fc_board
closed: 2026-07-28
```

用户结论：

- 模型不说话。
- 没有工具反馈。
- 消息列表中出现疑似模型生成错误信息。

后端证据：

- spoken track 持续返回 `listen`，没有 spoken delta。
- non-spoken 每 Unit 多次产生空文本 step，最终 `budget_reached`。
- 没有形成有效 think/tool-call span。
- 服务和 TP2 本身未崩溃，问题属于模型输出行为，不是 Worker 离线。

### O5 第二批四模型对照

公共用户结论：

- 4个模型均不说话。
- 4个模型均不调用工具。
- 页面日志中均观察到疑似错误信息。

```text
status: failed
training_job: 625366
model: pre-REF Full4850 LLM-only Step400
profile: o5_625366_full4850_llm_sdk005_step400
deploy_job: 631041
url: https://47.95.219.248:7001/fc_board
closed: 2026-07-28

status: failed
training_job: 625367
model: pre-REF MVP100 LLM+TTS Step100
profile: o5_625367_mvp100_llm_tts_sdk005_step100
deploy_job: 631039
url: https://47.95.219.248:7002/fc_board
closed: 2026-07-28

status: failed
training_job: 629252
model: REF-AUDIO-001 MVP100 LLM-only Step100
profile: o5_629252_refaudio_mvp100_llm_sdk005_step100
deploy_job: 631040
url: https://47.95.219.248:7011/fc_board
closed: 2026-07-28

status: failed
training_job: 629254
model: REF-AUDIO-001 MVP100 LLM+TTS Step100
profile: o5_629254_refaudio_mvp100_llm_tts_sdk005_step100
deploy_job: 631064
url: https://47.95.219.248:7014/fc_board
closed: 2026-07-28
```

后端证据与错误归因：

- `625366` 的 spoken 始终为空，non-spoken 直接以空内容 `no_action` 结束。
- `629254` 的 spoken 始终为空，non-spoken 连续产生空 step，最终
  `budget_reached`，与 `625365` 的失败模式一致。
- `625366`、`625367`、`629254` 在关闭 Session 时出现
  `NotImplementedError: o5 FC Adapter 不支持 dump_trace`。这是 O5 trace
  能力缺失导致的观测错误，不是模型不说话的根因。
- `629254` 另出现过一次 `backend already has an active session`，属于重复建连竞态；
  它不能解释4个模型共同不输出的现象。

### O5 最后一批四模型对照

公共用户结论：

- 4个模型均不说话。
- 4个模型均不调用工具。
- 页面日志中均观察到错误信息。

```text
status: failed
training_job: 629253
model: REF-AUDIO-001 Full4850 LLM-only Step400
profile: o5_629253_refaudio_full4850_llm_sdk005_step400
deploy_job: 631113
url: https://47.95.219.248:7001/fc_board
closed: 2026-07-28

status: failed
training_job: 629255
model: REF-AUDIO-001 Full4850 LLM+TTS Step400
profile: o5_629255_refaudio_full4850_llm_tts_sdk005_step400
deploy_job: 631114
url: https://47.95.219.248:7002/fc_board
closed: 2026-07-28

status: failed
training_job: 623666
model: Strict Treatment Step2000
profile: o5_623666_strict_treatment_sdk005_step2000
deploy_job: 631115
url: https://47.95.219.248:7011/fc_board
closed: 2026-07-28

status: failed
training_job: 621851
model: Mixed Pilot Step3000
profile: o5_621851_mixed_pilot_sdk005_step3000
deploy_job: 631116
url: https://47.95.219.248:7014/fc_board
closed: 2026-07-28
```

后端证据：

- 4个入口的 Gateway、Worker 和 TP2 Backend 均保持健康，模型和 LLM Graph 加载成功。
- 用户 Session 均正常到达 Backend，但没有形成 spoken/tool-call 输出。
- 4组关闭 Session 时再次出现 O5 `dump_trace` 不支持错误。
- `623666` 的一次重复连接出现 `unsupported runtime message type: session.init`；
  这是独立的重复建连协议错误，不能解释首个 Session 及全部模型共同无输出。

至此已登记的10个 O5 candidate checkpoint 已全部完成用户测试；10/10 均未通过，
而同一外层 API、前端和调度框架下的 O45FC 已通过。该分布强烈指向 O5 专属的模型内层
推理、特殊 token 或 adapter 边界，而不是10个训练 checkpoint 独立失效。

## 修复后待用户复测

```text
status: pending_retest
training_job: 629255
model: REF-AUDIO-001 Full4850 LLM+TTS Step400
profile: o5_629255_refaudio_full4850_llm_tts_sdk005_step400
deploy_job: 631623
code_commit: 913b1c7
url: https://47.95.219.248:7001/fc_board
```

机器验证：

- O5 完整 prepare 在 Backend ready 前预热14.2秒。
- 修复后首个浏览器 Session 创建耗时1.117秒，不再出现约27秒无响应。
- `generate_audio=true` 的46 Unit 回放产生完整 spoken text、7个 audio event、
  1次 think 和正确工具调用，warning/error 均为0。

前述10/10是修复前的用户体验结论；最后4个 Session 当时没有越过首次 prepare，不能用于
否定对应 checkpoint 的 generation 能力。当前只对 `629255` 开放修复后 Live 复测。

部署约定：

- 同一时段最多并行部署4个 O5 TP2 模型，每个模型占2张 A100。
- 每次向用户提供 URL、profile、训练 Job 和部署 Job。
- 收到反馈后先更新本文件，再关闭对应服务或进入下一批。
