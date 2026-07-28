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
deploy_job: 632505
code_commit: ee93a92
url: https://47.95.219.248:7001/fc_board
```

当前服务遵循模型忠实基线：O5 与 O45 一样不吞
ordinary-before-opener；模型协议违规会原样失败并保留 raw trace，不再为了页面“看起来
正常”隐藏输出。

机器验证：

- O5 完整 prepare 在 Backend ready 前预热14.6秒。
- 最终服务首个浏览器 Session 创建耗时1.003秒，不再出现约27秒无响应。
- `generate_audio=true` 的46 Unit 回放产生完整 spoken text、7个 audio event、
  1次 think 和正确工具调用，warning/error 均为0。
- Session close 已成功落盘原始 O5 token trace，不再抛 `dump_trace`
  `NotImplementedError`。
- 同进程 startup warm 与 cold、prompt cache reuse 与 rebuild 均达到30/30首步 logits
  exact、完整 output IDs/KV exact；预热只移动初始化成本，不改变模型输出。

前述10/10是修复前的用户体验结论；最后4个 Session 当时没有越过首次 prepare，不能用于
否定对应 checkpoint 的 generation 能力。当前只对 `629255` 开放修复后 Live 复测。

### 修复后 Live 复测（2026-07-28）

用户共完成6次有实际音频输入的 Live Session：

```text
1/6 成功
5/6 未响应
```

成功 Session `sess_55c3dd3428f3`：

- 38个1秒输入 Unit；
- spoken 回复：`没问题，你说到哪种动物我就给放到画板上。`；
- 正确调用4次工具：`小狗`、`小猫`、`大狗`、`波偶猫`；
- 无 warning/error。

5个未响应 Session：

```text
sess_670bbd846d9d  15 Units
sess_0aec03778364  18 Units
sess_210e0f311852  14 Units
sess_326e117bf9b5  15 Units
sess_52867741ad7f  11 Units
```

共同事实：

- 全部进入真实 generation，不再卡在 prepare；
- 每个 Unit 都生成合法 `listen + no_action`；
- 没有 `ordinary_before_opener`、连接错误或 trace 错误；
- 录音峰值为0.50–1.00，存在3–6秒连续有效语音，不是麦克风静音。

因此首次 Session 假死已修复；当前剩余问题是 checkpoint 对 Live 说法/节奏的
free-running 决策稳定性。成功样本具有“规则说明 → 模型确认 → 后续连续对象”的两阶段
结构；失败样本只有一次较短语音阶段。不得用 runtime 强制 speak/tool-call 掩盖该问题，
应把这6条录音固化为训练/评测回归集。

逐 Unit logits 复核：

- 成功录音的 Unit6 `listen=speak=25.125`，处于 BF16 完全平票；
- 同一录音重放时 greedy 改选 token ID 更小的 `listen`，原4次工具调用只复现1次；
- 其余失败录音最接近的 `listen-speak` margin 为1.0，其余可到4.875；
- 5个失败录音的 `no_action-tool_call_start` 最小 margin 仍为4.125以上。

所以1/6成功属于决策 margin 过薄导致的轨迹翻转，不是服务随机丢 Session。后续 Gate
必须检查协议关键 token margin，不能只检查 top1。

### 最新主观反馈（2026-07-28）

用户继续测试模型忠实服务后反馈：

- 主观成功概率相比第一轮有所提高；本轮未固定测试次数，暂不更新 `1/6` 定量记录。
- 已能听到 AI 语音。
- 语音内容连贯、字词正确，intelligibility 正常。
- 音色弱于 O45，但该差异可能来自风洞共同基础模型的音色能力；尚未做同文本、同
  reference audio、同响度的匿名 A/B。

当前将“FC 成功率”和“TTS 音色”分成两个训练问题，不通过推理 bias、强制触发或音色
后处理改善展示。

### `629253` LLM-only + clean-base TTS 对照

```text
status: usable_with_tts_quality_regression
training_job: 629253
model: REF-AUDIO-001 Full4850 LLM-only Step400 + clean-base frozen TTS
profile: o5_629253_refaudio_full4850_llm_sdk005_step400
deploy_job: 632640
url: https://47.95.219.248:7001/fc_board
```

用户结论：

- 模型可用。
- 偶发无响应的体感与 `629255` 基本一致，说明该问题不由是否训练 TTS 决定。
- 语音不连贯，音色和韵律都弱于 `629255`，并出现读错。
- `629255` 在相同 Full4850 主任务上额外训练 TTS 后，连贯性、字词正确性、音色和韵律
  均有主观改善。

判定：本轮 SDK-native TTS training 有效。它没有解决 FC `listen/speak` margin，
但改善了模型真正进入 speaking 后的语音质量。后续不应再把 `629255` 的音色差距完全
归因于共同基础模型；更准确的结论是 TTS 训练已明显改善基础音色，但相对 O45 仍有差距。

### MVP100 与 Strict Treatment 模型忠实复测

`629252` 和 `629254` 均能创建 Session，并在首个 Unit 正常生成 spoken `listen`；
随后 non-spoken 首 token 违反 SDK 协议：

```text
629252:
  repeated token_id: 874
  decoded ordinary: " no"

629254:
  observed token_id: 20699 / 874
  decoded ordinary: " spoken" / " no"

expected:
  <|no_action|> / <think> / <tool_call> / non-spoken terminator
```

模型忠实 View 与 O45 一样对 ordinary-before-opener fail-fast，因此 Session 约1秒关闭。
这不是启动、网络或 parser 故障，而是两个 MVP100 step100 checkpoint 没有稳定学会
non-spoken opener/control token。禁止恢复“warning 后吞 token”的体验补丁。

`623666` Strict Treatment Step2000：

- FC 功能正常；
- 偶发无响应体感未单独量化；
- 语音质量较差，与未训练 TTS 的 `629253` 接近。

该结果进一步支持：Agent/FC 能力与 TTS 音质是独立训练轴；没有 SDK-native TTS
supervision 的 checkpoint 即使 FC 功能可用，语音仍可能不连贯、音色/韵律较差。

部署约定：

- 同一时段最多并行部署4个 O5 TP2 模型，每个模型占2张 A100。
- 每次向用户提供 URL、profile、训练 Job 和部署 Job。
- 收到反馈后先更新本文件，再关闭对应服务或进入下一批。
