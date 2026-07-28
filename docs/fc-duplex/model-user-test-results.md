# O45/O5 FC 用户测试结论

本文件是统一 FC Demo 的用户测试结论单一记录入口。训练血缘和权重路径仍由 Training
模块维护；这里只记录实际部署、用户体验和后端证据。

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

## 待测试模型

转换完成、尚未部署：

```text
625366  pre-REF Full4850 LLM-only Step400
625367  pre-REF MVP100 LLM+TTS Step100
629252  REF-AUDIO-001 MVP100 LLM-only Step100
629253  REF-AUDIO-001 Full4850 LLM-only Step400
629254  REF-AUDIO-001 MVP100 LLM+TTS Step100
629255  REF-AUDIO-001 Full4850 LLM+TTS Step400
623666  Strict Treatment Step2000
621851  Mixed Pilot Step3000
```

部署约定：

- 同一时段最多并行部署4个 O5 TP2 模型，每个模型占2张 A100。
- 每次向用户提供 URL、profile、训练 Job 和部署 Job。
- 收到反馈后先更新本文件，再关闭对应服务或进入下一批。
