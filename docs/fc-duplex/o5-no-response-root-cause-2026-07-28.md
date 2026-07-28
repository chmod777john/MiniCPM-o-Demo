# O5 FC 实时不响应根因分析（2026-07-28）

## 结论

当前证据不支持“SDK 0.0.5 特殊 token ID 全局解析错误”或“LLM Graph 导致 O5
完全失效”。已经确认的首要问题是：

> O5 首个 `generate_audio=true` Session 把 TTS / Token2Wav 的重初始化放在
> Session prepare 热路径中。4个最后验收服务都在约27秒的 prepare 尚未完成时被关闭，
> 因而没有任何 Unit 进入 spoken / non-spoken generation。

这会在 UI 上表现为“模型不说话、不调用工具”，随后 Session close 又触发
`dump_trace` 不支持错误，使现象看起来像模型生成失败。

同时存在第二类现象：个别已经完成 prepare 的 O5 Live Session 会生成
think/tool-call opener 之前的普通 token。Semantic API 不能把这些 token 无损归类，
因此按协议发出 `unclassified_non_spoken_token` 并丢弃。这是真实的协议输出异常，但
它与“首个 Session 卡在 prepare”不是同一个问题。

## 已确认事实

### 1. 10个 O5 用户验收均失败

用户观察到的共同表象：

- 不产生 spoken 输出；
- 不产生工具调用；
- 页面存在错误或 warning。

完整模型、Profile、URL 和部署 Job 见
[`model-user-test-results.md`](model-user-test-results.md)。

### 2. 最后4个 Live Session 没有进入模型 generation

4个 Session 从 Backend WebSocket accepted 到用户关闭的时间：

```text
629253: 27.43s
629255: 27.26s
623666: 27.37s
621851: 27.84s
```

这4段日志都停在 O5 `prepare()` 的 reference voice / Token2Wav 初始化：

```text
fixed s3 encode
prompt_speech_tokens_lens tensor([175], ...)
prompt_mels.shape torch.Size([1, 350, 80])
prompt_mels_lens tensor([350], ...)
```

在 Session close 前没有出现 `fc_spoken` 或 `fc_non_spoken_step`。因此这4次测试只能判定
服务的首次交互不可用，不能判定对应 checkpoint 已经完成 free-running generation 且
模型选择始终沉默。

### 3. 阻塞点位于 O5 专属内层

O5 `FcDuplexCapability.prepare()` 在 `generate_audio=true` 时同步执行：

```text
model.init_streaming_processor()
→ _init_token2wav_cache()
  → model.init_token2wav(...)（首次）
  → audio_tokenizer.set_stream_cache(prompt_wav_path)
→ _reset_token2wav()
→ system/reference-audio LLM prefill
```

该过程发生在 Worker 已注册、HTTP health 已返回 healthy 之后。健康检查没有覆盖
“首个可生成音频的 FC Session 已 warm”这一条件。

O45FC 使用相同外层 Gateway、Worker、Semantic API 和 Unit scheduler，但没有观察到同样
的约27秒首次 prepare 阻塞，因此问题边界位于 O5 TTS/Token2Wav 生命周期，而不是共享
Realtime API。

### 4. SDK 0.0.5 特殊 token 能正确端到端工作

正式 O5 target：

```text
base rows: 248144
extension ids: 248144..248167
required model rows: 248168

ai_spoken_slot_start:     248159
ai_spoken_slot_end:       248160
ai_non_spoken_slot_start: 248161
ai_non_spoken_slot_end:   248162
no_action:                248163
non_spoken_eos:           248164
```

使用 `629255` canonical PT/backbone 和正式 SDK `0.0.5`，对同一个 Full4850
TrainingData case 做了两次完整 Semantic API 回放：

```text
Graph ON:  Job 631269
Graph OFF: Job 631270
```

两次均得到：

```text
46 Units
spoken: 39 listen + 6 slot_eos + 1 turn_eos
non-spoken: 40 no_action + 4 budget_reached + 2 eos
spoken text: 正常生成完整中文回复
think span: 正常打开、流式输出并关闭
tool call: display_object_on_board(name="红外感应相机")
response.warning: 0
```

Graph ON/OFF 的文本措辞和 delta batching 有轻微差异，但 spoken 状态、non-spoken
终止分布、工具名和参数一致。由此可以排除：

- SDK 0.0.5 control token ID 完全错位；
- O5 Adapter 无法识别 `<|speak|>` / `<think>` / `<tool_call>`；
- LLM Graph 必然吞掉 spoken 或 tool-call token；
- 共享 Unit scheduler 必然以错误顺序调用 spoken / non-spoken。

### 5. PT 与 TP2 backbone 没有发生词表行重排

对 `625365` canonical deploy PT 和对应 TP2 safetensors 做逐 tensor 对拍：

```text
llm.model.embed_tokens.weight: exact equal
llm.lm_head.weight:            exact equal
rows 248045..248167:           exact equal
max_abs_diff:                  0
```

因此 TP2 backbone 提取没有移动或覆盖 SDK special-token embedding / lm-head rows。

### 6. O5 modeling 与已跑通的 O5 FC 分支相同

当前 `modeling/o5/modeling_minicpmo_unified.py` 与历史 O5 FC Demo
`MiniCPMO45/modeling_minicpmo_unified.py` 的 SHA256 相同：

```text
c81eed9cb88523afb369d1f0799e215b62864c9d29d31346387d0f0c32766c74
```

`configuration_minicpmo.py`、`processing_minicpmo.py`、
`tokenization_minicpmo_fast.py`、`modeling_minicpmo.py`、`utils.py`、
`llm_graph.py`、`opt_flags.py` 也逐文件相同。

历史成功服务使用的是 SDK `0.0.5a1`、248174 rows；它不能直接证明正式
`0.0.5`、248168 rows 的 checkpoint 正确。但它能证明当前 vendored O5 modeling /
TP2 主实现不是迁移时重新手写的一套不同算法。

## 次要问题

### 1. Live 音频下出现 ordinary-before-opener

Session `sess_d353581bc822` 在 prepare 完成后处理了5个 Unit，共产生80条：

```text
code: unclassified_non_spoken_token
reason: ordinary_before_opener
message: 模型在 think/tool_call opener 前生成 ordinary token
```

这里不是 parser 把合法 `<tool_call>` ID 认错。模型 primitive 返回的原始 token 本身是
ordinary token，并且前面没有 `<think>` 或 `<tool_call>`。当前 View 丢弃这些 token，
避免把未分类文本伪装成 think/tool-call；这一策略符合公共 Semantic API 的类型边界。

仍需定位 ordinary-before-opener 的上游来源：

- Live 真人语音相对 TrainingData TTS 音频的分布差异；
- O5 training/inference post-APM embedding 一致性；
- checkpoint free-running protocol 稳定性；
- 首次 prepare 被中断后残留的模型状态。

### 2. O5 `dump_trace` 未实现

每次 Session close 都会触发：

```text
NotImplementedError: o5 FC Adapter 不支持 dump_trace
```

这是观测能力缺失，不是模型不响应的根因。但它遮蔽了最需要的原始 token / top-k
证据，应单独修复。

### 3. 重复连接错误

少数快速重连触发：

```text
backend already has an active session
unsupported runtime message type: session.init
```

这是前端重连与单 Session Backend 的竞态，也不是所有模型共同不输出的根因。

## 根因边界

当前按证据强弱排序：

1. **已确认 P0：O5 `generate_audio=true` 首次 Session 的 TTS/Token2Wav lazy init
   阻塞约27秒，健康检查却提前宣告服务可用。**
2. **已排除：全局 SDK 0.0.5 special-token 映射错误、TP2 vocab 行重排、LLM Graph
   必然错误、共享外层 scheduler 调用顺序错误。**
3. **待定位 P1：warm 后的 Live 真人语音会使部分 checkpoint 产生
   ordinary-before-opener；TrainingData 精确回放则 spoken/think/tool-call 全部正常。**
4. **待补 Gate：O5 training/inference waveform、feature、position 与同-policy
   post-APM 对拍，以及 DCP teacher-forced top1。Training 文档中这两项仍明确未完成。**

## 下一步验证顺序

1. 服务启动阶段完成 O5 Token2Wav/TTS warmup，warmup 未通过时 Worker 不注册 ready。
2. warm 后回放同一段用户 Live 录音，确认 spoken/tool-call 是否恢复。
3. 为 O5 Capability 增加原始 generated token ID、display name、top-k logits 和
   classification trace；不要只记录被 View 丢弃后的空文本。
4. 用同一 checkpoint、同一 waveform 做：
   - Training/Megatron full-forward teacher-forced；
   - HF eager incremental；
   - TP2 Graph OFF；
   - TP2 Graph ON。
5. 只有四路在首个分叉点对不上时，才修改对应 inference 内层；不要通过放宽 parser
   把 ordinary token 强行伪装成合法 think/tool-call。
