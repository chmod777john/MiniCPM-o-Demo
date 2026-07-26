# O5 FC API 与离线评测对齐排查报告（2026-07-23）

## 背景

目标是确认当前 O5 FC API 路径是否能复现 HWQ / train-data offline 评测路径的行为。选用 TauVoice 单 case 做对齐：

- case: `/user/heweiquan/dataset/O5_Dubplex_FC/overfit100/tau_function_call_sft_v13_think_cleaned_sample_2_colloquialize_tts_atom_export_relation_01029.json`
- 权重: `/user/weihongliang/o5_weights/iter_0001500_o5_with_tts.pt`
- 当前分支: `o5-no-fc-speedup-tp2-llmgraph-narrow-2026-07-23-fc-api-2026-07-23`
- 当前 commit: `efb321107f2f11036d20002321e6ad5253bfa8fe`
- HWQ 对照分支: `hwq_o5_duplex_fc`
- HWQ commit: `9209d29551301420a6cd9b9214f1182895b56e65`

这个 case 的语音问题是：把 `255 + 128` 的结果转成二进制。GT 中有两个工具调用：

1. `convert_decimal_to_binary(255)` -> `11111111`
2. `convert_decimal_to_binary(383)` -> `101111111`

## 最初现象

HWQ offline train-data 路径能预测两个工具调用。当前分支直接跑 offline train-data 路径也能预测两个工具调用。

但是 live API / TP2 路径最初只产生一个工具调用：

```text
convert_decimal_to_binary(383)
```

并且早期 probe 中模型没有继续说出最终 binary 结果。后来确认这是两个问题叠在一起：

- tool result payload 字段不匹配，probe 发的是 `content`，runtime 读的是 `contents`。
- API 在线调度没有复刻 offline train-data arrangement。

修正 `contents` 并补足 silent units 后，API 能说出 `101111111`，但仍然只产生 `383` 一个 tool call。

## Offline 路径行为

当前分支新增了最小 offline probe：

```text
scripts/probe_fc_train_data_offline.py
```

它调用：

```python
fc.offline_inference_from_train_data(...)
```

该路径会读取 train-data arrangement，并使用：

- arrangement 构造出的 unit audio chunks
- listening / speaking 两套 non-spoken budget
- train-data 中的 tool call id
- tool response schedule

重新跑当前分支 offline 后，结果目录：

```text
/user/weihongliang/fc_align_case_01029/ours_iter1500_auto_rerun_20260723_134745
```

关键结果：

```text
pred_tool_calls:
  convert_decimal_to_binary(255)
  convert_decimal_to_binary(383)

pred_spoken_text:
  Sure! So, 255 plus 128 is 383. And the binary representation of 383 is 101111111.
```

其中 offline 的 tool call 发生在 unit 8 / unit 9：

```text
unit 8: convert_decimal_to_binary(255)
unit 9: convert_decimal_to_binary(383)
```

## 排查过程

### 1. 先验证权重和基础 FC 能力

同一个 iter1500 权重在 HWQ offline 和当前分支 offline 都能预测两个工具调用。因此排除：

- 权重本身不支持 FC
- 当前分支基础 FC primitive 完全损坏
- TP2 之前的 checkpoint 加载问题

### 2. 修正 tool result payload

`py_backend/fc_duplex_runtime.py` 的 `queue_tool_result()` 读取：

```python
payload.get("contents")
```

但早期 probe 发的是：

```json
{"content": "..."}
```

修正 probe 同时发送 `contents` 后，API 能在收到 `383` 工具结果后继续生成最终答案。

### 3. 增加 API arrangement 调度能力

为了让 API 尽量复刻 offline，给 `FcDuplexSessionRuntime` 增加了 debug / probe 级调度能力：

- `non_spoken_budgets_while_listening`
- `non_spoken_budgets_while_speaking`
- `force_listen_units`

同时给 `scripts/probe_fc_tauvoice_backend.py` 增加：

- `--budget-units-info`
- `--normalize-tools`
- `--no-generate-audio`
- `--force-listen-units`

仅靠这些调度参数，一开始仍然无法得到两个 tool call。这说明差异不只是 budget。

### 4. 插 prepare / unit trace

继续增加轻量 trace，不抓大 tensor，只记录：

- prepare prefill length / hash / render head
- 每个 unit finalize 的 spoken ids / non-spoken ids / closed spans

这一步定位到第一处分叉发生在 prepare 阶段。API 的 prepare render 开头为：

```text
<|im_start|><|audio_start|><|audio_end|>You are provided with function...
```

而 offline 无 ref audio 时不应该出现：

```text
<|audio_start|><|audio_end|>
```

API prepare hash 当时是：

```text
prefill_len=226
prefill_sha=7319a2504fe0e215
```

去掉默认 ref audio 后变为：

```text
prefill_len=224
prefill_sha=ac24ac25d87dd4f4
```

这才和 offline 的无 ref audio 语义一致。

## 根因

根因是 `PyTorchBackend.fc_duplex_prepare()` 在 FC 路径中错误 fallback 了 backend 默认 ref audio。

原逻辑：

```python
ref_audio_path=ref_audio_path or self.ref_audio_path
prompt_wav_path=prompt_wav_path or ref_audio_path or self.ref_audio_path
```

即使 API 请求里 `generate_audio=false`，只要 backend 有默认 `self.ref_audio_path`，FC prepare 仍会插入默认 ref audio，导致初始 token stream 多出：

```text
<|audio_start|><|audio_end|>
```

这个差异会从 unit 0 开始改变 hidden state，后续再怎么调整 budget 都无法完全对齐 offline。

修复后逻辑：

```python
effective_ref_audio_path = ref_audio_path or (self.ref_audio_path if generate_audio else None)
effective_prompt_wav_path = prompt_wav_path or ref_audio_path or (self.ref_audio_path if generate_audio else None)
```

也就是说，只有 `generate_audio=true` 时才使用默认 ref audio。FC offline / no-audio 对齐场景不再隐式插入 ref audio。

## 最终验证

修复后，用 API 路径按 offline arrangement 调度运行：

```text
scripts/probe_fc_tauvoice_backend.py
  --normalize-tools
  --budget-units-info /user/weihongliang/fc_align_case_01029/ours_iter1500_auto_rerun_20260723_134745/units_info.json
  --force-listen-units 14
  --no-generate-audio
```

最终 API 输出：

```text
tool_calls:
  convert_decimal_to_binary(255)
  convert_decimal_to_binary(383)

spoken_text:
  Sure! So, 255 plus 128 is 383. And the binary representation of 383 is 101111111.
```

对齐摘要文件：

```text
/user/weihongliang/fc_align_case_01029/alignment_summary_api_matches_offline_20260723.json
```

摘要中记录：

```text
offline_tool_units:
  unit 8: convert_decimal_to_binary(255)
  unit 9: convert_decimal_to_binary(383)

live_tool_units:
  unit_009: convert_decimal_to_binary(255)
  unit_010: convert_decimal_to_binary(383)

offline_spoken_text == live_spoken_text
```

tool event 的 unit 编号有 1 个 unit 的偏移，主要是事件发出 / closed span 标记的 input_id 口径不同；工具调用顺序、参数和最终 spoken text 已经对齐。

## 第二条 case 交叉验证

为避免只在单一 binary case 上过拟合，又选了一条工具链更长的 TauVoice case：

```text
/user/heweiquan/dataset/O5_Dubplex_FC/overfit100/tau_function_call_sft_v13_think_cleaned_sample_2_colloquialize_tts_atom_export_relation_00025.json
```

用户请求是给室内 herb garden 浇 500ml 水并检查土壤湿度。GT 工具调用有 4 个：

```text
fetch_container(volume=500)
fill_container_with_liquid(container_id=container_12345, volume=500)
water_plant(plant_id=herb_garden_001, container_id=container_12345)
measure_soil_moisture(plant_id=herb_garden_001)
```

当前分支 offline 路径先跑通，结果目录：

```text
/user/weihongliang/fc_align_case_00025/ours_iter1500_auto_20260723_142145
```

API 路径使用相同 offline-like 调度，并通过 probe 回放 train-data 中的 GT tool responses。结果文件：

```text
run-logs/fc_tauvoice_00025_api_align_20260723_142724.json
/user/weihongliang/fc_align_case_00025/alignment_summary_api_20260723.json
```

API 输出的工具调用顺序和参数为：

```text
fetch_container({'volume': 500})
fill_container_with_liquid({'container_id': 'container_12345', 'volume': 500})
water_plant({'plant_id': 'herb_garden_001', 'container_id': 'container_12345'})
measure_soil_moisture({'plant_id': 'herb_garden_001'})
```

这与 offline 预测的工具调用语义一致。该 case 的 spoken text 没有逐字一致；offline 与 API 都是自然语言生成，评测关键点是 FC tool calls 的顺序与参数。这个交叉验证说明修复不是只对 `01029` 单个 case 生效。

## 带音频速度测试

在第二条 case 上继续跑了带音频生成的 API timing probe。测试命令保留 `generate_audio=true`，使用 TP2 + LLM graph + FC API + TTS/token2wav 端到端链路。

结果文件：

```text
run-logs/fc_tauvoice_00025_timing_audio_20260723_143057.json
/user/weihongliang/fc_align_case_00025/timing_audio_summary_20260723.json
```

该次运行共统计 40 个 unit。分段耗时如下：

```text
prefill:
  avg 197.6 ms
  p50 175.5 ms
  max 369.2 ms

spoken:
  avg 185.0 ms
  p50 31.1 ms
  max 668.3 ms

non_spoken:
  avg 211.8 ms
  p50 46.3 ms
  max 485.9 ms

finalize:
  avg 26.9 ms
  p50 14.8 ms
  max 45.2 ms

unit total:
  avg 624.1 ms
  p50 665.5 ms
  max 1144.9 ms
```

最慢的几个 unit：

```text
unit_023 total 1144.9 ms
unit_024 total 1006.1 ms
unit_025 total 932.1 ms
unit_014 total 922.6 ms
unit_013 total 889.2 ms
```

带文本输出的 speaking units 中，TTS 相关开销大致为：

```text
cost_llm:        约 0.38-0.67 s
cost_tts_prep:   约 0.006 s
cost_tts:        稳态约 0.113 s
cost_token2wav:  典型约 0.18 s，首个有音频输出 unit 可到约 0.31 s
```

从实时性角度看，平均 unit total 约 0.62s，多数 unit 在 1s 内；最慢 unit 约 1.145s，边界上略超过 1s budget。主要耗时来自：

- speaking unit 的 LLM 生成与 TTS/token2wav；
- non-spoken budget 为 30 的工具/思考 unit。

因此当前 TP2 + graph + TTS 链路已经接近实时要求，但仍有少数 tail latency 需要继续优化，尤其是 spoken LLM 和 30-budget non-spoken 阶段。

## 代码改动

当前关键提交链：

```text
efb321107f2f11036d20002321e6ad5253bfa8fe  # FC API/offline arrangement 对齐与 ref audio 修复
af041e4c3a54b019a386e80bf54debe66232fbe8  # GT response replay 与 timing probe
```

主要改动：

- `core/processors/pytorch_backend.py`
  - 修复 FC prepare 在 `generate_audio=false` 时隐式 fallback 默认 ref audio 的问题。
- `py_backend/fc_duplex_runtime.py`
  - 支持 per-unit listening/speaking budget。
  - 支持 debug `force_listen_units`。
  - 增加 prepare / unit finalize trace。
- `scripts/probe_fc_tauvoice_backend.py`
  - 修正 tool result `contents`。
  - 支持 offline-like probe 参数。
  - 支持回放 train-data GT tool responses。
- `scripts/probe_fc_train_data_offline.py`
  - 当前分支最小 offline train-data runner。
- `scripts/summarize_fc_alignment.py`
  - 汇总 offline / live API tool calls 和 spoken text。

## 结论

这次问题不是权重问题，也不是 FC 模型基础能力问题。真正关键差异是：

1. API probe 的 tool result payload 字段需要用 `contents`。
2. API 要复刻 offline，需要显式对齐 unit budget / spoken gate / tools normalization 等调度条件。
3. 最关键根因是 FC API prepare 在 `generate_audio=false` 时仍隐式插入默认 ref audio，导致初始上下文和 offline 不一致。

修复 ref audio fallback 后，在按 offline arrangement 调度的 API probe 下，可以复现两个工具调用和相同 spoken 输出。

## 后续建议

短期建议保留这套 probe 和 summary 脚本，用于后续 FC / TP2 / graph 改动的回归测试。

如果要把 offline-like 调度变成正式 API 能力，需要进一步整理：

- 明确哪些参数只是 debug / eval 对齐用，例如 `force_listen_units`。
- 将 per-unit budget、tool response schedule、silent tail 等参数设计成正式 eval/probe protocol，而不是普通 demo 默认行为。
- 如果后续还出现不一致，再升级到 tensor 级插桩，按 prepare、prefill、spoken generate、non-spoken generate 的 input/output token 和 logits 逐层比较。

## 前端 Demo 与 Live 行为观察

后续又在同一分支上增加了一个最小 TauVoice FC 音频 demo：

```text
static/fc-demo/fc-demo.html
static/fc-demo/fc-demo.js
static/fc-demo/fc-demo.css
static/fc-demo/cases/tauvoice_01029_user.wav
```

对应提交：

```text
b92f26778d3ed50f5a87669984a47e335fbf8202  # Add TauVoice FC audio demo
```

页面走正式 API：

```text
/v1/realtime?mode=audio
```

它支持两种输入方式：

- 麦克风实时输入。
- `发送 Case 音频`：前端直接播放并发送 `01029` 的 user wav，按 1 秒 chunk 切分，经同一条 WebSocket API 发送。

页面展示四类内容：

- spoken 回复文本与音频。
- non-spoken / think / tool-args delta。
- tool call 与前端执行结果。
- session / timing / unit 事件。

### Non-spoken 的语义

这次确认了页面展示的 non-spoken 不是前端伪造的 debug 文本，而是真实模型轨道的一部分。它能产生可执行工具调用，也能在 tool result 插回后继续基于结果推理。

例如 live session 中 non-spoken 生成：

```text
<function name="convert_decimal_to_binary"><param name="decimal_number">255</param></function>
<function name="convert_decimal_to_binary"><param name="decimal_number">128</param></function>
```

后端解析为真实 tool call，前端执行工具并回传 `tool_result.contents` 后，后续 non-spoken 能看到：

```text
- 255 in binary = 11111111
- 128 in binary = 10000000
```

因此当前 FC demo 中：

```text
non-spoken = 模型内部可执行 / 可记忆 / 可驱动工具的思考与调度轨道
spoken     = 用户真正看到和听到的回复轨道
```

这两个轨道不是等价的。non-spoken 可以已经算出答案，但 spoken 未必会把答案完整说出来。

### 静音输入问题

用前端空点“开始”以及脚本直接发送全零静音音频都复现了相同现象：即使没有用户语音，non-spoken 也可能 hallucinate 一个 Fibonacci 任务。

静音 probe：

```text
run-logs/fc_silence_https_gateway_3s_20260723_154445.json
```

输入为 3 秒全零 float32 PCM，配置与页面一致：

```text
non_spoken_budget_per_unit = 30
non_spoken_scheduling = quality
```

non-spoken 开头为：

```text
The user is asking me to predict the next 3 terms of a sequence: 1, 1, 2, 3, 5, ...
Looking at this sequence, I can see it's the Fibonacci sequence ...
```

因此这不是麦克风权限或前端展示问题，而是当前 live FC runtime 在 listen / 静音输入下仍给 non-spoken 较大 budget 后，模型会基于先验 hallucinate 任务。普通实时麦克风模式尤其容易被启动阶段的静音或环境声污染上下文。

短期规避方向：

- 前端在用户真正说话前不要发送静音。
- 或后端在 listen 且无有效语音时压低 / 禁用 non-spoken budget。
- 或引入 VAD / energy gate，只有检测到有效语音后才进入 FC non-spoken 生成。

### Case 音频经前端发送的行为

用脚本走公网 HTTPS gateway、按 1 秒 chunk 发送 `01029` case 音频，验证链路是通的：

```text
run-logs/fc_tauvoice_01029_https_gateway_1s_20260723_152317.json
```

结果：

```text
tool call:
  unit_013: convert_decimal_to_binary(383)

spoken_text:
  Hi! Sure, I can help with that. First, let's add 255 and 128 together. That gives us 383.
```

说明：

- 公网 HTTPS gateway 正常。
- `/v1/realtime?mode=audio` 正常。
- 1 秒切片本身正常。
- TauVoice 工具调用链路正常。

但前端 `发送 Case 音频` 的 live session 不一定逐次复现相同 planning。最新观察到的一次 session 中，模型没有先算 `255 + 128 = 383` 再调用工具，而是改成：

```text
convert_decimal_to_binary(255)
convert_decimal_to_binary(128)
```

然后在 non-spoken 里手算二进制加法，得到：

```text
So the result is 101111111
```

但是 spoken 轨道没有把最终答案完整说出来，后续持续静音输入下变成 listen / no_action。

对应 session 观察：

```text
unit_030 到 unit_045:
  n_audio = 10
  is_listen = true
  is_speaking = false
  non_spoken_terminator = no_action
```

结论是：补充更长静音能避免前端过早截断，但不能保证模型一定把 non-spoken 中的最终结论外化为 spoken。这个问题更接近 FC duplex 的策略 / 训练行为：内部轨道已经完成推理，spoken gate 后续选择了 listen/no_action。

### 与 offline / probe 对齐的关系

前端 demo 是面向交互体验的 live 路径，不等价于 offline-like 对齐 probe。主要差异包括：

- 前端按 wall-clock 发送 chunk，tool result 插入时机受浏览器事件和模型返回时机影响。
- case 音频结束后可以持续发送静音，可能改变后续 spoken / non-spoken 决策。
- live duplex 本身存在逐 unit 的调度不确定性，planning 可能从 `convert(383)` 漂移到 `convert(255), convert(128)`。

因此：

- 要验证 FC primitive 与权重能力，优先看 offline / offline-like API probe。
- 要验证 demo 体验，前端 case 模式能覆盖真实 API、真实 tool call、真实 tool result 插入和音频播放。
- 要修复实时体验，需要重点处理静音输入下 non-spoken hallucination，以及 non-spoken 答案未外化为 spoken 的策略问题。
