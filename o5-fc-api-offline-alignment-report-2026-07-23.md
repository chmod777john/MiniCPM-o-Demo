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

## 代码改动

当前关键提交：

```text
efb321107f2f11036d20002321e6ad5253bfa8fe
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
