# O5 FC warm/cache 模型忠实等价性门禁（2026-07-28）

## 目标

验证 startup full-prepare warmup 和 prompt cache 跨 Session 复用只改变服务就绪时延，
不改变模型 logits、greedy token、完整输出序列或 KV 长度。

固定条件：

```text
checkpoint: 629255 REF-AUDIO-001 Full4850 LLM+TTS Step400
SDK: 0.0.5 / O5 / 248168 rows
deployment: TP2 + LLM Graph
decode: greedy
input: sess_670bbd846d9d 的15个原始1秒 Live audio Unit
code: ee93a92
```

## 三条路径

```text
cold
  FC_DUPLEX_STARTUP_WARM=0
  FC_DUPLEX_PROMPT_CACHE_REUSE=0
  Job 632350

startup warm + cache reuse
  FC_DUPLEX_STARTUP_WARM=1
  FC_DUPLEX_PROMPT_CACHE_REUSE=1
  Job 632413

same-process cold repeat
  FC_DUPLEX_STARTUP_WARM=0
  FC_DUPLEX_PROMPT_CACHE_REUSE=0
  同一进程连续回放2次
  Job 632474
```

每个 Session 保存前15个 Unit 的 spoken/non-spoken 首步完整 float32 logits，共30个 tensor，
Session close 同时保存完整 `output_ids`、Unit 数和 KV 长度。

## 结果

### Prompt cache reuse

同一 warm 进程的第一次实际 Session 与第二次 cache-reuse Session：

```text
logits exact equal: 30/30
argmax equal:       30/30
max_abs_diff:       0
output_ids equal:   true
output token count: 777 / 777
KV length:          847 / 847
Unit count:         15 / 15
Session created:    0.976s / 1.246s
```

结论：相同 reference WAV 的 prompt cache 复用逐元素等价。

### Startup warm

同一 cold 进程中，第一次 cold Session 与完成全路径 warm 后的第二次 Session；两次均强制
重建 prompt cache：

```text
logits exact equal: 30/30
argmax equal:       30/30
max_abs_diff:       0
output_ids equal:   true
output token count: 777 / 777
KV length:          847 / 847
Unit count:         15 / 15
Session created:    28.022s / 1.289s
```

结论：APM/LLM/TTS 热路径执行历史不改变同进程模型输出；startup warm 可以把首次等待移到
Backend ready 之前。

### 跨进程数值差异

不同 TP2 进程之间，即使都关闭 startup warm：

```text
logits exact equal: 0/30
argmax equal:       30/30
max_abs_diff:       3.890625
```

cold 进程与 startup-warm 进程之间：

```text
logits exact equal: 0/30
argmax equal:       30/30
max_abs_diff:       3.3515625
```

因此跨进程非 bitwise 现象不是 warmup 引入；相同现象在 cold-vs-cold 中更大。模型忠实门禁
必须优先使用同进程 A/B 判断生命周期改动，再单独报告 TP2/BF16 跨进程数值稳定性。

## 判定

- **保留 startup full-prepare warmup**：同进程30/30 logits 逐元素一致，首次 Session
  `28.022s → 1.289s`。
- **保留相同 prompt 的 cache reuse**：30/30 logits、完整 token、KV 全部逐元素/逐项一致。
- **不将 warm/cache 视为效果补丁**：它们没有改变模型输入、logits、argmax、协议 token
  或 KV，只移动确定性初始化成本。
- **继续保留模型忠实边界**：不增加 speak/tool bias，不吞协议违规，不用推理改模型 margin。

机器证据：

```text
/user/sunweiyue/lib/swy-dev/tmp/o5-fc-root-cause/equivalence/
├── cold/
├── warm/
├── cold_repeat/
└── comparison.json
```
