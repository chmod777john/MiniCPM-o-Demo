# 普通音视频双工选项

适用于 `O5_CANONICAL_PATH` 启用的外部 HF 路径。普通视频页面 `/omni`、
普通音频页面 `/audio_duplex` 继续使用现有 gateway / worker / View，
经 `modeling/external_o5_facade.py` 调用 HF 的 prepare、streaming_prefill、
streaming_generate 和 finalize_unit。

## 页面与 API 对应

选项在 `session.init.payload.config` 传入，普通会话不设置 FC runtime。
`session.created.session_config` 回传本会话的有效配置。

| 页面 / 作用 | config 字段 | 行为 |
|---|---|---|
| 滑窗模式 | `sliding_window_mode` | 页面 Off / Basic / KV surgery；API 另接受 HF 的 context 模式 |
| Basic 高、低水位 | `basic_window_high_tokens` / `basic_window_low_tokens` | 默认 8000 / 6000；每项最多 262144；由 HF 执行裁剪 |
| KV surgery 保留范围 | `context_max_units`、`text_sink`、`text_sliding_window` | 原样传给 HF window config；页面默认 250 / 512 / 10240 |
| Listen/Speak 开关 | `ls_decision_mode` | 开启为 binary，关闭为 legacy |
| Speak Threshold | `speak_threshold` | 页面默认 0.3；API 要求 0 < threshold < 1 |
| 视频 Fixed HD | `max_slice_nums` | 会话默认切片上限，范围 1–9；每个 input 可覆盖 |
| 视频 Strategy HD | `strategy_hd`、`strategy_hd_max_slice_nums` | 首 unit 为 1；上一 unit SPEAK 则下一 unit 用策略值，否则 1 |
| TTS 开关 | `generate_audio` | 页面 `payload.use_tts` 映射到该字段 |

已有 `DuplexConfig` 的采样、max speak、force listen、TTS temperature 仍可由 API 设置。
额外 API 字段包括 `context_previous_max_tokens`、`verify_mode`、`ls_strict_unit`、
`ls_log_prob`、`tts_repetition_penalty`。未知字段报错，不静默忽略。

策略 HD 开启后，单个 input 的切片覆盖不生效，以策略为准；fixed 模式可用运行中的
HD 按钮开关，关闭时显式传 1。`effective_max_slice_nums` 回传实际交给 HF processor
的上限，它不是实际图像 patch 数或实测视觉 token 数。

`MaxKV` 是页面的自动停止阈值，默认与上限均为 **262144（256K）**，
不是后端 KV 容量分配，也不替代 Basic 高低水位。外部 HF 会话等 finalize 完成、
读取裁剪后的 KV 再检查该阈值。高低水位仍默认 8000 / 6000，需体验 128K 时单独调整。
服务可通过 `/api/frontend_defaults` 的环境设置覆盖页面默认策略，见 DEPLOYMENT.md。

本版页面移除了 Draft/veto 和 Length Penalty 控件：当前 HF 普通生成没有对应接口。
旧客户端传 `length_penalty` 时保留警告和 `inactive_options` 回传，不声称其生效。
slot/non-slot 仍由现有 HF 启动配置决定，本次不新增页面 slot 切换。

## 会话生命周期

每次 prepare 都重新应用配置；省略的滑窗字段恢复初始化默认，LS 字段在 HF prepare
重置策略后再覆盖；binary 同时打开 HF decoder 的 fixed token-ID 选择。
HD 下一 unit 状态、滑窗事件队列也逐会话重置。
TTS generate_chunk 在 HF 内根据每次传入的 repetition penalty 构造采样处理器，
桥接层只传参数，不改写其生成过程。

## 滑窗指标

生成输出中的 metrics 继续携带 prefill/generate 时间、LS 概率和有效切片上限。
finalize 完成后，另发 `response.metrics`：

```json
{
  "type": "response.metrics",
  "session_id": "sess_...",
  "input_id": "input_123",
  "unit_finalized": true,
  "finalize_ms": 25.0,
  "kv_cache_length": 5990,
  "window_event_count": 2,
  "window_events": [
    {"count": 2, "before": 8100, "after": 5990, "mode": "basic", "duration_ms": 24.0}
  ]
}
```

上面只是字段示例，不是性能实测。事件来自 decoder 实际裁剪计数增加；不根据相邻
unit 的 KV 下降猜测。计时是 host wall time，不额外 synchronize GPU，不能当 CUDA
kernel 时间或完整 unit 时间。页面显示累计次数、前后 KV，日志附 input ID 和耗时。

## 断线恢复与长文件

普通页面主动加 `ordinary_reconnect=1`。gateway 保留同一个 worker 连接，
向 `session.created` 注入随机恢复 token；同一进程内 30 秒宽限期、16 MiB 输出缓存。
重连不重新 prepare，也不重发输入；连接断开期间浏览器停止发送新 chunk，
媒体源可能仍继续前进，因此不是离线媒体的无损续传或 exactly-once 协议。
超时、缓存溢出或原 backend 会话结束会释放保留会话。服务重启不保留该状态；
多 gateway 进程部署需要把重连路由到原进程。

新版 Gallery 的 FC 恢复协议保持原样。普通文件页面取消了音频 5 分钟、视频 10 分钟
的强制截断；浏览器仍在本地解码媒体，长文件内存开销并未消失。

## 本次交付边界

本次完成源码迁移和调用链审查，未新增或执行测试，未启动服务；浏览器、真实 API、
GPU 运行及断网回归尚未进行。此前三物料验收不能作为本次新增功能的运行证据。
来源和文件范围见 `ORDINARY_DUPLEX_MIGRATION_20261013.md`。
