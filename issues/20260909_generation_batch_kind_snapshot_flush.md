# 20260909: generation delta 攒批按开攒时 kind 快照归属(修复方案)

## 结论(先说清楚)

**bug 确认存在**,位于 `py_backend/fc_duplex_runtime.py`。修复采用「方案 B」:把攒批从语义层
下沉到传输层的语义无关 coalescer。本文是修复前的方案登记,避免遗忘,也是修复依据。

---

## 1. bug 现象

服务端把连续 delta 攒一小批再发 WebSocket。以前 flush 时看的是「此刻」的 kind,
攒的却是「刚才」那个 kind 的文本。产物:

- 失败模式 A(错归类):batch 里已有 think 文本,模型切到 tool_call 后触发 flush →
  按 live kind 误发 `response.tool_call.delta`,think 的字打进 tool_call 事件(或反过来)。
- 失败模式 B(崩溃):think 已结束、`_current_block_kind` 被清空,稍后 flush 走
  `else` 分支直接 `raise RuntimeError("non-spoken text delta has no active semantic message")`。

## 2. 根因

`_flush_generation_batch()` 在 flush 时刻读 `self._current_block_kind`,而 batch 里的
`steps` 是开攒时刻就累积好的。kind 不在 batch 状态的「开攒快照」分组键里,而是挂在独立的
`_current_block_kind` 上,由 `_begin_block` / `_clear_current_block` 各自维护。

- 分组边界只覆盖 `stream_id` / `track`(spoken↔non_spoken)/ `unit_index`(fc_duplex_runtime.py:1019-1027),
  **缺 kind**。
- think→tool_call 切换发生在同一 stream、同一 track(均 non_spoken)、同一 unit 内,
  三个维度都没变 → 不触发预冲;batch 也可能未达 16 条 / 50ms 阈值 → 旧批留着而 kind 已换。

本质:**传输层(攒批)越界去读语义层(block kind)的 live 状态**,缓存的是过去、标签是现在。

## 3. 修复方案(方案 B' — batch 自带 kind 快照)

> 注:原方案 B(把攒批下沉成语义无关 coalescer、改为单步发射)受 wire 契约约束——
> 现有测试与 resume canonicalizer 都把「一个 delta 事件含多步 steps」当协议契约
> (看 `tests/test_fc_duplex_runtime_resumable_events.py:792` 断言单事件含两 step)。
> 因此保留攒批,改为对 batch 增加 kind / tool_call_id 的开攒快照。二者本质一致:
> **batch 决定自身身份,不读 live 状态**。

### 3.1 思路

- 开攒时(`_record_generation_steps` 首次建立 batch),把当时的 `_current_block_kind`
  与 `_current_tool_call_id` 一并快照进 batch 状态(`_generation_batch_kind` /
  `_generation_batch_tool_call_id`)。
- `_flush_generation_batch()` **只用这份快照**决定 event_type 与 `tool_call_id`,不再读 live
  `_current_block_kind` / `_current_tool_call_id`。
- 分组边界补齐 kind:开攒后若 live kind 变了(`_current_block_kind != _generation_batch_kind`),
  先冲旧批再开新批(与现有 stream_id/track/unit_index 变化即冲的规则一致)。
- block 生命周期边界显式冲批:`_begin_block`(新 kind 进入)与 `_emit_close_for_span`(闭旧后
  清 kind)之前各加一次 `await self._flush_generation_batch()`,保证「旧 kind 的尾巴文本」在
  新 block 打开 / 旧 block 关闭前就已经带着正确 kind 发出去,顺带修掉「`think.end` 后还漏一个
  `think.delta`」的顺序违例。

### 3.2 关键性质

| 维度 | 现状 | 方案 B' |
|---|---|---|
| 谁决定 delta 属于哪个 segment | flush 时读 live `_current_block_kind` | 开攒时快照进 batch |
| batch 层知道什么 | think/tool_call/spoken + `tool_call_id`(读 live) | 各自快照,只看自身 |
| 边界维度 | stream/track/unit(漏 kind) | + kind(隐含覆盖 tool_call_id) |
| 顺序保证 | 靠 flush 时机兜底,漏 block 边界 | block 边界显式冲批 |
| 协议/客户端适配 | — | **零改动**(见 §4) |

### 3.3 代码改动范围

- `py_backend/fc_duplex_runtime.py`:
  - `__init__`:新增 `_generation_batch_kind: Optional[str] = None`、
    `_generation_batch_tool_call_id: Optional[str] = None`。
  - `_record_generation_steps`:开攒分支(1028-1033)同时快照 kind 与 tool_call_id;
    在「source_step_count 校验 / 追加 step」前,若 live kind 与 batch kind 不同 → 先 `_flush_generation_batch()`。
  - `_flush_generation_batch`(1073):改用 `self._generation_batch_kind` /
    `self._generation_batch_tool_call_id`,不再读 live 状态;flush 后一并清空快照。
  - `_begin_block`(1297)开头、`_emit_close_for_span`(1335)开头(闭旧清 kind 前)各加
    `await self._flush_generation_batch()`。
  - `_clear_current_block`(1402)不清 batch(只清 block),避免误清快照。
- 新增/调整测试:`tests/test_fc_duplex_runtime_resumable_events.py` 或新文件:
  - think→tool_call 切换时 delta 归属正确(think 文本不进 tool_call.delta)。
  - `think.end` 之前所有 delta 已发出(顺序正确)。

### 3.4 明确不动的东西

- 对外 WebSocket 事件集合 `response.*` 与字段形状,保持协议 v3 契约。
- `core/fc_duplex_resume.py` 的 v3 canonicalizer(`_build_fc_duplex_semantic_v2_resume_plan`)
  消费 `response.think.delta`/`tool_call.delta`/`spoken.delta`,不读 batch_index/step_batch,
  故不受影响。已验证。

---

## 4. 协议与客户端影响

- **协议不改,客户端不用重新适配。** 事件类型、字段不变,delta 内容只是从「攒批后一次发出」
  变为「单步事件 + 传输层自动合并」;对客户端来说,收到的事件序列按协议文档即正确、确定。
- 唯一变化是行为从**不确定**变**确定**(修好后同一输入必然同一正确顺序),且顺带修掉了
  「`think.end` 后还漏一个 `think.delta`」的顺序违例 → 更贴合协议文档,而非破坏兼容。
- semantics 层其它(block begin/end、tool call done、non_spoken end、resume)不受影响。

---

## 5. 验收标准

1. 复现用例(模型在 think 块尾部切到 tool_call)不再把 think 文本发成 `tool_call.delta`。
2. think 结束时,`response.think.end` 之前已完成所有 `response.think.delta`。
3. 现有 `tests/test_fc_duplex_semantic_v2.py`、`test_fc_duplex_semantic_v3_prepare.py`、
   `test_fc_duplex_runtime_resumable_events.py` 等全绿。
4. 协议 v3 对外事件集合不变(resume 仍可 consumption)。
