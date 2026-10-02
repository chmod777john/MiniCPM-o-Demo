# FC duplex 限制解码（协议状态机）设计 —— FUTURE WORK

> 状态：**未实现**。本文记录动机、已查实的协议事实、设计决策与待办。
> 日期 2026-10-02。触发来源见下文「动机」。

## 1. 动机：模型会吐出不合协议的 token 序列

2026-10-02 跑 FDB-v3（`--deployment-mode single_eager`）时，
`travel_14_66c4f3cb14cbfc4db836bd4e` 三轮稳定失败，服务端诊断：

```
FC non_spoken stream opened before previous stream closed: think_2 -> tool_call
```

含义：该 session 的第 2 个 think span 还没闭合（没出 `THINK_END`），
模型就吐出了 `TOOL_CALL_START`。抛错点 `core/processors/unified.py:1623-1630`。

**这是模型侧缺陷，不是推理代码的 bug**，三条证据：

1. 同一条 case 在另外 5 次跑里都正常（3 次我们 `single_opt` + 同事已发布 + 我用同事代码跑），
   `single_opt` 那跑里它的 span 是规整的：`<think>` 3 次 / `</think>` 3 次 / 1 次工具调用。
   只在 `single_eager` 的数值条件下漏了 `THINK_END`。
2. 我们把 span 跨 unit 保持打开是**刻意设计且合协议**：
   `core/processors/unified.py` 的 `terminate_non_spoken_text_stream` 里写明
   「Budget 只关闭 Unit slot；semantic closer 才拥有 DecodeStream 生命周期」，
   `budget_reached` 只关当前 unit 的 slot，think span 继续跨到下一个 unit
   —— 不是我们把闭合 token 吃掉了。
3. SDK 协议里 `THINK_START/THINK_END`、`TOOL_CALL_START/TOOL_CALL_END` 是**显式成对**的
   （`minicpm_o5_sdk/protocols/duplex/special_tokens.py:71-74`），
   所以「think 未闭合就开 tool_call」确实非法。

这条 case 是「想得多又被反复截断」的类型（72 unit 里 4 个 `budget_reached`，
think 被切成多段跨 unit），模型在这种情形下更容易丢掉「我还有个 span 开着」。

## 2. 设计决策（用户 2026-10-02 明确拍板）

**只做限制解码：把绝对不应该出现的 token ban 掉，让模型在其余 token 里采样。
不注入任何 token、不猜测模型意图。**

曾提过一个「语法修复」方案（检测到非法开启符时先注入缺失的 `THINK_END` 再放行），
**已被否决** —— 理由是那会造成不连续：注入的 token 不是模型自己的分布产生的，
等于替模型编内容。ban 只是把采样投影到合法子集上，不发明任何东西。

推论（同一机制自然覆盖的另一例）：**turn 还没 eos 时就不该采样出 `LISTEN`**，
这同样是一条 ban 规则，不是特例。

### 开关语义

- API 侧（`session.init`）透传一个开关，与 `decode_mode` 同级。
- **关**：现状行为 —— 遇到非法序列抛协议错误。
- **开**：理论上永不发生协议错误。
- 开关值必须落进 `session_preset.json`，否则事后无法归因某组分数是哪套口径跑的
  （per-ckpt 事实只在 preset 里，这是 o5_benchmark 的底线）。

## 3. 已查实的协议事实

### 3.1 turn 结束的形态（权威依据：SDK 训练数据序列化器）

`minicpm_o5_sdk/protocols/duplex/tokenization/slots/ai_spoken.py:183-193`：

```python
if any_turn_final:
    sequence.append_special(O5SpecialTokenKey.SPOKEN_TURN_EOS, source="runtime", ...)
sequence.append_special(O5SpecialTokenKey.SPOKEN_SLOT_EOS, source="runtime", ...)
```

训练数据里 turn 结束就是 `SPOKEN_TURN_EOS` → **无条件紧跟** `SPOKEN_SLOT_EOS`。
我们推理侧的做法与之一致（采到 `SPOKEN_TURN_EOS` 即终止 spoken 循环，
再 `_close_spoken_slot(append_spoken_slot_eos=True)` 补 `SPOKEN_SLOT_EOS`）。

> 📌 顺带记一个外部事实：同事的 vendored 服务
> （`Full-Duplex-Bench/v3/fc_duplex_service/fc_service/worker.py`）
> `terms` 里**不含** `SPOKEN_TURN_EOS`（只记标志位、继续采到 24 步），
> 关 slot 时只喂 `AI_SPOKEN_SLOT_END`、不喂 `SPOKEN_SLOT_EOS`。
> 所以他那边从「第一次说完整句话」起 KV 就偏离训练分布了。

### 3.2 span 是显式成对的

`special_tokens.py:71-74` 有 `THINK_START/THINK_END`、`TOOL_CALL_START/TOOL_CALL_END`；
`:56-57` 另有 `TOOL_CALL_ID_START/END`。所以 span 必须显式闭合。

### 3.3 有一类 token 模型永远不该采样

SDK 序列化时带 `source="runtime"` 标记的那些（如 `NON_SPOKEN_BUDGET_REACHED`、
各种 `*_SLOT_START/END`、`UNIT_*`）是运行时写入的，**应在所有 decode step 被 ban**。
具体清单待从 SDK 逐项提取。

## 4. 实现轮廓

### 4.1 状态机独立成模块

建议 `core/fc_duplex/protocol_fsm.py`，只依赖 SDK 的 `O5SpecialTokenKey` + registry，
**不依赖推理栈** —— 这样可以不起 GPU 单独跑测试（协议语法本身是纯逻辑）。

只暴露两个东西：

```python
fsm.advance(token_id)        # 按实际采出的 token 推进状态
fsm.banned_token_ids()       # 当前状态下必须 ban 的 token id 集合
```

### 4.2 挂载点

在 `_sample` 之前把 banned 位置的 logit 置 `-inf`。
调用点在 `modeling/o5/modeling_minicpmo_unified.py` 的
`streaming_spoken_generate` / `streaming_non_spoken_generate`。

⚠️ decode 走 CUDA graph 时 logits 是从固定 buffer 读出来的，
**mask 必须加在读出之后、`argmax` 之前**，不要碰图本体。

`_sample` 本身两边（我们与同事）逐字节相同，greedy 就是
`int(torch.argmax(logits[0]).item())`，没有 temperature / top-k / top-p / 重复惩罚，
所以注入点很干净。

### 4.3 为什么 ban 不改 KV 结构

ban 只改变「被选中的是哪个 token」，**不额外喂 token**，
所以 KV 的长度与结构不变，只有内容不同。（待实现时复核确认。）

### 4.4 可复用件

- 推理侧已有的流式状态跟踪：`core/processors/unified.py` 的 `start_keys`、关闭 key 表、
  `_spoken_text_stream` / `_non_spoken_text_stream` / `_non_spoken_aggregate_kind`。
  它已经足以检出「span 未闭合就开新 span」，所以状态维度至少部分够用。
- SDK 侧值得先读的：`protocols/duplex/tokenization/validator.py`、`tokenization/span.py`
  （很可能就是权威语法）、`tokenization/slots/*.py`（正例生成器）、
  `parsing/parser.py`（很可能已经是个状态机，若有「状态 → 合法 next token」的现成函数可直接复用）。

## 5. 待办

1. **产出完整的协议状态机转移表**（状态 → 合法 next token / 必须 ban 的 token）。
   这是实现的前提，尚未完成（两次后台调研都被会话中断打断）。
2. 从 SDK 提取「只由运行时写入」的 token 全集。
3. 实现 `protocol_fsm.py` + 单元测试（纯逻辑，不需要 GPU）。
4. 接上 `_sample` 前的 mask；API `session.init` 透传开关；写入 `session_preset.json`。
5. **死角风险待查**：是否存在「ban 完合法集合为空」的状态
   （例如 budget 只剩 1 步、或某状态下只有运行时 token 合法）。
   若存在，「开了就永不协议错误」这个保证需要加兜底条款，
   **不能假装它不存在**。

## 6. 配套：ProtocolError 的呈现与评分（已与用户议定）

与限制解码配套、但可独立先做：

- 把这类错误作为 **ProtocolError** 呈现，该 case **停止推理**。
- 评测侧 `protocol_error_count += 1`，**该计数必须出现在 `summary.json` 顶层**，
  不能只埋在日志里。
- **计入分母、该 case 判失败**（Pass@1=0，工具类指标按「漏调」处理，
  WER/latency 该 case skip 但 skip 数要显式报）。
  理由：这是**模型**的锅，不是基础设施的锅。
  tau2 把 `INFRASTRUCTURE_ERROR` 排除出分母是对的（非模型责任），
  但 ProtocolError 若排除出分母，等于「吐非法序列反而不扣分」，形成反向激励。
- ⚠️ **跨人对比注意**：同事的服务没有这个校验，同样的模型失误在他那边会静默通过并照常给分。
  所以我们报 `protocol_error=1` 并判 0 分时，他那边同一条可能「有分」——
  两边的 Pass@1 从此不是同一口径，必须在报告里写明。
- 发生时自动落盘当时的 token 序列（`FC_DUPLEX_TOKEN_PROBE=1` 已有），
  作为给训练侧的证据：**模型在 budget 截断下会丢 `THINK_END`，这是数据/训练要修的缺陷，
  不该在推理侧用限制解码「刷掉」后就当不存在。**
