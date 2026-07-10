# O5 推理加速 —— 对接说明（INTEGRATION）

一句话:MiniCPM-O5 全双工推理的一组**纯推理期、默认关闭、可一键开启**的加速。开启后 demo 每个 1s unit 都 <1s(语音 0.50s / 全模态 0.61s),对接**只需翻一个开关**;不开 = 原始行为逐字节不变。

---

## 1. 怎么开(对接同学只需这一步)

二选一,默认都是**关**:

- **环境变量**(最简单,推荐先用它验证):启动 backend 前设 `O5_OPTIMIZE=1`(docker-compose / 启动脚本里加一行)。
- **配置文件**:`config.json` 里 `model.optimize: true`。

开启后 `core/processors/unified.py` 会在 `init_unified` 之后自动调用 `MiniCPMO45.o5_enable.enable_o5_optimizations(model)`,并**自动跳过 torch.compile**(手写 CUDA graph 替代 compile,二者互斥、不能叠加)。

> 起 demo 的方式**完全不变**(`python gateway.py` / docker-compose)。TTS/LLM 的 CUDA graph 在**首个请求**时一次性捕获(~1-2s),想让首个真实请求也快,可在服务启动后先打一条 warmup 请求。

## 2. 改了什么(便于 review)

**模型包 `MiniCPMO45/`:**
- 新增:`opt_flags.py`(运行时开关,全默认 False)、`tts_graph.py` / `llm_graph.py`(TTS/LLM decode 的 CUDA-graph runner)、`vocoder_graph.py`(vocoder DiT graph)、`o5_enable.py`(一键 enable 函数)。
- 改动(全部 `OPT` 门控,默认关 = 原逻辑):`utils.py`、`modeling_minicpmo.py`、`modeling_minicpmo_unified.py`。
- 注:linear-attn 的 chunked-prefill monkeypatch **base 里本来就有**,不是本次引入。

**demo 集成(3 处小改):**
- `config.py`:`ModelConfig` 加 `optimize: bool = False`。
- `core/processors/unified.py`:`init_unified` 后加默认关的 enable 调用;`torch.compile` 块在 optimize 开时自动跳过。
- `core/processors/backend_factory.py`:把 `optimize` 从 config 传入 processor。

**验证脚本 `tools/o5opt/`:** 等价性 gate + 端到端 verify(见 §5)。

## 3. 加速效果(实测,A100-80GB,iter1200)

demo 自带 `benchmark.py`(真实 duplex pipeline,同一视频)baseline vs 开启:

| per-unit(demo 口径) | baseline | 开启后 | 提速 |
|---|---|---|---|
| listen | 2368 ms | **299 ms** | 7.9× |
| speak  | 3297 ms | **817 ms** | 4.0× |

隔离长会话口径(200 unit,每 50 unit 换 session):语音 speak 均值 **0.42s**/max 0.63s、全模态 speak 均值 **0.61s**/max 0.81s,**每个 unit <1s(0/200 超时)**,无随 cache 长度的漂移。原始 O5 是 speak ~2.1s、listen ~1.0s。根因是 A100(cc<9)上 **batch=1 的 host-sync/kernel-launch 开销**,解法是消除 host 同步 + 手写 CUDA graph。

**token 余量("每 unit 留 15-30 个 decode 文本 token"):** 单文本 token ~18.5ms(graphed LLM 前向+采样),余量=(1s−unit_time)/18.5ms。**语音每个 unit 都满足**(均值 31、最差 20)。**全模态均值 21、约 96% unit ≥15**;剩 ~4%(最差 ~10)是**长回复 unit(n_tts 24-25)+ 抖动**的结构性上限,**仍全部 <1s**(硬实时不破),非尖峰,无法再砍(除非降 omni prefill floor 或限回复长度=改行为)。

## 4. 显存(实测)

| | 峰值 | 余量(/79.14GB) |
|---|---|---|
| 权重加载后 | 70.31 GB | — |
| 原始(eager MoE + DynamicCache) | 72.01 GB | 7.1 GB |
| **开启后** | **76.31 GB** | **2.8 GB** |

开启后 **+4.3GB**(LLM/TTS StaticCache 各 8192 预分配 + 3 个 CUDA graph 池 + batched_mm 专家 gather 瞬时)。单卡 80GB **装得下但偏紧**(~2.8GB 余量,`preload_both_tts=True` 也跑通了)。要留更多余量(常量级、不影响正确性):调小 `llm_graph.py`/`tts_graph.py` 的 `max_cache_len`(按实际会话长度)、或去掉 vocoder graph(牺牲一点全模态尾部)。**注意:不要靠"开滑窗"省显存——见 §7,滑窗对 llm_graph 的 StaticCache 无效。**

## 5. 数值一致性(重要)

这些 lever 是**保 argmax 的浮点重排**(batched grouped-gemm / StaticCache / CUDA graph / eager-vs-sdpa attn),**不是逐比特一致**——这正是任务书"数值等价(允许浮点重排)"那条线。实测(全模态,pure-eager vs 全优化栈,真实 omni 路径):

- **决策(listen/speak)逐 unit 一致;每 unit 首 token argmax 一致**;
- **teacher-forced 逐 token(上下文逐位相同,200-unit 复检):argmax 80/82 一致**,分歧都是真·近似平手(top1-top2 gap 0.125 ≪ 重排噪声 ~1.0),两个候选都合法;
- 逐比特一致(0.0)的 lever:`tts_fast`/`lmhead`/`tts_graph`、vocoder 精确尺寸图 {50,302};
- **vocoder 短 flush chunk(cs<50)分桶(pad 到 50 + mask)是浮点重排 mel|Δ|=0.003(<0.01,1-3 fp16 ULP,音频不可感知)**,不是逐比特 0.0——padding 改了 matmul 归约维度,同 batched_mm 一类。

含义:**决策、分布、每步 argmax 保持**,但长贪心 rollout 里极少数近似平手 token 可能翻(都合法);**部署用采样(temp=0.7)时该差异被采样噪声完全淹没,输出分布无可观测差异**。音频波形因 TTS multinomial 采样本就每次不同,不能用它判等价。

> **换权重须知:** 上面是在 `iter1200` 上验的。三个"保 argmax 的浮点重排"lever 换到**差异很大的同架构 checkpoint** 时,理论上某个近似平手可能翻——**换权重后重跑一遍 §6 的等价 gate 确认即可,不用改代码**。只依赖模型结构(config),不读权重特定值;`vocoder{50}` 的 50 是 TTS 数据率决定的结构量。

## 6. 自己验证(可选)

```bash
cd tools/o5opt
# a) 端到端加速 + 集成不崩(跑 demo 自己的 benchmark,baseline vs O5_OPTIMIZE=1)
#    见脚本内注释设置 MODEL_PATH/PT_PATH/VIDEO
# b) 全模态等价 gate(换权重后重跑这两个):
#    ab_omni_full_equiv.py  —— pure-eager vs 全优化栈,逐 unit 决策/argmax
#    ab_omni_tf_equiv.py    —— teacher-forced 逐 token argmax(黄金标准)
# c) vocoder 分桶音频等价(换 vocoder/权重后重跑):
#    ab_vocoder_gate.py     —— eager vs 分桶逐 flush chunk 比 mel(判据 mel|Δ|<0.01)
```

## 7. 回滚 / 注意

- **回滚**:不设 `O5_OPTIMIZE` / `model.optimize=false` → 原始路径,逐字节不变。
- **不要**和 torch.compile 叠加(开启时已自动跳过 compile)。
- **本套是 batch=1 单流 duplex 专用**(静态形状 CUDA graph);实时 duplex 本就单流分时,契合;别拿去做多路并发批处理。
- **长会话 / cache 上限(重要,踩过坑):** `llm_graph` 的 StaticCache 硬顶 **8192 token**。base demo 默认 `sliding_window_mode='off'`,即会话无界增长——所以 **llm_graph 引入了一个 baseline(DynamicCache)没有的硬顶**(baseline 只是耗更多显存,不崩)。**⚠️ 滑窗对 llm_graph 无效**:StaticCache 没有 `.crop`,`sliding_window_mode='basic'` 会**静默 no-op**(实测三重确认),开了也不会裁 cache。**正确做法(三选一):**
  1. **在自然 turn 边界重置 session**(推荐;实时对话本就一问一答,reset 不丢当前对话上下文的语义,已跑 200/240-unit 验证 cache 稳定在 ~8180 峰值、sawtooth 回落、不崩);
  2. **调大 `llm_graph`/`tts_graph` 的 `max_cache_len`**(占显存,顶更高);
  3. **关 `llm_graph`**(走 DynamicCache,有真 crop 支持滑窗,但丢 LLM decode 的 graph 加速)。
  已内置 **crash-safe 守卫**:`_pos+L>max_cache_len` 时自动 reset session + warn(不崩,但会丢上下文);超长单 turn(TTS >8192 位)clamp 降级(不崩);超长首 prefill(L>8192)也 clamp。
- 权重打包若是"代码目录 + 单独 .pt"(dev 形态),demo 的 `from_pretrained` 需换成空初始化 + `load_state_dict`;正常 HF checkpoint 无此问题。

## 8. 复现 / 重建

`tools/o5opt/apply_all.sh`:从 base worktree 一键重建带全部补丁的模型包并 grep 校验每个补丁签名(开发用;交付分支已把改动直接提交,无需再跑)。
