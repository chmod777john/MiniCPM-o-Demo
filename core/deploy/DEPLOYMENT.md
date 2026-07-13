# MiniCPM-O5 多部署模式框架(core.deploy)

一个**可插拔的部署模式框架**:同一套 demo 代码,通过配置切换单卡 / 单卡优化 / 双卡张量并行,并可扩展未来模式(EP、4 卡、量化…)。

## 1. 为什么

- 生产 server 之前**没有启用任何推理优化**(单卡也跑 eager);优化只存在于 benchmark 脚本和实验 probe 里。
- 32K 上下文单卡装不下,需要双卡张量并行(TP)。
- 未来还会有更多部署形态。需要一个统一、可扩展的框架,而不是每种形态改一遍加载/serving。

## 2. 架构

```
config.model.deployment_mode  ──►  backend_factory (设 O5_DEPLOY_MODE 等 env)
                                        └► UnifiedProcessor._load_model
                                              └► core.deploy.get_mode(mode).build(cfg) ──► BuildResult(model, world_size, is_driver, engine, broadcast_input)
```

- `core/deploy/base.py` —— `DeploymentMode`(name/world_size/requires_spmd/build)、`BuildResult`。
- `core/deploy/registry.py` —— `register_mode / get_mode / list_modes`。
- `core/deploy/modes.py` —— 三个内置模式的构造器 + 共享 helper(手术式加载、引擎启用、token 广播)。

**加新模式** = 在 `modes.py` 写一个 `build(cfg, rank, world) -> BuildResult` 并 `register_mode(...)`,serving 栈其余不动。

## 3. 内置模式

| mode | 卡 | 引擎 | 启动 |
|---|---|---|---|
| `single_eager` | 1 | 无(原始 eager,默认,行为不变) | `python -m py_backend.server` |
| `single_opt` | 1 | batched_mm + tts/llm/vocoder CUDA-graph + 融合视觉 | 同上 |
| `tp2` | 2 | 上述 + 骨干张量并行 + token 广播同步 | `core/deploy/launch_tp2.sh`(torchrun 2-rank) |

配置(`config.json`):
```json
{ "model": { "model_path": "...", "pt_path": "...",
             "deployment_mode": "tp2", "backbone_dir": "/path/o5_backbone_hf", "llm_cache_len": 32768 } }
```
或用环境变量 `O5_DEPLOY_MODE / O5_BACKBONE_DIR / O5_LLM_CACHE` 覆盖。

## 4. tp2 前置:抽取骨干

TP 用 `from_pretrained(tp_plan="auto")` 边加载边分片,需要 HF 格式的骨干:
```
python tools/extract_backbone.py   # .pt 里 llm.* -> HF safetensors 到 O5_BACKBONE_DIR(~65GB,一次性)
```

## 5. 实测(可信 benchmark,含 finalize、严谨同步、warmup 丢弃、跨 rank max、p95/p99/超1s)

omni-speak(最紧场景,120 units,32K 上下文能力):

| mode | p50 | p95 | p99 | max | 超1s |
|---|---|---|---|---|---|
| single_opt (1 卡) | 687ms | 752 | 855 | 861 | 0/80 |
| **tp2 (2 卡)** | **585ms** | 632 | 715 | **747** | **0/120** |

双卡 decode per_tok 15.2ms(单卡 28.6ms@32K / 部署单卡 18.5ms@8K),权重读减半 → 更快 + 装得下 32K。全部 <1s。

## 6. benchmark 正确性(重要)

demo 自带的 `model.benchmark()` 经审计**不可信**(14 个 bug):零 `cuda.synchronize`(prefill GPU 工作漏进 generate 计时)、漏计 finalize、无 warmup、只报 avg/min/max(无尾部)、测的引擎生产没开、从不采样长上下文。**请用框架 benchmark `tools/deploybench.py`**(修掉全部问题:双 rank 显式 sync、finalize 计入 unit_total、warmup 丢弃、p50/p95/p99/超1s、跨 rank max、部署引擎)。

## 7. 剩余(live gateway SPMD serving)

框架已能在 tp2 下**构建模型 + 跑完整 duplex**(benchmark 证明)。live 对外服务还差最后一个 hook:`py_backend/server.py` 的 duplex 请求循环要用 `BuildResult.is_driver` 分流(rank0 收 gateway 请求)+ 每个 unit 前 `BuildResult.broadcast_input(audio_chunk)` 把输入广播给 rank1(保证两卡跑相同 collective)。launcher `launch_tp2.sh` 已就位;这一步需一次 live 2-GPU serving 冒烟测。
