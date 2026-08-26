# MiniCPM-O5 部署对接说明（多模式框架 + 双卡）

**给对接同学:** 本分支在原 demo 上加了一个**可插拔部署模式框架**,一个配置字段切换 单卡eager / 单卡优化 / 双卡张量并行(TP),并留了扩展位。不改配置 = **原始单卡 eager 行为逐字节不变**。

> 本文取代旧的 `INTEGRATION.md`(那份只讲被本框架取代的 `O5_OPTIMIZE` 开关)。

---

## 0. TL;DR(三种模式怎么起)

| 目标 | `config.model.deployment_mode` | 起法 | 前置 |
|---|---|---|---|
| 原始单卡(默认,行为不变) | `single_eager` | 现有起法不变 | 无 |
| 单卡 + 推理优化(<1s) | `single_opt` | 现有起法不变 | 无 |
| **双卡 TP(32K 上下文,更快)** | `tp2` | `bash core/deploy/launch_tp2.sh` | 完整 safetensors bundle |

改 `config.json`(或环境变量 `O5_DEPLOY_MODE`)即切换。

---

## 1. 实测(可信 benchmark:含 finalize、显式同步、warmup 丢弃、p95/p99/超1s、跨卡 max)

omni-speak(最紧场景):

| 模式 | 卡 | p50 | p99 | max | 超1s | 上下文 |
|---|---|---|---|---|---|---|
| single_opt | 1 | 687ms | 855 | 861 | 0/80 | 8K |
| **tp2** | 2 | **585ms** | 715 | **747** | **0/120** | **32K** |

四场景(voice/omni × listen/speak)双卡全部 <1s。

---

## 2. 发布 artifact（推荐：一套完整 safetensors）

把旧 `.pt` 转成完整 bundle（包含 VPM/APM/TTS/LLM 等所有模型权重）：

```bash
python tools/o5deploy/export_full_safetensors.py \
  --pt-path /path/to/weights.pt \
  --output-dir /path/to/o5_full_hf
```

bundle 根目录保存 Demo 外层权重，`llm/config.json` 保存 TP2 使用的 Qwen
配置；TP2 复用根目录的同一批 shard，不再额外复制一份 LLM backbone。
运行时通过 `O5_WEIGHTS_DIR=/path/to/o5_full_hf` 指定即可。Token2Wav 的
`assets/token2wav` 是运行时文件，不属于模型权重；可通过
`O5_ASSETS_DIR=/path/to/assets` 指定包含 `token2wav/` 的目录。若 bundle 和
assets 放在默认位置，则两个环境变量都可以省略，服务也不需要传
`--model-path` 或 `--pt-path`。

## 3. 配置

`config.json`(参考 `core/deploy/config.example.tp2.json`):
```json
{ "model": {
    "deployment_mode": "tp2",                 // single_eager | single_opt | tp2
    "llm_cache_len": 32768,                     // tp2 上下文上限(single_opt 默认 8192)
    "weights_dir": null,                        // 可选；默认自动发现完整 bundle
    "assets_dir": null                          // 可选；目录内应包含 token2wav/
} }
```
或用环境变量覆盖:`O5_DEPLOY_MODE=tp2 O5_WEIGHTS_DIR=... O5_LLM_CACHE=32768`。

---

## 4. 起服务

- **single_eager / single_opt(单进程,起法不变):**
  ```bash
  python -m py_backend.server --port 22500
  ```
- **tp2(双卡 SPMD,torchrun 2 进程):**
  ```bash
  O5_WEIGHTS_DIR=/path/to/o5_full_hf O5_ASSETS_DIR=/path/to/assets \
    bash core/deploy/launch_tp2.sh --port 22500
  ```
  rank0 起 HTTP(对外 `/backend` WebSocket),rank1 自动进模型提供的 `worker_loop`(不起 HTTP,只参与 LLM/graph TP 计算)。gateway / 客户端只连 rank0,**协议不变**。

---

## 5. 验证(两个,都已在本分支端到端跑过)

- **可信 benchmark**(比 demo 自带 `benchmark.py` 可信,后者见 §7):
  ```bash
  # tp2:  torchrun --nproc_per_node=2 tools/o5deploy/deploybench.py  (MODE=tp2 FORCE=speak)
  # single_opt: python tools/o5deploy/deploybench.py  (MODE=single_opt)
  ```
  产出 p50/p95/p99/超1s，engine 字段记录实际测的引擎。
- **live 网络冒烟**:起 tp2 server 后,`python tools/o5deploy/live_client.py`(连真实 `/backend` WS,推音视频,收 listen/text/audio delta)。本分支实测:双卡端到端出真实对话(文本+TTS),0 超 1s。

---

## 7. 必读注意事项

- **双卡 = 单会话吃两张卡** → 固定卡数下并发会话减半。tp2 是为「单会话大上下文 + 低延迟」,不是提吞吐。追求吞吐用 single_opt + 多副本。
- **数值等价**:所有优化/TP 是 argmax-preserving 浮点重排(决策逐 unit 一致,采样输出分布无差异)。换权重后建议重跑一次 teacher-forced 等价 gate。
- **回滚**:`deployment_mode=single_eager`(默认)= 原始单卡 eager,逐字节不变。
- **扩展**:加新模式(EP/4卡/量化)= 在 `core/deploy/modes.py` 写一个 `build(cfg)->BuildResult` 并 `register_mode(...)`,serving 栈其余不动。
- **原 `O5_OPTIMIZE` 开关已被本框架取代** —— 用 `deployment_mode=single_opt` 代替。

---

## 7. demo 自带 `benchmark.py` 不可信(重要)

审计发现 14 个 bug:**零 `cuda.synchronize`**(异步下 prefill GPU 工作漏进 generate 计时,per-module 分解是假象)、**漏计 finalize**(每 unit 必跑 ~100ms)、**无 warmup**(首 unit 吃图捕获尖峰)、**只报 avg/min/max 无尾部**、**测的优化引擎生产没开**、从不采样长上下文。**验时延/达标请用 `tools/o5deploy/deploybench.py`,别用 `benchmark.py`。**

---

## 8. 代码位置

- 框架:`core/deploy/{base,registry,modes,llm_wrapper}.py`;接线:`core/processors/{unified,backend_factory}.py`、`py_backend/server.py`、`config.py`。
- 工具:`tools/o5deploy/{extract_backbone,deploybench,live_client,bench_2card,grid_2card}.py`。
- 启动:`core/deploy/launch_tp2.sh`。
