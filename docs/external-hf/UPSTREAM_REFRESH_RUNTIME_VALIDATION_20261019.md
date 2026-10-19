# O5 FC 上游更新：实际验证记录（2026-10-19）

## 结论与固定代码

**新版三物料通过代表性的真实 TP2 API 验证，无需修改产品推理代码。** CPU 回归和 GPU API 各自说明验证范围；单卡全 Graph 的容量失败保留。这里不声明长时间实时、业务质量或新一轮 GPU 数值等价验收。

| 物料 | 被测运行源码 | 测试/文档提交 |
|---|---|---|
| HF `fc` | `db7541095e458bf71269a49fafea508369556574` | `528cb102868278f276d99a562a67c8912eee8996` |
| Demo `demo-adapter-fc` | `ba64bb95f70a118b435ad336ad2fa4eec3d13258`，其中实现为 `79c08878` | `e55d21c2de01526e292cd42cbc74adaca42a0813`，另加本验收文档提交 |
| accel `hf-fc-accel` | `337ac3e84b69dde135efaad7f4113bc244fe7bd4` | 不变 |

追加提交仅涉及测试、文档和证据，运行文件与上表源码一致。源输入仍固定为 `o5-fc-dev@e0f2ada084d395a00d1956957d345e82b8a5f8f7`。未推送远端、替换镜像、改 LIS 或重启已有服务。

## CPU 与接口回归

| 范围 | 结果 | 边界 |
|---|---|---|
| Demo Python 合并回归 | 642 passed，4 deselected | API v5、Gateway/重连、Gallery、外部桥接、ordinary/simplex 等；包含20项新增迁移边界检查 |
| 桌面/移动端 Node | 25 passed | 选项、会话、取消、音频引用等自动测试；未做浏览器人工交互或生产构建 |
| 独立脚本 | 3 个 exit 0 | 外部 ChatView、取消收尾、accel 参数 |
| 外部 HF 组件 | 30 passed | 真实方法体通过 AST 隔离加载，模型/processor/codec 为显式替身；真实 canvas 和固定 accel 代码执行 |
| accel 既有 CPU 合同 | 59 passed | FC feed、attention policy、graph buffer/warmup 等；非真实 CUDA Graph 数值验收 |
| 迁移生成器 | 7个输出完全复现 | 固定源及配方 |

HF检查覆盖新 TTS condition、`spoken_turn_eos` hidden、轮末 `text_eos → audio_bos`、legacy开关、两种pad协议、causal chunk/tail及reset；视频检查包括原生positions/spec、合并feed、compact-feed、几何和模式校验。18组网格及15组多帧序列配置与上游位置逐元素一致。一次 synthetic CPU 合并矩阵 logits 出现最大 `2.38e-7` 舍入差，输入/位置/hidden仍严格一致；测试保留`1e-6` logits容差，不作为真实GPU精度证明。

首次失败均保留：4项O45测试因旧processor注册写法在transformers5.13下导入失败，相关源/测试与合并前字节一致，未扩展修O45；Node首次缺TypeScript，项目内安装测试依赖后25项通过；HF测试替身的AST筛选错误只修测试脚本。

## GPU 环境、资源与输入

最终任务 **1295039**：`langfang_train` 的用户授权 `deploy` 池534，2×A100-SXM4-80GB、16CPU、256GiB，状态 `Succeeded`。本任务后端、Worker、Gateway均为临时独立进程，端口28191/28192/28519/28419；三个测试任务及服务均已结束，无遗留GPU占用，原9007和其他任务保持原状态。

环境：Python3.10.12、torch2.11.0+cu128、CUDA12.8、transformers5.13.0、flash-attn2.8.3、triton3.6.0、minicpmo-utils1.0.7、minicpm-o5-sdk0.0.6。使用既有只读项目venv，没有修改共享环境。

权重：`job_1054182_iter_0001275.pt`。录制音频/工具fixture：`OmniAgentBench/runs/0812_cjp_2/20260924_064536/basic-1`。视频为重复448×448合成几何图片，用于检查帧进入模型及位置/指标链路，无视觉问答质量结论。

策略：slot、`tts_pad_protocol=legacy`、`constrained_decoding`、`non_spoken_scheduling=quality`；使用本次更新后的TTS condition，未打开旧condition比较开关。LLM cache设为8192。工具返回用录制fixture，不访问真实外部工具；未命中fixture的`show_card`返回显式合成错误，验证错误结果也能被模型消费。

## 真实 API 结果

| 场景 | 完成范围 | 结果 |
|---|---|---|
| FC v5 audio | 12 units；3次工具调用，3个返回被后续unit消费；7.92秒非零有限音频 | 通过；unit3、11出现turn_eos |
| FC v5 Omni | 5 units，每unit64个视觉token；1个工具返回被消费；1.36秒非零有限音频 | 通过；unit3出现turn_eos；最后一次工具调用的返回未等到下一unit消费 |
| audio错误携带视频 | 预期`input_mode_error` | 正确拒绝并清理 |
| 错误后的v4 audio | 新会话2 units | 通过，listen-only |
| Omni缺帧 | 预期`input_mode_error` | 正确拒绝并清理 |
| 错误后的Omni | 新会话1 unit | 通过，listen-only |
| FC后切普通视频双工 | 同进程3 units，basic配置 | 通过，均为listen；未触发滑窗，未验证普通发声 |

测试观察包装只记录原方法返回和状态，不替换推理结果。3次`end_of_turn`均有非空音频，随后`token2wav_buffer=0`、`tts_text_start_pos=0`。无需为通过测试修改HF/accel/Demo产品代码。

## 加速开启与实际观察

TP2运行记录确认：LLM Graph、TTS Graph、vocoder Graph、Merge-unit、compact logits、TTS FA2 whole decode、retained KV append、canvas window/exact-chain均启用。日志记录LLM warmup及16次TTS图捕获。

Omni的真实合并示例：8次feed（含66-token image spec块）合为一次82-token执行；每帧64个视觉token，native image-spec账本逐unit增长。合并后的`cost_video`可能包含排队时间，不能单独解释为视觉GPU计算耗时。

**没有逐项Graph replay计数，也没有本次真实checkpoint的native/accel paired数值结论。** replay计数包装在后端启动后才写入，未作为本轮证据；实际启动包装另存`backend_entry_started.py`。A100日志还提示可选dense-causal预编译artifact不可用，走原attention；不能笼统说所有硬件专用优化都已验收。

## 耗时与失败边界

下表从串行`input.append`发出到`response.unit.committed`；包含首次形状开销、模型自然生成的工具/文本工作量，不是持续按秒发送的实时验收，也不是等输出加速倍率。

| 场景 | unit数 | 中位数 | 最大值 | >1秒 |
|---|---:|---:|---:|---:|
| TP2 FC audio | 12 | 0.776秒 | 1.732秒 | 4/12 |
| TP2 FC Omni | 5 | 0.426秒 | 1.006秒 | 1/5 |

单A100任务1294952在Graph/Merge关闭、其余accel挂接的配置下6个API场景通过。单A100任务1294993全Graph/Merge开启，在第4个音频unit轮末重置时申请700MiB失败：总占用78.88GiB、Graph私池4.75GiB。后续场景在同一显存不足进程中也失败；日志完整保留。换为全开TP2后通过，未以关闭双卡加速或修改推理语义规避失败。

未覆盖：allmerge权重、128k/长上下文、真实滑窗触发、持续实时/并发、业务与音质等价、revised pad真实权重、H20/Pro5000、本次新单工GPU。普通双工仅验证FC后恢复及短视频listen流程。

## 证据与复现

机器可读摘要：[runtime-acceptance.json](evidence/upstream-refresh-20261019/runtime-acceptance.json)。原始证据索引及哈希：[raw-evidence-index.json](evidence/upstream-refresh-20261019/raw-evidence-index.json)。运行源码哈希：[runtime-source-hashes.json](evidence/upstream-refresh-20261019/runtime-source-hashes.json)。

原始目录：`/user/weihongliang/analysis-runs/20261019-o5fc-runtime-validation/`。

- `gpu-validation/tp2-all-on/job_entry.sh`：环境、三物料路径、checkpoint、独立端口及有限任务入口。
- `driver.py`：启动真实Gateway/Worker/backend、注册端点、运行probe并清理。
- `backend_entry_started.py`：本轮实际观察方法包装；无后来未执行的replay计数。
- `probe.py`：音频、工具返回、视频、模式错误、v4及普通视频回归；事件、PCM和trace保存在各case目录。
- `environment.json`、`ACCELERATION_INFO.json`、`OBSERVATIONS.json`、`job-final.json`及三层服务日志提供环境、启用项、执行观察和任务收尾证据。

如需重跑，应先检查并认领可用GPU，在新的实验目录配置独立任务与日志；不要直接复写本次固定证据或复用已有公共服务。
