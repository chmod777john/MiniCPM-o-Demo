# 外部 HF 三物料镜像：双卡 TP2 API 验收

2026-10-14 UTC。此记录补充此前 v6 单卡原生验收，旧结果范围不变。

## 结论与范围

**v9 镜像在 34 机器的两张 Pro6000 上，使用 Pro5000 TP 加速组合，已跑通当前 raw FC checkpoint 的单工、普通音视频双工和 FC 音频 API。**实际硬件不是 Pro5000；不将本次结果写成真实 Pro5000 的显存或性能验收。

| 项目 | 结果 |
|---|---|
| 当前镜像真实 C++ 扩展编译/加载 | direct、streamed、parallel、batched 共4项通过；batched仅编译，未启用 |
| 小矩阵精度检查 | 两GPU × 三实际实现 × BF16 reduction true/false，12项逐位一致；6项 unsupported splitK=False 明确拒绝 |
| 单工 API | 文字流、语音流2项通过；返回PCM有限且非零 |
| 普通双工 | 音频2unit、视频2unit均 finalized；关闭会话通过 |
| FC 音频 | 8unit完成，3次模型工具调用，2个录制结果被后续unit消费，语音输出、0错误 |
| FC 后切回单工 | 文字/语音2项通过 |
| 实际配置与文件 | attach report确认启用；应用/accel/包装/前端文件哈希匹配 |
| 生命周期 | 本轮TP2容器正常停止exit0；两GPU回到3MiB，无compute进程；SSH隧道已关闭 |

FC工具结果来自录制fixture，并未调用实时天气后端。最后一个unit发出第三次调用，测试已发送结果但没有再追加unit，因此只将前两次记为消费。普通4unit均listen，未触发实际滑窗裁剪；Basic阈值8k/6k和视频slice2只验参数接通。

本轮是打包与API功能验收。FC客户端完整unit耗时0.803–1.762s包含客户端等待、网络和服务调度；不声称实时性达标。未新增整模型hidden/logits精度对照，小矩阵门只覆盖此次C++兼容修复。实际Pro5000机器全部8卡被已有服务占用，未分配GPU。

## 不可变物料

| 项目 | 身份 |
|---|---|
| 本地最终镜像 | `o5-external-hf:20261014-v9` |
| 本地 image ID | `sha256:93aa6ee9df9abc96669d09235bcbd499be6a369c4a4fa63ec3507e3402afaaf2` |
| HF | `5cccbf73e8ca7bc01b0377eff299891e9dd45248`，`/opt/o5/hf` |
| accel | 基础`6e33931`，隔离兼容修复`66c4d070e67b8fd091fa1549499c12cb7c99873b`，`/opt/o5/accel` |
| Demo | 基础`08c22855`；`core/deploy/modes.py`及测试按`2c07b81c6119f4aea1fff21493ed65d2f9f33cf4`显式Git archive刷新，含上游修复`21d4410a`/`571defd2` |
| 镜像包装 | `2c07b81c6119f4aea1fff21493ed65d2f9f33cf4` |
| 权重 | raw `job_1054182_iter_0001275.pt`，74606011097bytes |
| 权重SHA256 | `0789c953677dc8ea71b6085543e1a946ae64bddecd2f6b911ac6380376ef01ff` |

v9只在34本地，**没有推送DockerHub或LIS**。之前已发布的v6是原生验收版，早于本轮双卡修复，不能当作双卡交付版本。LIS从未执行写操作，现有授权查询401；用户要求暂不推进LIS。

运行环境固定为Python3.10.22、Torch2.14.0+cu130、Transformers5.13.0、Triton3.8.0、FA2.8.3.post1、FLA0.5.1、causal-conv1d1.7.0、minicpmo-utils1.0.7、o5-sdk0.0.6。Python库位于`/opt/o5/.venv`。具体archive/native/ninja和文件tree哈希见[交付manifest](evidence/docker-tp2-20261014/delivery-manifest.json)。

## 实际启用配置

启动日志来自`model._acceleration_info`，并非只抄环境变量：

- `hardware=pro5000, mode=tp, devices=(cuda:0,cuda:1)`；
- `llm_graph/tts_graph/t2w_graph/merge_unit_feed/compact_logits=True`；
- `tts_fa2_whole_decode/canvas_window_exact_chain=True`；
- `long_context_head_split/retained_kv_append=True`；
- runtime hooks含`tp_selected_prefill/split_norm/parallel_concurrent_prefill/retained_kv_append`。

完整[解析后配置](evidence/docker-tp2-20261014/tp2-v9-resolved-acceleration.json)记录TTS私有启动16captures、RNG exact，warmup的peak_allocated_bytes=76976994816。停止前实际GPU显存0/1为65379/24797MiB。Pro6000有96GB；这些数字不能证明72GB Pro5000加载峰值有足够余量。

## 34上复起已验镜像

当前服务已停止，原停止容器保留为`o5-external-hf-tp2-20261014-v9-completed`。下列host launcher会先核实两张指定GPU均无compute进程且显存占用低于1GiB，并核对本地完整image ID；它不会终止已有任务。若卡被其他服务占用，应先由owner释放。

```sh
ssh weihongliang@34.66.244.63
python3 /home/weihongliang/o5-docker-lis-20261014/start_tp2.py \
  --engine docker --sudo --devices 0,1 \
  --model-dir /home/weihongliang/o5-docker-lis-20261014/model \
  --logs-dir /home/weihongliang/o5-docker-lis-20261014/logs-tp2-restart \
  --name o5-external-hf-tp2-restart-20261014 --port 18014 \
  --image sha256:93aa6ee9df9abc96669d09235bcbd499be6a369c4a4fa63ec3507e3402afaaf2
```

launcher显式设`O5_ROLE=standalone`，同容器启动gateway/backend/worker，gateway只映射`127.0.0.1:18014`。默认worker角色本身不是gateway API入口。权重目录只读挂载到`/models/o5`，包含`model.pt`和`token2wav_causal_vocoder/`。首次无cache启动会编译扩展；镜像已包含编译工具/headers/ninja及冻结cuBLAS链接名。

其他机器访问时自行启动SSH转发：

```sh
ssh -N -L 18014:127.0.0.1:18014 weihongliang@34.66.244.63
```

验收客户端使用`ws://127.0.0.1:18014/v1/realtime?mode=chat`和普通/FC对应实时API。原始实测命令见[actual-start-command.json](evidence/docker-tp2-20261014/actual-start-command.json)。host launcher/doc后续提交不会改变已验v9镜像。

## 发现和修复

1. v7实际TP2首次触发`streamed_blas_grouped.cpp`编译，旧bool写法不支持当前Torch enum返回类型。`66c4d07`统一四实现的bool/enum适配，并拒绝无法兑现的splitK=False；保留GEMM参数、compute类型、stream及reduction策略。详见[三阶段历史](evidence/docker-tp2-20261014/ROOT_CUBLAS_HISTORY.md)。
2. enum修复后链接器找不到`-lcublas`，镜像已有`.so.13`运行库。包装仅补指向同一冻结库的开发链接名，没有更换库。
3. Demo原先没消费两个加速ENV。`21d4410a`接线，`571defd2`把真正attach报告加入启动日志；默认和显式cfg优先级CPU门通过。
4. 发布前发现生成前端目录残留旧hash JS，改为完整替换目录，逐文件hash与manifest一致。
5. 进程管理补齐自然退出/信号竞态与TERM超时回收，CPU故障注入、gateway退出与真实TP2退出均通过。

## 真实Pro5000的准备状态

`118.89.233.148`是8×Pro5000 72GB；其`docker`为rootless Podman兼容入口。只在任务目录生成CDI配置，无全局修改。v6 OCI已导入，CPU依赖导入、配置解析、gateway健康/退出通过；raw1275及当前声码器hash均匹配。所有GPU均被caitianchi现有服务占用，未停止、未占用。v9尚未同步到该机；真实Pro5000双卡GPU启动、显存峰值及API速度仍待另行验收。
