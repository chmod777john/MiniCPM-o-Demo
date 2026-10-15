# LIS 镜像、权重与 Pro5000 双卡 API probe 交付说明

状态核对时间：2026-10-15T10:15:29+00:00。以下镜像是2026-10-14在37机器验收的原镜像，2026-10-15上传LIS时没有升级代码。

## 1. LIS 资产标识

|物料|LIS 资产 ID|名称|版本|当前状态|
|---|---|---|---|---|
|Worker|`5bf9a66a-f9b9-40e0-a8f6-ef3c77c95f11`|`o5-external-hf-worker`|`pip-20261014`|上传/登记Completed，exists=true|
|Gateway|`aeedb241-c4a9-4cb8-bc7d-07982d6d66fa`|`o5-external-hf-gateway`|`pip-20261014`|上传/登记Completed，exists=true|
|权重+vocoder|尚未生成（后台转存未完成）|`o5-fc-raw-1275-with-vocoder`|`iter1275-20261015`|Pushing|

- 两镜像共同 `publisher_release_id`：`o5-external-hf-pip-20261014`。
- `engine_type=omni_duplex`，角色分别为worker/gateway；资产环境dev，保留180天。
- Worker上传任务：`eb000755-37de-463b-9bcd-25377791794e`。
- Gateway上传任务：`593c2d96-cd1b-47e8-9a60-82ed6001bb75`。
- **权重上传任务 ID：`af790c8d-bb41-4a9e-8233-f64559520950`。这是上传任务ID，不是模型资产ID，不能填入要求model weight ID的部署字段。**
- 本次核对权重状态：流式推送到 SWR：43.62 GiB / 70.71 GiB（87%）。该进度百分比是整个流程的阶段加权进度；字节进度见机器可读快照。
- 镜像资产 `scan_status=Pending`；不把上传成功称为扫描完成。
- 尚未创建LIS推理服务，因此没有本次LIS服务ID或在线体验地址。

### 最终镜像引用

Worker：

```text
modelbest-kt8zv1.swr-pro.myhuaweicloud.com/lis-dev-images/o5-external-hf-worker@sha256:c0463e5ea82eba878b8d111460fc7f2350ec61c6d76eac54749346909a4659eb
```

Gateway：

```text
modelbest-kt8zv1.swr-pro.myhuaweicloud.com/lis-dev-images/o5-external-hf-gateway@sha256:6f88f803f5ecbfa68ab1068eb5aface102c5d11fa243df7607aabd4e27a5783a
```

原验收时worker OCI digest为`sha256:c35318d4cc66ffc9d15a6d21cf69f8c5d46526ebd7ff1b1144f8a2604aa3f8f6`，gateway为`sha256:143bdde56a917513253998c8ccfa032fc84ffcb72e7c38b1da95bbbcd7a5e3cd`。发布时转为Docker schema2 manifest，config与全部17层的digest、大小、顺序均相同；最终仓库tag也已读回核对。此次用BuildKit推送，没有新增代码层或更改启动配置。

## 2. 镜像与权重内容

|组成|镜像内路径|固定commit|
|---|---|---|
|HF|`/opt/o5/hf`|`5fea76841a268720e83ce68bd14f8c064ad72bde`|
|accel|`/opt/o5/accel`|`337ac3e84b69dde135efaad7f4113bc244fe7bd4`|
|Demo/打包|`/opt/o5/demo`、`/opt/o5/package`|`79177c666a0bbafe5b0ed360bde836254d5cd09c`|

镜像内Python位于`/opt/o5/.venv/bin/python`。固定依赖经pip安装到新venv，未复制本地site-packages：torch2.14.0+cu130、transformers5.13.0、triton3.8.0、flash-attn2.8.3.post1、fla-core0.5.1、minicpmo-utils1.0.7、minicpm-o5-sdk0.0.6。以上版本是原镜像身份，不代表当前正式分支HEAD。

权重为FC训练checkpoint：

```text
/user/chenjinpeng/Checkpoints/job_1054182/checkpoints/iter_0001275/job_1054182_iter_0001275.pt
```

上传目录保留raw checkpoint，不转换为safetensors：

```text
MODEL_PATH/
  model.pt
  token2wav_causal_vocoder/    # 8个文件
```

`model.pt`为74,606,011,097bytes，SHA256为`0789c953677dc8ea71b6085543e1a946ae64bddecd2f6b911ac6380376ef01ff`。完整9文件共75,920,482,457bytes。权重没有嵌入镜像。

权重通过标准`lisctl model upload`走本地→平台OBS→平台Worker流式转存SWR→模型资产登记。客户端到OBS已完成；最终SWR模型资产ID/digest必须等待后台完成后读回，不能提前把上传任务ID当资产ID。

## 3. 原镜像如何调度到37双卡

验收日期为2026-10-14，使用37 Kubernetes节点`10.20.1.37`，namespace=`eval`。Pod为`o5-pip-validation-20261014`，UID=`1a0f3098-540c-462d-bf69-5a83082d6e0a`。

硬件实测为**2张 NVIDIA RTX PRO 5000 72GB Blackwell**。Kubernetes通过`nvidia.com/gpu: 2`给worker容器分配两卡；gateway容器不申请GPU。实际执行的加速配置为`hardware=pro5000, mode=tp, devices=(cuda:0,cuda:1)`。

当时尚未发布SWR，因此先由节点loopback临时只读registry服务原OCI，再由kubelet/containerd按固定digest正常拉取。没有通过挂载宿主runtime socket手工注入镜像。此临时registry仅用于原验收，现已清理。

因现场CNI可用Pod IP不足，最终采用**同一个Pod中的两个独立容器**：worker/backend与gateway，共享Pod网络。该验收组织方式使两者通过127.0.0.1互通；它不等于已经在LIS跑过分Pod部署。

|配置|验收实际值|
|---|---|
|worker资源requests|8CPU / 96Gi RAM / 2GPU|
|worker资源limits|24CPU / 192Gi RAM / 2GPU|
|gateway资源requests / limits|1CPU/1Gi；4CPU/8Gi；无GPU|
|共享内存|`/dev/shm`，memory emptyDir 16Gi|
|权重源目录|`/data/weihongliang/o5-pip-image-20261014/model`|
|容器权重挂载|只读`/models/o5`，`MODEL_PATH=/models/o5`|
|探针/日志挂载|`/data/weihongliang/o5-pip-image-20261014/gpu-validation` → `/validation`|
|Gateway对外API端口 / 内部注册端口|8006 / 8007（验收使用HTTP）|
|Worker / Backend端口|22400 / 22500|
|Worker注册地址|`http://127.0.0.1:8007/internal/workers`|
|Worker对Gateway通告地址|`127.0.0.1:22400`|

这些资源是验收时的配置，不是测得的最低资源需求。如果以后Gateway和Worker分属不同Pod，应将注册地址和通告地址替换为实际Service/Pod可达地址，不能照抄127.0.0.1。

## 4. 服务启动与请求调度

### 启动方式的准确边界

Gateway：Pod没有覆盖`command`，只设置`args`，由容器运行时直接使用镜像内的ENTRYPOINT。

Worker：Pod显式覆盖了`command`，由Kubernetes启动一个等待脚本；native算子独立进程对比通过、写入`/validation/start-api-approved`后，该脚本用`exec`调用与镜像ENTRYPOINT完全相同的程序。省略提示日志后，实际命令是：

```sh
sh -ec '
  while test ! -f /validation/start-api-approved; do sleep 2; done
  exec /opt/o5/.venv/bin/python /opt/o5/package/entrypoint.py
'
```

Worker与Gateway镜像记录的ENTRYPOINT都是：

```json
["/opt/o5/.venv/bin/python", "/opt/o5/package/entrypoint.py"]
```

所以该轮不是进入一个已运行容器后，靠交互shell另起Backend/Worker来充当镜像启动验收；服务由Pod启动进程链调用原entrypoint并管理。容器内执行过的是native检查、探针客户端及标记文件操作。

**但这也不是worker完全不覆盖command、只凭镜像默认ENTRYPOINT/CMD自动启动的实测。** 本轮已经覆盖原entrypoint的配置加载、模型加载、子进程启动、注册与正常关闭；对默认入口的直接启动，只核实了镜像元数据与实际exec目标相同，不能把它说成已独立实测。Gateway则实际使用了默认ENTRYPOINT。

入口按原包装逻辑依次启动Backend和Worker，日志记录的子命令为：

```sh
/opt/o5/.venv/bin/python -m py_backend.server   --host 0.0.0.0 --port 22500 --gpu-id 0   --model-path /opt/o5/hf --pt-path /models/o5/model.pt

/opt/o5/.venv/bin/python worker.py   --host 0.0.0.0 --port 22400 --gpu-id 0   --backend-server-url http://127.0.0.1:22500
```

这里`--gpu-id 0`是服务入口参数；实际LLM TP2由accel配置使用两张卡，不能据此认为只有单卡。

Gateway使用gateway角色镜像原入口，参数为`--host 0.0.0.0 --port 8006 --internal-port 8007 --http`。Worker完成模型加载后向Gateway幂等注册，并每20秒刷新；验收记录显示8006、8007、22400、22500的health均为200，registry有一个idle worker。

实际请求链路：

```text
Probe WebSocket → Gateway:8006（队列/分配worker）
                → Worker:22400（会话调度/转发）
                → Backend:22500（加载外部HF，附加accel，执行推理）
                → 同一路径返回文本、音频、工具调用和unit事件
```

探针从Gateway公开API进入，没有直接调用模型函数代替API验收。服务进程共用同一个FC checkpoint，并在单工、普通双工和FC模式之间切换。

日志确认启用LLM/TTS/vocoder graph、merge-unit、compact logits、长上下文head split、retained KV append、canvas window、TTS FA2 whole decode等既有Pro5000 TP配置。模型启动约92秒，其中LLM warmup约49.94秒。

## 5. Probe 怎样判定通过

### Native依赖对齐

在目标镜像内分别用新编译native和历史reference运行独立进程，共17组输出：FA2五组、causal prefill六组、causal update输出/状态两组、dense causal四组。全部finite、逐位相同，relative_l2与max_abs均为0。这是native模块门，不等于全模型hidden paired精度验收。

### 单工12请求

入口为`ws://127.0.0.1:8006/v1/realtime?mode=chat`。覆盖文本流/非流、语音流/非流、历史消息、高清图片、输入语音、语音cap1、视频+语音流/非流、较长流式语音、thinking文本，12/12通过。检查完成事件、预期文字，以及语音PCM非空、finite、非零。

这是原HF5fea768版本的结果。不要把该12/12套到后来精简过单工逻辑的新HF镜像上。

### 普通音视频双工

采用已提交的移动端helper生成的API payload，通过Gateway逐unit提交并等待`response.metrics`中的`unit_finalized`。先做音频/视频各2unit监听，再补充各12unit以覆盖真实说话、TTS和vocoder。

关键配置：`chunk_ms=1000`、输入16kHz、`generate_audio=true`、`sliding_window_mode=basic`、高/低水位8000/6000、`max_new_speak_tokens_per_chunk=20`；说话覆盖设`force_listen_count=0, speak_threshold=1e-6`，视频`max_slice_nums=2`。这是为了覆盖说话路径的验收配置，不是自然对话的说话比例质量评测。

结果：音频12/12和视频12/12 finalized；分别12和11个音频chunk，PCM finite且非零。前端选项抵达session配置。

|普通双工|Backend unit耗时p50 / max|Backend prefill p50|本批>1s|
|---|---|---|---|
|音频12unit|567.05 / 608.8ms|198.85ms|0/12|
|视频12unit|650.7 / 698.3ms|302.45ms|0/12|

耗时来自后端`wall_clock_ms`，排除客户端排队/网络，finalize单独报告；prefill包含媒体处理，不是纯LLM token prefill曲线。音频KV107～334、视频KV316～2863，未达到basic滑窗触发水位。

### FC8unit与跨模式恢复

通过官方v4 payload转换器回放历史`basic-1`录制fixture，使用`mode=audio`、`tts_pad_protocol=legacy`、`decoding_policy=constrained_decoding`、语音输出开启。等待每unit的`response.unit.committed`。

结果：8/8unit committed，模型真实发出get_client_location、amap_weather、show_card共3个工具调用；探针回送3份录制结果，其中2份在停止前被后续unit消费。收到9个spoken.delta，其中5份携带PCM，共104640 samples，finite且非零，errors为空。工具结果来自fixture，未验证真实外部天气服务。FC客户端提交到commit为约0.200～1.018s，不能与上表后端计时混用。

FC/普通双工结束后，同一服务再做单工文字和语音2请求，2/2通过，覆盖跨模式恢复。

### 探针调用与证据

最终脚本、输入、事件和PCM位于原归档`pro37-gpu-api-evidence.tar`，不能仅用最初的probe脚本包替代最终修正后的脚本。以下是准备好`/validation/probes`及其原输入后，在当时worker容器内的调用形式：

```sh
PROBE_URL='ws://127.0.0.1:8006/v1/realtime?mode=chat' PROBE_OUTPUT=/validation/api-simplex /opt/o5/.venv/bin/python /validation/probes/probe_api.py

PROBE_BASE=http://127.0.0.1:8006 PROBE_OUTPUT=/validation/api-ordinary-speech-r3 /opt/o5/.venv/bin/python /validation/probes/ordinary_speech_probe.py

/opt/o5/.venv/bin/python /validation/probes/run_recorded.py
```

历史首次执行有探针侧路径/断言修正：系统参考音频改为同内容的`/validation/probes`路径；说话阈值0被API拒绝后改合法1e-6；普通音频事件断言由错误的`kind=speak`改为实际`kind=audio`并校验PCM。服务和镜像没有为这些探针修改；失败记录与最终成功记录均保留。

## 6. 证据路径与验收边界

原始记录根目录：`/user/weihongliang/analysis-runs/20261014-docker-pip-install/`。

- `PRO37_GPU_REPORT.md`：原验收逐步记录。
- `pro37-validation-pod.json`：实际调度Pod配置。
- `pro37-worker-full.log`、`pro37-gateway-full.log`：真实启动/注册/退出日志。
- `pro37-gpu-api-summary.json`：最终总汇总，`passed=true`。
- `pro37-simplex-summary.json`、`pro37-ordinary-speech-summary.json`、`pro37-fc-summary.json`、`pro37-after-modes-summary.json`：分项结果。
- `pro37-gpu-api-evidence.tar`：最终脚本、输入、事件和PCM，39,557,120bytes，SHA256=`4297806c560934d8603f858272a8229f685630cea9401cbdf3f9be213f1c0b38`。

当前发布记录根目录：`/user/weihongliang/analysis-runs/20261015-lis-upload-capability/`。

- `ASSET_IDS_READBACK.json`：本文对应的LIS资产与权重任务状态快照。
- `images/ORIGINAL_PUBLICATION_REPORT.md`：原双镜像发布完成报告。
- `images/original/*-final-registry-identity.json`：最终镜像与原验收镜像内容一致性。

本次没有重新跑模型/API probe；复用的是原镜像在37的验收。验收后的validation Pod和临时registry已正常删除，当前不提供该历史Pod的访问链接。

**已证明：** 原镜像在真实双Pro5000上完成native模块对齐与上述完整API链路，并且上传的镜像内容与原验收产物一致。

**尚未证明：** LIS实际部署/公网服务可用、LIS分Pod网络配置、全部hardcase与长上下文/实际basic裁剪、完整模型hidden paired、真实外部工具服务质量、浏览器人工体验。
