# 外部 HF 三物料：LIS 分容器本地验收

日期：2026-10-14。验收主机34（`34.66.244.63`），两张96GB RTX PRO6000；使用固化的Pro5000 TP2预设。**这是本地gateway/worker分容器验收，不是实体Pro5000或LIS在线验收。**

## 固定物料

| 物料 | 固定身份 |
|---|---|
| worker local image | `sha256:e2a1f41bd37d50343b3882a4fb7f62fe7aae45b1adf4e30166f257c83af3962b` |
| gateway local image | `sha256:d7e3b5ed6619a78b9e0d9e327cf507ef3830681b4c9df692cd35c75d11c0e3dd` |
| tags | `o5-external-hf-worker:20261014-split-v11` / `o5-external-hf-gateway:20261014-split-v11` |
| package commit | `c8f7bba8`，执行实现`78aa5c45` |
| accepted numerical/runtime base | `o5-external-hf:20261014-v9`, `sha256:93aa6ee9df9abc96669d09235bcbd499be6a369c4a4fa63ec3507e3402afaaf2` |
| HF | `5cccbf73e8ca7bc01b0377eff299891e9dd45248` |
| accel | `6e33931cdf7e1fcff2f896d706bb73b949fc4a39` + `66c4d070e67b8fd091fa1549499c12cb7c99873b`兼容修复 |
| Demo | `08c22855e618e8b979d1b11a496ee9551146fd18` + manifest中显式列出的`core/deploy/modes.py` ENV/实际attach报告更新 |
| raw checkpoint | 74,606,011,097 bytes，SHA256 `0789c953677dc8ea71b6085543e1a946ae64bddecd2f6b911ac6380376ef01ff` |

完整source/runtime及逐文件SHA在`evidence/docker-lis-split-20261014/release-input-manifest.json`。容器内全部package文件逐一对manifest验证通过，entrypoint SHA256为`650df93a98ab2106c7c3cc179f0bb4a7dbfadf6d460e331ff4485d222acf658a`。两最终镜像保留v9全部基底层，仅增加package及manifest层。**v11未推送到任何registry。**

环境沿用已固定Pro5000来源runtime：Python3.10.22，torch2.14.0+cu130，transformers5.13，FA2.8.3.post1，FLA0.5.1，minicpmo-utils1.0.7，minicpm-o5-sdk0.0.6。完整版本/来源见`docker/external-hf/requirements-pro5000.lock`、`runtime-provenance.json`和前次`DOCKER_TP2_ACCEPTANCE.md`，本轮没有升级依赖或修改HF/accel计算。

## 包装层调整

- gateway接受LIS追加的`--host/--port/--internal-port/--http`，实测使用非默认18006/18007。
- 读取只读`/app/config.json`并保留全部字段，在`/tmp/o5-entrypoint/config.json`生成有效副本；recording=false保持，数据目录默认`/app/data`。平台原文件SHA前后不变，实际请求后`/app/data`没有录制文件。
- worker从`MODEL_PATH`加载raw `model.pt`及`token2wav_causal_vocoder/`；实测非默认`/mnt/model-cache/split-fixture/model`。backend绑定0.0.0.0，worker仍走localhost访问；gateway跨容器可访问两者health。
- 明确选择历史契约允许的`periodic_idempotent`：成功后20s refresh，失败1/2/4/8/16/20s退避，HTTP timeout5s。镜像固定策略与template声明必须一致。历史worker未挂载publisher-runtime，因此不依赖worker读取该ConfigMap。
- 注册使用既有gateway幂等PUT，父进程有限时监督，不增加脱离管理的后台注册进程。模型加载与registry暂时不可达相互独立。

## 最终v11实测

| 检查 | 结果与范围 |
|---|---|
| 初始gateway不存在 | worker/backend已ready，DNS注册失败按退避重试；创建gateway后自动注册成功 |
| 单工文字、真实音频 | 2/2 PASS，音频84480bytes，finite/nonzero |
| ordinary audio/video | 各2unit finalized、正常关闭；basic选项及视频slice=2抵达session；短probe均listen |
| FC recorded basic-1 | 8unit committed、3模型tool-call、2录制result被消费、9spoken.delta、5份有效PCM；无错误 |
| 活跃session跨refresh | 26unit、31.922s，26次状态采样保持同一ticket `q_4c0dee472126`且duplex_active；另一连接始终排队 |
| 活跃期间实际PUT | gateway日志07:42:40.841及07:43:00.866 UTC两次PUT200，均落在该session观测区间内 |
| gateway故障恢复 | 故意停22s，再启动后9.155s恢复1idle worker；worker容器ID、StartedAt、父/模型/worker PID均未变 |
| 重启后API | 文字+真实音频2/2 PASS；无需重载模型 |
| 退出与释放 | 07:47UTC两个容器exit0、noOOM；GPU各3MiB、无compute；测试network及SSH隧道已清理 |

CPU门：process-group消失/TERM升级/KILL/launch竞态，readonly配置，MODEL_PATH，非默认CLI，重试/刷新/stop，镜像ID解析全部PASS。固定LIS源码`08f73fd63a0da5c8fac12813079f6dddd937d7e2`的flat schema build/parse roundtrip由root独立PASS。

最终GPU常驻观测64579/24797MiB；启动TTS capture报告peak_allocated_bytes=76976994816。worker cgroup memory.peak为4848459776bytes，gateway为91459584bytes。这些是当前host/缓存/映射模型条件下的观测，**不能推导宿主RAM最低要求或实体72GB Pro5000装载可行性**。资源示例16CPU/192GiB是保守规划值，未做最低资源搜索。

## 34本地复起

权重及资产仍保存在：

```text
/home/weihongliang/o5-docker-lis-20261014/model/model.pt
/home/weihongliang/o5-docker-lis-20261014/model/token2wav_causal_vocoder/
```

完整已验命令存放`/home/weihongliang/o5-docker-lis-20261014/split-v11/commands.json`，同时归档在本目录evidence。复起助手在host34以及本仓库`tools/start_lis_split_host34.py`；它先检查镜像ID、GPU空闲、端口及名称冲突，不自动杀任务：

```sh
ssh weihongliang@34.66.244.63 \
  'python3 /home/weihongliang/o5-docker-lis-20261014/start_lis_split_host34.py'
```

固定角色镜像无需O5_ROLE覆盖。worker占两卡、gateway不占GPU。启动后34本机API为`http://127.0.0.1:18024/v1/realtime`（WebSocket用`ws://`），registry18025仅测试用。访问本机隧道示例：

```sh
ssh -N -L 18024:127.0.0.1:18024 weihongliang@34.66.244.63
```

这是API部署，不自动提供HTTPS公网页面。原验收容器保留为`o5-lis-split-{gateway,worker}-20261014-v11-completed`，退出态不占GPU；另保留v10完成态容器。复起脚本只启动，停止前应确认owner与实际状态，参照本次cleanup证据；不删除旧完成态证据容器。

## LIS交付接口

- `docker/external-hf/lis-contract-flat.json`才是历史`create-realtime --contract-json`输入；nested `lis-contract.json`是说明片段，不能作为flat输入。
- `lis-resources-pro5000.example.json`指定一个worker两GPU；`gpu_types`需按当前平台目录替换，其他资源为示例。
- gateway CLI/挂载配置、worker env、模型artifact和启动命令在`docker/external-hf/README.md`。平台镜像UUID与Docker image ID不同。
- 模型artifact只需同一root下raw model.pt+vocoder，遵守平台MODEL_PATH；没有重新导出HF权重。
- 新镜像push、平台镜像记录/模板/ModelWeight注册和LIS实际部署均未做；不能将本地验收声称LIS在线通过。

## 证据与边界

关键结果归档`evidence/docker-lis-split-20261014/`；全部原始客户端event/PCM和host日志在`/user/weihongliang/analysis-runs/20261014-docker-lis/`，34原始现场在`/home/weihongliang/o5-docker-lis-20261014/split-v11/`。

本轮是部署与生命周期门：未新做hidden/logits paired，未验证长上下文/128k/basic实际裁剪，未验证在线外部工具质量、实体Pro5000显存或LIS权限/公网。FC8unit客户端耗时0.498–1.462s，不宣称全unit实时。gateway恢复指新请求可用，不是活跃WebSocket会话透明续传。root的独立API/PCM和执行链审查、独立reviewer的schema审查随证据归档。
