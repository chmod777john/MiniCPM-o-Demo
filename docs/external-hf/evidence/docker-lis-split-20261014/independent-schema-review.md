# LIS split packaging 独立复核

日期：2026-10-14 UTC。Reviewer：`/root/simplex_final_review`。

## 范围与版本

- 打包代码：`/user/weihongliang/worktrees/o5-fc-dev-external-hf-20261013-hf-chat-20261014-docker-lis-20261014`，`78aa5c45cd78718c8f81c50c056c310a85083ba6`，开始复核时工作树干净。
- LIS 固定源码：`/user/weihongliang/lis-source`，`08f73fd63a0da5c8fac12813079f6dddd937d7e2`。
- 本次是源码/已有 CPU 测试审查，没有新增或运行测试，没有 SSH、GPU、Docker、部署操作，没有修改源码或读取凭据。
- 重点：平台声明/JSON schema、挂载配置、LIS 生成的运行参数与现有 Demo 的关系。进程监督及真实注册执行链由 root 负责，host34 两容器实测由 packager 负责。

## 结论：有一个声明错误和一个交付输入缺口

### 1. 必须修正：`registration_status.service_role="gateway"` 不合法

`docker/external-hf/lis-contract.json` 的 `lifecycle_contract.registration_status.service_role` 当前为 `gateway`。

固定 LIS：

- `realtime_base_contract.py:24` 的 `REGISTRATION_STATUS_SERVICE_ROLES` 仅允许 `public` / `registry`。
- 同文件 `:638` 检查该字段并拒绝 `gateway`。
- `resource_bundle_builder.py:71` 默认 `public`，`:210` 同样验证枚举。
- Demo `gateway.py:393` 的 `/status` 属于公开 app；`/health` 分别存在于公开 app `:342` 和 internal app `:426`。

因此应改为 **`public`**，配套 `port_name=http`、`path=/status`、`schema=minicpm_o_gateway_status_v1`。这是平台模板声明问题，本地两容器 API 成功不会自动覆盖这个校验。

### 2. 需要可直接供 CLI 使用的 flat contract/resource 输入

现有 JSON 的 nested `worker_registration` / `lifecycle_contract` 是描述性结构。README:83 已明确称它为 template fragment、不是完整 LIS resource，因此不能说文档声称它可直接部署；但它也不能直接作为 `create-realtime --contract-json` 输入。

`realtime_base_contract.py:54` 列出的 `_CONTRACT_FIELDS` 是扁平字段；`:177` 会拒绝额外字段，例如 `contract_version`、`gateway_transport`、`worker_registration`、`lifecycle_contract`、`gateway_config`、`worker_backend`。即使修正 nested role，直接传此文件仍会被拒绝。

建议把描述性文件和 CLI 输入明确分开，提供 `lis-template-contract.json` 与 resource 文件及创建命令；或把原文件替换成 flat contract，再将补充描述放入 README。当前 README 中写到的周期刷新要求必须在模板真正声明，不能仅存在于说明文字中。

可用的 flat contract 内容（源码静态核对，未执行 LIS CLI）：

```json
{
  "min_workers": 1,
  "max_workers": 1,
  "gateway_public_port": 8006,
  "gateway_registry_port": 8007,
  "worker_port": 22400,
  "backend_port": 22500,
  "registration_collection_path": "/internal/workers",
  "registration_path_template": "/internal/workers/{worker_id}",
  "registration_refresh_mode": "periodic_idempotent",
  "registration_refresh_interval_seconds": 20,
  "registration_max_backoff_seconds": 20,
  "gateway_liveness_path": "/health",
  "worker_readiness_path": "/health",
  "registration_status_service_role": "public",
  "registration_status_port_name": "http",
  "registration_status_path": "/status",
  "registration_status_schema": "minicpm_o_gateway_status_v1",
  "route_request_timeout_seconds": 1800
}
```

`periodic_idempotent` 下不应加入仅属于 startup-only 的 `registration_retry_interval_seconds` / `registration_max_attempts`；固定源码 `:615` 会拒绝非空 inactive 字段。

Resource 文件必须使用实际 schema：`gateway_cpu_cores`、`gateway_memory_gb`、`worker_cpu_cores`、`worker_memory_gb`、`gpu_types`、`gpu_count`。本版本 TP2 应为一个 worker 使用 **`gpu_count=2`**；不是两个单卡 worker。固定 LIS 的 worker memory 校验下限为64GiB，不能照旧 builder 的32GiB legacy 默认值直接写 canonical resource。实际 GPU type 名称应从平台资源目录确认，不能凭硬件口语名称编造 selector。具体 CPU/内存容量由打包/部署负责人结合实际加载峰值决定。

## 源码中已对齐的部分

### 挂载配置与外部 HF 路径

`entrypoint.py:91 configure`：

- 读取显式 `O5_DEMO_CONFIG_PATH` 或 `/app/config.json`；先完整复制 JSON，再增加必要 model/service 字段，写到 `/tmp/o5-entrypoint/config.json`，不写只读挂载。
- 保留 `recording.enabled=false`、平台 service 参数及其他字段；有效路径先写入环境再启动子进程。
- 平台有 config 时默认 `service.data_dir=/app/data`，与 builder 的 emptyDir 一致；已有 service.data_dir 会保留。
- worker 的 `MODEL_PATH` 指向含 `model.pt` 与 `token2wav_causal_vocoder/flow.pt` 的挂载目录，与本次 raw checkpoint 物料布局相符。
- `compile=false` 写在 Demo schema 的 `service` 层，model 下设置 canonical 路径/TP/attention/pt_path。
- gateway 不要求权重；其角色由薄镜像固定，符合平台不额外注入自定义环境的部署方式。

固定 LIS 的 `_config_map` 生成 gateway runtime/config；gateway mount 配置只读、data 可写；worker 不挂 publisher-runtime。worker 没有收到 gateway 的 recording=false 不构成 worker 本地录制缺陷，录制由 gateway 消费。

### 端口/角色/注册参数

- Gateway 接收并转发 LIS CLI 参数，缺省项才从 runtime/env 补齐；平台生成的 `--host 0.0.0.0 --port ... --internal-port ... --http` 可覆盖 baked 默认值。
- `validate_publisher_runtime` 检查存在的注册声明是否为 periodic20秒/maxbackoff20秒，拒绝 startup-only；worker 不挂该文件，因此仍依赖模板声明与 image baked 常量一致。
- builder 传入 `WORKER_ID`、可解析的 worker service endpoint、`WORKER_GPU_GROUP`、`GATEWAY_REGISTRY_URL`、`WORKER_PORT`、`BACKEND_PORT`、`MODEL_PATH`；entrypoint 使用同名变量。
- Backend 监听 `0.0.0.0`，worker 连接 loopback；与固定 LIS backend bind 契约相符。
- 注册 PUT URL/body 满足 Demo 注册接口；成功周期20秒、失败指数回退上限20秒、单次 HTTP timeout5秒。进程级恢复实测由 packager 提供，CPU fake clock 只能证明调度逻辑。

### Health 与幂等边界

- 公共 `/status` 返回 `minicpm_o_gateway_status_v1` 的聚合数量字段，与所选 schema 相符。
- gateway `/health` 不代表有可用 worker；worker `/health` 对 loading 也可能 HTTP200，remote backend 的 model_loaded 判断主要看已配置远端地址。不能把 HTTP200 单独当作模型/graph/精度验收。
- entrypoint 先等 backend health 才启动 worker，并监督子进程退出，缓解上述阶段问题；完整执行链由 root 独立审查。
- `worker_pool.py:332 register_worker` 同 endpoint 更新会复用原 WorkerConnection；只有 endpoint 改变才清理状态。`:459` 对 gateway 已派发但 health 暂时 idle 的状态保留忙碌，`:469` 保留已派发 ticket。因此同 endpoint 周期注册没有明显“清空正在进行会话”的源码问题。

## CPU 测试覆盖与缺口

`test_lis_contract.py` 已有：只读配置/字段保留、任意 MODEL_PATH、compile schema、注册契约 mismatch、CLI 覆盖、PUT URL/body、初次失败/成功刷新/成功后 gateway 消失/恢复/回退上限/STOPPING 的 fake clock 检查。

**缺口：未读取 `lis-contract.json` 并通过固定 LIS 的实际 schema。** 所以现有 CPU 断言不可能发现 `gateway` 枚举错误或 nested/flat 不匹配。建议新增以固定 validator 生成/解析 serving profile 的 CPU gate，至少包含 flat contract/resource 被接受、nested原文件拒绝、错误 role拒绝、往返生成的 publisher-runtime 为20/20/public，并保留所用 LIS commit。

## 验收边界与交接

- 修正上述声明及 CLI 物料后可继续模板创建；本复核不能替代平台实际创建/调度/公网 API 验收。
- host34 split 两容器功能测试不等同 LIS Pod 配置已通过，也不等同真实 Pro5000 TP2 数值/性能验收。
- 未触碰任何 source、GPU、服务、job、镜像；仅创建此报告。后续修改由 root/packager 执行。


## 同轮修正复查（未提交状态，后续目标 metadata v11）

收到root通知后重新读取真实工作树，确认：

- `lis-contract.json` 的 nested role 已改为 `public`。
- 新增 `lis-contract-flat.json`，字段和值与本报告建议flat契约一致，无额外或漏失的活动字段。
- 新增 `lis-resources-pro5000.example.json`，gateway2CPU/4Gi、worker16CPU/192Gi、GPU2卡。GPU selector为`NVIDIA-RTX-PRO-5000-Blackwell`；这份例子经过类型schema不等于集群目录存在该拼写，创建前仍以实际平台资源为准。
- packager/root报告固定`build_realtime_base_profiles`已PASS；reviewer本轮独立做了源码及文件核对，没有重跑测试。
- 当前`git status`是修改nested一文件、新增flat/resource两文件，尚未提交；最终artifact应将这三项及README用法一并固化，不应把78aa5c45原始错误文件当最终可用声明。

**在本review职责范围内，没有另外发现阻断两容器启动/现有API执行的问题；声明枚举与CLI输入缺口已在工作树修正。** 建议root继续收尾，无需等待所有GPU结果才处理此review。README应指明`--contract-json`接收flat文件，并记录resource中的GPU selector须与集群目录一致。

root通知split已通过simplex2项，以及约32秒/26unit活跃普通音频ticket保持、第二连接排队；此为owner上报，最终实测以LIS_SPLIT_REPORT及原始产物为准，本review不将这些结果外推到真实LIS部署/公网/Pro5000验收。


## 最终 split v11 运行证据复核（2026-10-14 UTC）

新任务范围：仅核对本地产物，重点容器/进程身份、真实注册与活跃会话重叠、配置、镜像/manifest。未运行新的CPU/GPU测试、未SSH或操作容器；root另行重数了API/PCM，见`ROOT_SPLIT_V11_API_AUDIT.json`。

### 镜像与源码归档

- package修正已提交 `c8f7bba8b5877ef0a63d20e07a335198641edd82`，读取时工作树干净。README已明确flat文件作为`--contract-json`输入，并写清GPU selector需替换成平台真实SKU。
- worker镜像：`sha256:e2a1f41bd37d50343b3882a4fb7f62fe7aae45b1adf4e30166f257c83af3962b`。
- gateway镜像：`sha256:d7e3b5ed6619a78b9e0d9e327cf507ef3830681b4c9df692cd35c75d11c0e3dd`。
- 两者均实际用于`commands.json`中的immutable image启动，并与各自initial/final Docker inspect及cleanup记录一致。镜像不是旧v10 ID。
- `release-input-manifest-split-v11.json` SHA256为 `486479af6f71d2d0646b37a4f707000e1a30313523e9ba02cd5247b5d1d02622`，与两容器内部捕获的manifest hash相同。
- reviewer将`split_packaging.files`全部18文件逐一与所标明c8f7bba8 Git blob SHA256比较：0 mismatch。两容器capture还逐一读实际`/opt/o5/package/`并assert相同hash；capture脚本已读过，`all_packaging_hashes_match`不是只复述manifest。
- manifest准确保留继承HF5cccbf7、Demo08c22855、accel6e33931，以及Demo/accel显式补丁和v9数值runtime base；split是package-only overlay。此门覆盖列出的18个包文件及manifest身份，不重新声称全继承runtime ELF/所有源文件/权重均在v11逐文件重验。

### Gateway重启时 worker 身份保持

`split-v11-host/split-v11/gateway-restart-summary.json`记录before/while-down/after三份worker身份完全一致；初始/final Docker inspect也独立对应：

- worker容器ID `7d8ba48ade27fcd7e6744b76253e9ef7a1ee4567d0c653e4828f87c748101520`。
- StartedAt始终 `2026-10-14T07:38:42.819629898Z`，主PID `558209`、backend PID `558460`、worker PID `558797`。
- gateway容器ID始终 `0a3159e8de8ece4f6b0af54002df69995c247d00341bd489d30887245510fdc3`，PID确实从559887变为562957，StartedAt从07:40:51变为07:44:43，证实gateway实际重启，而worker没有重建。
- gateway stop开始07:44:20.539 UTC，start开始07:44:43.247，约22.7秒；worker日志实有connection refused/DNS失败及1/2/4/8/16秒回退，随后07:44:52.115的PUT200恢复。状态确认耗时自start起9.155秒，total_workers=idle_workers=1。
- restart脚本是在idle状态启动此试验，验证模型进程保留及新会话恢复；不证明活跃会话可跨gateway重启透明续传。后续simplex2项通过由API原始产物/root独立重数支持。

### 注册 PUT 确实发生于活跃26unit内

活跃会话 `sess_359350e3e1b6` 时间窗07:42:34.534711至07:43:06.456903 UTC（31.922秒）。`gateway-final.log`有两次实际 `PUT /internal/workers/worker-0` 200：

- 07:42:40.841397815，距会话开始6.307秒；前后worker采样在6.196/7.388秒。
- 07:43:00.866409042，距会话开始26.332秒；前后worker采样在25.960/27.152秒。

两次PUT前后及全部26份采样均为`duplex_active`且ticket=`q_4c0dee472126`。第二客户端11条事件全为`session.queued`/`session.queue_update`，没有`session.created`/被派发。此项补齐root API audit早先标作“pending final gateway log”的重叠证据；不只是以运行超过20秒来推断刷新发生。

### 只读配置与退出

- initial/final inspect均显示`/app/config.json`和`/etc/lis/publisher-runtime`是RW=false的bind；`/app/data`可写。v11沿用名为split-v10的host fixture目录，实际内容和挂载属性已核对，不应因目录名推断运行的是旧镜像。
- host初始config hash、重启检查hash、容器最终`/app/config.json` hash全为 `2da49d6ae6ce5072cd561f40045f767ebc48eaa7aed250393cafe3b0ea8bf549`，对应原始字节`{"recording": {"enabled": false}}`。
- 最终effective config保留recording=false并添加service.data_dir=/app/data，其独立hash为`13b0716f73c0ac3be59a894848a9d1d420a146de15895ad603e69459cdb68fca`；没有写回只读源。`session_recording_files=[]`。
- `cleanup.json`显示两容器均ExitCode0/OOMKilled=false、Running=false；GPU0/1各3MiB且无compute进程。reviewer仅检查owner留存结果，没有停止任何资源。

在上述证据范围内未发现新阻断。当前本地split包装/注册恢复/API功能门已具备相互吻合的证据；仍不能替代LIS在线部署、镜像发布、真实Pro5000硬件、完整端到端数值或实时性能验收。交付清单生成后继续核对其引用的镜像与manifest，避免把旧版本ID写进最终说明。
