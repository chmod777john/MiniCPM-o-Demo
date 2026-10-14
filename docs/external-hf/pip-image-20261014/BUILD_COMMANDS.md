# 37 构建与复核命令记录

## 权限和输入

专用Pod `eval/o5-pip-buildkit-20261014` UID `c29170bc-ec89-4da6-b584-873e27f37117`，node `10.20.1.37`。Host `/data/weihongliang/o5-pip-image-20261014` ↔ Pod `/work`。CPU4/16GiB，无GPU。

`pro37_exec.py`每次只为该Pod建立经118的SSH临时隧道、核对UID/Running，然后kubectl exec；TLS验证保持，原kubeconfig不修改，不输出凭据。Root拥有Pod生命周期。

本次固定image代码为79177c66。后续docs commit不重新标记image。context：`/work/context/pip-image`；来源和输入均附hash。

## 最终worker

```sh
buildctl build --frontend dockerfile.v0 \
  --local context=/work/context/pip-image \
  --local dockerfile=/work/context/pip-image/docker/external-hf \
  --opt filename=Dockerfile \
  --oci-layout python-base=/work/inputs/python-base.oci \
  --opt context:python:3.10-slim-bookworm@sha256:5be5aaecb962a41567bc8041f1e9e3432cb085b037fe057f9a562630b2e4b370=oci-layout://python-base@sha256:db6ee5170ebce0f4d165af242e8e77addc08bd9ffc37e8b1815752cf027aa4f8 \
  --output type=oci,name=o5-external-hf:pip-20261014,dest=/work/output/o5-external-hf-pip-20261014.oci.tar \
  --metadata-file /work/output/o5-external-hf-pip-20261014.metadata.json \
  --progress plain
```

最终日志：`pro37-final-image-v3.log`。所有RUN及输出成功。此前v1遗漏context .dockerignore的apt-offline白名单，v2严格拦decord原始Tag问题；最终保留两例外并加入CPU实测门，不将前两轮失败隐藏。

## Gateway派生

worker OCI完整hash通过后解到`/work/output/worker.oci`。`/work/context/gateway/Dockerfile`只有：

```dockerfile
FROM o5-pip-worker AS gateway
ENV O5_ROLE=gateway
```

构建：

```sh
buildctl build --frontend dockerfile.v0 \
  --local context=/work/context/gateway --local dockerfile=/work/context/gateway \
  --opt filename=Dockerfile --oci-layout worker=/work/output/worker.oci \
  --opt context:o5-pip-worker=oci-layout://worker@sha256:c35318d4cc66ffc9d15a6d21cf69f8c5d46526ebd7ff1b1144f8a2604aa3f8f6 \
  --output type=oci,name=o5-external-hf-gateway:pip-20261014,dest=/work/output/o5-external-hf-gateway-pip-20261014.oci.tar \
  --metadata-file /work/output/o5-external-hf-gateway-pip-20261014.metadata.json \
  --progress plain
```

Docker格式导出用同一OCI named context，worker Dockerfile仅FROM；gateway仍ENV角色。`--output type=docker,name=...,dest=....docker.tar`。生成的image config和所有layer与OCI完全一致；序列化manifest媒体类型不同，所以manifest digest不同。用image config digest确认同一Docker image身份。

## 镜像复核

- `inspect_oci_archive.py`：全archive SHA、每blob size/SHA；实际image config角色、三物料路径；所有层无model.pt/flow.pt；抽取真实source/install/audit/CPU JSON。
- `verify_image_inputs.py`：实际各层文件内容与Git材料tar/packaging/mobile manifest SHA逐文件比较，1444文件完全一致；实际137包版本与install manifest完全一致。
- `installed-native-consistency.json`：实际image安装FA2/causal/dense binary SHA与新源码wheel一致。
- `gateway_cpu_probe.py`：从确切worker OCI启动实际入口gateway，loopback随机端口、无GPU；public/internal health和桌面/移动页面200；SIGTERM exit0。
- `docker-archive-identity.json`和`*-docker-archive-verification.json`：普通Docker归档manifest/config/layer核验。尚未在GPU宿主执行docker load+模型服务。

## 权重传输

原checkpoint74606011097B，SHA `0789c953677dc8ea71b6085543e1a946ae64bddecd2f6b911ac6380376ef01ff`。

`transfer_checkpoint_pro37.py`固定512MiB分块，每次远端stat+SHA，核对后合入`model.pt.part`并marker；最终全文件大小+SHA通过后才rename为model.pt。单一进程独占锁，已验块可恢复。vocoder8文件按原34逐文件manifest核验，未改变原件。

最终以`checkpoint-transfer-result.json`/`vocoder-transfer-result.json`为完成证据。没有将公开对象存储作为中转，也没有转换checkpoint。
