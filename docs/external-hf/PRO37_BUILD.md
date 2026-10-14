# Pro5000 37 的 Kubernetes 构建入口

2026-10-14，owner root，构建执行交给 docker_lis_package。

## 当前状态

独立 CPU BuildKit Pod 已创建并通过小镜像构建。此验收覆盖 Dockerfile RUN、venv/pip 和 OCI 导出；不代表最终模型镜像、GPU算子或API已通过。

- 集群配置：`/user/weihongliang/37-pro5000.config`，目标节点`10.20.1.37`。
- Pod：`eval/o5-pip-buildkit-20261014`，UID `c29170bc-ec89-4da6-b584-873e27f37117`。
- 最大4CPU/16GiB，不申请GPU，不挂宿主runtime socket，不安装宿主Docker。
- 任务目录：宿主`/data/weihongliang/o5-pip-image-20261014`，容器`/work`；检查时空闲约4.6TiB。
- 固定BuildKit amd64镜像：`moby/buildkit@sha256:f4f5220ec5e56f4d63d81cb73086c85a8cdb0195b355e53a9667c90bdc5443ea`。
- Rootless运行使用native snapshotter和本Pod内no-process-sandbox；Pod专用于可信的本任务构建，非多租户构建服务。

同事`eval/dev-8gpu-37`保持原样，8GPU仍归它分配；未动其中任务、环境或节点标签。GPU运行验收待资源协调，不可凭瞬时显存空闲绕过分配。仅构建不需要释放GPU。

## 网络和固定输入

本机无法直接访问API内网地址；使用经`weihongliang@118.89.233.148`的临时SSH转发，kubectl以原配置加`--server=https://127.0.0.1:<port> --tls-server-name=10.20.1.44`访问，不修改配置、关闭TLS验证或复制凭据。

构建Pod直连Docker Hub超时。通过34下载并逐层校验官方固定Python基础镜像，作为OCI layout离线输入`/work/inputs/python-base.oci`。固定index为`sha256:5be5aaecb962a41567bc8041f1e9e3432cb085b037fe057f9a562630b2e4b370`，对应amd64 manifest `sha256:db6ee5170ebce0f4d165af242e8e77addc08bd9ffc37e8b1815752cf027aa4f8`。没有替换基础镜像内容。

实际RUN构建创建`/opt/o5/.venv`、运行pip并检查sys.prefix通过；输出`/work/output/run-smoke.oci.tar` SHA256 `3516175e2cb5c8b4d897b0c8ada79ae8ba8ffe21ff34e17114960a43816f5088`。

## 证据与后续

相邻`evidence/pro37-builder-20261014/`记录Pod配置、Dockerfile、buildctl命令和日志。完整现场证据：`/user/weihongliang/analysis-runs/20261014-docker-pip-install/pro37-builder/`。

Packager使用同一builder完成真实pip/wheel镜像；后续处理最终Dockerfile的固定frontend/系统包输入并核验依赖、原生算子与API。无registry推送、LIS发布或现有服务变更。构建Pod生命周期由root负责，操作前核对UID；仅删除本任务Pod，不能改同事Pod。
