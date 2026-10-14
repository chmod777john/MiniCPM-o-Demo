# 2026-10-14 pip/wheel 镜像构建与 CPU 验收

## 范围

本次把历史 site-packages 快照安装改成原始 wheel / 固定源码构建 wheel 的真实 pip 安装。137 个 Python distribution 在镜像全新 `/opt/o5/.venv` 中使用 `--no-index --no-deps --require-hashes` 安装；248 个系统包使用带签名索引及固定 SHA 的原始 Debian 包离线 apt 安装。

镜像构建于37节点的专用4CPU/16GiB BuildKit Pod，无GPU。没有更改同事Pod或宿主runtime，没有推送registry或部署LIS。

## 固定代码身份

| 物料 | 镜像中的路径 | 输入 commit |
| --- | --- | --- |
| HF | `/opt/o5/hf` | `5fea76841a268720e83ce68bd14f8c064ad72bde` |
| acceleration | `/opt/o5/accel` | `337ac3e84b69dde135efaad7f4113bc244fe7bd4` |
| Demo / packaging | `/opt/o5/demo`、`/opt/o5/package` | `79177c666a0bbafe5b0ed360bde836254d5cd09c` |

本文件及同目录README的后续文档提交不属于镜像内代码身份。镜像中仍记录79177c66。HF/accel计算源码未在此次打包中修改。

## 镜像身份

| 角色 | OCI manifest digest | image config / Docker image ID |
| --- | --- | --- |
| worker | `sha256:c35318d4cc66ffc9d15a6d21cf69f8c5d46526ebd7ff1b1144f8a2604aa3f8f6` | `sha256:b094e9eea2ad246129947c9f940df7881328c2ee5a47305d87742d6453ff7c7b` |
| gateway | `sha256:143bdde56a917513253998c8ccfa032fc84ffcb72e7c38b1da95bbbcd7a5e3cd` | `sha256:48d6767f7b03f8beee76e2fcfebaa48999eb7372eed7c05d0f51c4025c7532bf` |

两角色17层内容完全相同，gateway只固定`O5_ROLE=gateway`。默认worker为Pro5000 TP2、独立HF和accel路径、slot及全部已选加速开关。权重在镜像外只读挂载，不导出safetensors、不内嵌model.pt。

完整归档在37宿主：

```text
/data/weihongliang/o5-pip-image-20261014/output/
  o5-external-hf-pip-20261014.oci.tar
  o5-external-hf-gateway-pip-20261014.oci.tar
  o5-external-hf-pip-20261014.docker.tar
  o5-external-hf-gateway-pip-20261014.docker.tar
```

两个OCI归档已备份共享存储：`/user/weihongliang/analysis-runs/20261014-docker-pip-install/`。Docker格式与OCI格式的manifest媒体类型/digest不同，config/image ID与层内容保持；Docker格式可供`docker load`，本轮尚未在GPU运行时load执行。

| OCI归档 | 大小（bytes） | 完整 SHA256 |
| --- | ---: | --- |
| worker | 4148780544 | `cb1ffe46cd2f2d4a6723327522f2ada3f32f7675a647741ee74870ecf4d0da04` |
| gateway | 4148781056 | `bf6488cb23abb0a786ee416ab996624214dd4d26e27a9255554d5c3ebf372eda` |

## 已通过的检查

- 原始公共wheel/SHA与新native源码构建来源锁定。FA2/causal从官方sdist新编译；dense从固定资格源码/headers/toolchain新编译。禁止从旧site-packages重封。
- 真实容器安装137包完成；CUDA13官方redistribution最小SM120编译通过。
- 14项CPU门：新FA2/causal/dense加载、FLA/SDK/utils导入、外部HF modeling与backend/gateway/worker导入、实际decord视频解码、3项入口生命周期/LIS配置/镜像选择契约。
- 实际gateway入口启动：public/internal health、桌面Omni、mobile、mobile-omni均HTTP200；SIGTERM退出0。
- OCI归档全SHA、manifest/config/每层blob的大小及SHA全部通过；镜像中source/install/audit/CPU报告已提取。
- 实际镜像中的137包版本完全匹配输入；三份物料、packaging和移动资产共1444个文件，missing/changed/extra全部0。移动端两个tree hash与原已验版相同。
- Root独立复核decord例外、核验脚本、源文件一致性及两角色layers相同，记录在`ROOT_ENV_REVIEW.md`。

## 两项明确元数据例外

`pip check`真实exit1，保留两条原输出：

```text
minicpmo-utils 1.0.7 has requirement librosa==0.9.0, but you have librosa 0.11.0.
decord 0.6.0 is not supported on this platform
```

第一项维持已验版本组合；第二项来自官方decord wheel文件名`py3-none`但内部WHEEL残留`cp36-cp36m`。原wheel与元数据都未修改；固定WHEEL/native SHA，并实测ctypes读取4帧32x32视频与ffmpeg RGB逐像素相等。这个测试不覆盖所有视频编码。

## 尚未覆盖

新镜像的GPU算子新旧A/B、TP2真实模型API、Pro5000性能和完整音视频媒体回归尚未运行。旧镜像的GPU/API结果不能继承为这版验收。37尚未交接GPU；后续由root协调。

原始checkpoint及vocoder传输以证据目录的`checkpoint-transfer-result.json`和`vocoder-transfer-result.json`为准；大checkpoint必须最终完整SHA通过后才原子命名。中途分块进度不代表完成。

## 证据

可提交的摘要：`docs/external-hf/pip-image-20261014/`。全部日志、来源归档、镜像归档及原始JSON：`/user/weihongliang/analysis-runs/20261014-docker-pip-install/`。
