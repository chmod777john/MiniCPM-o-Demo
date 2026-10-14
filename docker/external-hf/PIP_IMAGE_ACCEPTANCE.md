# 2026-10-14 pip/wheel 镜像构建与 Pro5000 GPU/API 验收

## 范围

本次把历史 site-packages 快照安装改成原始 wheel / 固定源码构建 wheel 的真实 pip 安装。137 个 Python distribution 在镜像全新 `/opt/o5/.venv` 中使用 `--no-index --no-deps --require-hashes` 安装；248 个系统包使用带签名索引及固定 SHA 的原始 Debian 包离线 apt 安装。

镜像构建于37节点的专用4CPU/16GiB BuildKit Pod。构建完成后按用户新授权正常关闭指定旧8GPU Pod，再使用独立2GPU运行真实新镜像验收；模型数据和其他任务保持，宿主runtime未改。临时loopback只读镜像源由kubelet正常CRI拉取，无外部registry推送或LIS部署。

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

四个OCI/Docker归档均已完整备份共享存储并通过全SHA和各blob校验：`/user/weihongliang/analysis-runs/20261014-docker-pip-install/`。Docker格式与OCI格式的manifest媒体类型/digest不同，config/image ID与层内容保持；Docker格式可供`docker load`；本轮实际GPU验收使用正常containerd/CRI按固定OCI digest拉取运行，没有使用Docker load路径。

| OCI归档 | 大小（bytes） | 完整 SHA256 |
| --- | ---: | --- |
| worker | 4148780544 | `cb1ffe46cd2f2d4a6723327522f2ada3f32f7675a647741ee74870ecf4d0da04` |
| gateway | 4148781056 | `bf6488cb23abb0a786ee416ab996624214dd4d26e27a9255554d5c3ebf372eda` |

| Docker归档 | 大小（bytes） | 完整 SHA256 |
| --- | ---: | --- |
| worker | 4148783104 | `d21385e842eb4923c84db5335408b7b169229e0b74612c4c57a11768066b32a4` |
| gateway | 4148783616 | `98a6365f9924b70ad1c3098724b1cfd98e98d0224907879ff48c61827e167505` |

## 已通过的检查

- 原始公共wheel/SHA与新native源码构建来源锁定。FA2/causal从官方sdist新编译；dense从固定资格源码/headers/toolchain新编译。禁止从旧site-packages重封。
- 真实容器安装137包完成；CUDA13官方redistribution最小SM120编译通过。
- 14项CPU门：新FA2/causal/dense加载、FLA/SDK/utils导入、外部HF modeling与backend/gateway/worker导入、实际decord视频解码、3项入口生命周期/LIS配置/镜像选择契约。
- 实际gateway入口启动：public/internal health、桌面Omni、mobile、mobile-omni均HTTP200；SIGTERM退出0。
- OCI/Docker四个归档全SHA、manifest/config/每层blob的大小及SHA全部通过；镜像中source/install/audit/CPU报告已提取。
- 实际镜像中的137包版本完全匹配输入；三份物料、packaging和移动资产共1444个文件，missing/changed/extra全部0。移动端两个tree hash与原已验版相同。
- Root独立复核decord例外、核验脚本、源文件一致性及两角色layers相同，记录在`ROOT_ENV_REVIEW.md`。

## 两项明确元数据例外

`pip check`真实exit1，保留两条原输出：

```text
minicpmo-utils 1.0.7 has requirement librosa==0.9.0, but you have librosa 0.11.0.
decord 0.6.0 is not supported on this platform
```

第一项维持已验版本组合；第二项来自官方decord wheel文件名`py3-none`但内部WHEEL残留`cp36-cp36m`。原wheel与元数据都未修改；固定WHEEL/native SHA，并实测ctypes读取4帧32x32视频与ffmpeg RGB逐像素相等。这个测试不覆盖所有视频编码。

## Pro5000 GPU/API已通过范围

- 实际Pro5000 72GB双卡Pod内，新native与旧reference分别独立进程运行，17项输出全部逐位一致：FA2、causal-conv1d及dense causal。
- TP2原始checkpoint加载、grouped扩展编译、既有加速开关和20秒注册刷新正常。
- 单工12/12请求，包含文本/语音流与非流、图片、真实视频解码和输入音频，返回PCM有限非零。
- 普通双工初始音频2+视频2 unit通过；补充说话各12unit全部finalized并有有效PCM。后端wall_clock音频p50/max567.05/608.8ms、视频650.7/698.3ms，均0/12超过1秒。
- FC录制basic1 8/8unit，通过3次实际工具调用/3份fixture回送/2份后续消费；5个PCM片段。客户端unit最大1017.86ms，无错误。
- 上述模式后单工text/audio2/2再次通过。自己的GPU Pod和loopback registry已正常删除，资源交还root；BuildKit保留。

详细证据见`docs/external-hf/pip-image-20261014/pro37-gpu/`。API是录制fixture及有限场景功能/媒体验收，未调用外部工具服务。

## 尚未覆盖

本轮未新增64k/128k、全部hardcase、长时滑窗、完整模型hidden逐层paired精度或模型质量评测。普通说话KV最大2863，不能外推为所有工况实时；单工视频首次请求12.426s含cold执行/编译，非双工unit时延。公开registry/LIS部署仍待授权。

## 模型文件传输

原始checkpoint及vocoder均已完成传输。37宿主模型目录为`/data/weihongliang/o5-pip-image-20261014/model`，可作为`MODEL_PATH`整体挂载：

- `model.pt`：74606011097 bytes，完整SHA256 `0789c953677dc8ea71b6085543e1a946ae64bddecd2f6b911ac6380376ef01ff`。139分块已验，最终整文件SHA通过后原子命名，传输进程已退出，`.part`及锁已清。
- `token2wav_causal_vocoder/`：8个原始文件逐SHA通过。

最终证据见`checkpoint-transfer-result.json`、`checkpoint-final-state.json`和`vocoder-transfer-result.json`。文件传输与随后本节列出的实际GPU模型加载分别有证据。

## 证据

可提交的摘要：`docs/external-hf/pip-image-20261014/`。全部日志、来源归档、镜像归档及原始JSON：`/user/weihongliang/analysis-runs/20261014-docker-pip-install/`。
