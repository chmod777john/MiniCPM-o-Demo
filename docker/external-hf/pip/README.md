# 固定 wheel 安装（CPU 构建验收完成，GPU/API 待验）

此目录替代旧 `runtime/pro5000-python310-site.tar.gz` 环境快照。所有 Python 包必须在新 venv 中通过 pip 安装；禁止从已有 site-packages 重封 wheel。

## 输入

- `public-wheel-inputs.json`：132 个来自 PyPI / PyTorch 官方索引的固定 wheel URL、版本和官方 SHA256。
- `requirements-public.lock`：上述 wheel 的 `--require-hashes` 安装输入。
- `source-inputs.json`：FA2、causal-conv1d 官方 sdist 和 NVIDIA CUDA 13.0.2 redistribution 中 nvcc/cudart/cccl/crt/libnvvm 的 URL 与 SHA。
- `flash-attn-cxx20.patch`：历史 Pro5000 构建相对官方 sdist 的两处构建标准调整；不改 kernel 算法。
- 内部 `minicpmo-utils==1.0.7` 原始 wheel 来自 commit `11700a4382531fd7a54a48f6409f09f7ebc90ad5`；SDK `0.0.6` 原始 wheel 与 PyPI SHA 对齐。新native源码构建完成，137包完整安装锁已在交付context中生成。

下载到普通磁盘构建目录：

```sh
python docker/external-hf/pip/download_locked.py --output /path/to/context
```

使用正常 Docker 引擎。先准备至少35GiB可用空间，建议45GiB；模型只读挂载，不复制进镜像。禁止全局 prune 或清理没有明确 owner 的文件。

## 元数据例外

固定清单用 `pip --no-index --no-deps --require-hashes` 安装。`--no-deps` 用于锁定整组已验收依赖，不代表依赖一致性检查自动通过。

137 个实际安装包完成容器审计（包括 torch 激活的 CUDA extras）。`pip check` 真实返回1，保留两项明确元数据例外：utils声明librosa0.9.0但固定环境为0.11.0；原始decord wheel内部CPython tag与文件名不一致（详后文）。其他冲突均阻断。所有核心包导入与新native加载通过；GPU/API仍待独立验收。

## dense 自定义扩展

固定 `dense_causal/build.py` 在原资格工具链 GCC13.2.1/CUDA13.0.88、Torch2.14cu130、固定Torch/CUTLASS headers 下重新编译。不能复制旧已安装 cache 作为本轮输入，不改资格检查。

新编译产物由 `package_dense.py` 打成 `cp310-cp310-linux_x86_64` wheel，pip 将 data files 安装到 venv `var/cache/minicpm-o-accel/<key>`。wheel 包含新 binary、manifest、build.ninja、derived header、diff、来源哈希。新旧 SASS dump 已逐字节一致，并在目标镜像实际CPU加载通过；GPU执行仍待验。

## 当前验证范围

来源核对、137包全新venv hash安装、dense/FA2/causal固定源码编译及目标容器CPU加载完成。37独立BuildKit已构建并导出worker/gateway OCI；14项容器CPU门、真实gateway HTTP静态页与退出流程通过。实际image内137包版本、三物料/打包/移动资产1444文件与输入完全一致。GPU算子及模型API尚未验收。证据在 `/user/weihongliang/analysis-runs/20261014-docker-pip-install/`。不得以旧镜像验收记录替代本轮门。

## Debian 离线输入

固定 Python base 的原始 apt 索引确定248个包，`debian-package-inputs.json`锁定版本、官方URL、大小及签名Packages中SHA256。构建context的`apt-offline/`保存原始deb、签名InRelease和Packages；先用base自带Debian keyring验签，并验证每个包，再在`RUN --network=none`中通过`apt-get --no-download`安装。没有复制已安装系统库，也不需要目标节点外网。安装清单保存在镜像`/opt/o5/debian-installed.tsv`。

### 实际安装发现的第二个元数据例外

官方 `decord-0.6.0-py3-none-manylinux2010_x86_64.whl` 内部 WHEEL 标记为 cp36-cp36m，与文件名不符。保留官方原wheel，不编辑安装元数据；固定原wheel、内部WHEEL和libdecord.so的SHA。新pip check因此报平台标签警告，审计明确允许这一条及utils/librosa一条，其他任何错误均阻断。镜像CPU门额外要求真实decord ctypes视频解码与ffmpeg RGB逐像素完全相等；native没有CPython PyInit入口。`pip check`仍是exit1，不能写成全部无冲突。

最终Docker构建内执行CPU导入/新native真实加载、外部HF加载、decord视频、入口生命周期与LIS配置契约门。未调用GPU、不加载模型权重；GPU精度和真实模型API仍独立待验。
