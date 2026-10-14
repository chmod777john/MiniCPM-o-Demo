# 固定 wheel 安装（构建流程正在验收）

此目录替代旧 `runtime/pro5000-python310-site.tar.gz` 环境快照。所有 Python 包必须在新 venv 中通过 pip 安装；禁止从已有 site-packages 重封 wheel。

## 输入

- `public-wheel-inputs.json`：132 个来自 PyPI / PyTorch 官方索引的固定 wheel URL、版本和官方 SHA256。
- `requirements-public.lock`：上述 wheel 的 `--require-hashes` 安装输入。
- `source-inputs.json`：FA2、causal-conv1d 官方 sdist 和 NVIDIA CUDA 13.0.2 redistribution 中 nvcc/cudart/cccl/crt/libnvvm 的 URL 与 SHA。
- `flash-attn-cxx20.patch`：历史 Pro5000 构建相对官方 sdist 的两处构建标准调整；不改 kernel 算法。
- 内部 `minicpmo-utils==1.0.7` 原始 wheel 来自 commit `11700a4382531fd7a54a48f6409f09f7ebc90ad5`；SDK `0.0.6` 原始 wheel 与 PyPI SHA 对齐。最终安装锁会在 native 构建完成后补齐。

下载到普通磁盘构建目录：

```sh
python docker/external-hf/pip/download_locked.py --output /path/to/context
```

使用正常 Docker 引擎。先准备至少35GiB可用空间，建议45GiB；模型只读挂载，不复制进镜像。禁止全局 prune 或清理没有明确 owner 的文件。

## 元数据例外

固定清单用 `pip --no-index --no-deps --require-hashes` 安装。`--no-deps` 用于锁定整组已验收依赖，不代表依赖一致性检查自动通过。

134 个公共/SDK/utils wheel 按 Linux CPython3.10 检查（包括 torch 激活的 CUDA extras），已知唯一冲突：内部 utils 声明 `librosa==0.9.0`，验收环境为 `0.11.0`。保留现有版本并明确记录该例外。FA2/causal 加入后仍需重做完整 pip check、import 和 GPU/API 门。

## dense 自定义扩展

固定 `dense_causal/build.py` 在原资格工具链 GCC13.2.1/CUDA13.0.88、Torch2.14cu130、固定Torch/CUTLASS headers 下重新编译。不能复制旧已安装 cache 作为本轮输入，不改资格检查。

新编译产物由 `package_dense.py` 打成 `cp310-cp310-linux_x86_64` wheel，pip 将 data files 安装到 venv `var/cache/minicpm-o-accel/<key>`。wheel 包含新 binary、manifest、build.ninja、derived header、diff、来源哈希。新旧 SASS dump 已逐字节一致，仍需在目标镜像实际加载与执行。

## 当前验证范围

来源核对、公共依赖全新 venv hash安装、dense固定源码编译/平台wheel/pip安装/CPU加载已经完成。最终正常Docker镜像、FA2/causal新编译、GPU/API验收尚未完成。证据在 `/user/weihongliang/analysis-runs/20261014-docker-pip-install/`。不得以旧镜像验收记录替代本轮门。

## Debian 离线输入

固定 Python base 的原始 apt 索引确定248个包，`debian-package-inputs.json`锁定版本、官方URL、大小及签名Packages中SHA256。构建context的`apt-offline/`保存原始deb、签名InRelease和Packages；先用base自带Debian keyring验签，并验证每个包，再在`RUN --network=none`中通过`apt-get --no-download`安装。没有复制已安装系统库，也不需要目标节点外网。安装清单保存在镜像`/opt/o5/debian-installed.tsv`。
