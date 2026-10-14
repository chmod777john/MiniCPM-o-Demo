# 37 Pro5000 pip/wheel 镜像与原始权重交付

## 当前范围

完整镜像构建、CPU入口及归档验收已完成；checkpoint完整SHA通过并已原子命名model.pt，配套vocoder8文件逐SHA通过。授权的构建与传输任务已完成。GPU资源仍由root协调，本轮没有GPU/API新结果，也没有推送registry或部署LIS。

## 固定代码与依赖

- 镜像build commit：`79177c666a0bbafe5b0ed360bde836254d5cd09c`。
- HF：`5fea76841a268720e83ce68bd14f8c064ad72bde`。
- accel：`337ac3e84b69dde135efaad7f4113bc244fe7bd4`。
- 后续文档commit不是镜像内代码版本。此次未修改HF/accel计算源码。
- Python3.10.22；Torch2.14.0+cu130、Transformers5.13.0、Triton3.8.0、FLA0.5.1、FA2.8.3.post1、causal-conv1d1.7.0、SDK0.0.6、minicpmo-utils1.0.7。
- 137个distribution通过原始wheel或固定源码新编译wheel，在全新venv内由pip安装；248个系统包为签名索引和精确SHA约束的原始Debian包。

## 镜像身份

| 角色 | Docker image ID/config SHA256 | OCI manifest SHA256 |
| --- | --- | --- |
| worker | `b094e9eea2ad246129947c9f940df7881328c2ee5a47305d87742d6453ff7c7b` | `c35318d4cc66ffc9d15a6d21cf69f8c5d46526ebd7ff1b1144f8a2604aa3f8f6` |
| gateway | `48d6767f7b03f8beee76e2fcfebaa48999eb7372eed7c05d0f51c4025c7532bf` | `143bdde56a917513253998c8ccfa032fc84ffcb72e7c38b1da95bbbcd7a5e3cd` |

两个角色的17层内容完全相同。Docker归档改变manifest格式，但config/image ID和各层与OCI相同。

## 交付布局

37宿主 `/data/weihongliang/o5-pip-image-20261014`，Pod内对应 `/work`：

```text
output/
  o5-external-hf-pip-20261014.oci.tar
  o5-external-hf-gateway-pip-20261014.oci.tar
  o5-external-hf-pip-20261014.docker.tar
  o5-external-hf-gateway-pip-20261014.docker.tar
model/
  model.pt                   # 74606011097B，完整SHA通过
  token2wav_causal_vocoder/  # 8文件已逐SHA验收
```

四个镜像归档均有完整共享副本，路径 `/user/weihongliang/analysis-runs/20261014-docker-pip-install/`；大小/SHA见DELIVERY_MANIFEST.json和PIP_IMAGE_ACCEPTANCE.md。

## 已验收

- wheel/source、工具链、系统deb来源和SHA锁定，FA2/causal/dense新源码构建。
- 镜像实际137包版本和1444份代码/静态资产与输入完全一致。
- 镜像实际native文件SHA与本次新编译产物相同。
- 14项CPU门通过；实际gateway两个health及omni/mobile/mobile-omni页面200，SIGTERM正常退出。
- 四个镜像归档的完整SHA和所有config/layer blobs通过；权重未写入镜像。
- checkpoint完整74606011097B，SHA256 `0789c953677dc8ea71b6085543e1a946ae64bddecd2f6b911ac6380376ef01ff`，139块验证、全SHA通过后原子rename；传输已exit0，.part和锁不存在。
- vocoder目录8个原始文件逐SHA通过。

## 明确例外与待验范围

`pip check`真实exit1：utils1.0.7要求librosa0.9.0而保留历史0.11.0；decord原wheel内部Tag残留cp36-cp36m。二者已由root独立审核，原包metadata未改，其他异常仍阻断。decord原native在真实容器读取4帧32x32视频与ffmpeg逐像素相等；不据此扩大到所有codec或GPU运行。

新native的GPU算子A/B、双卡模型加载、真实TP2 API和完整音视频回归未执行。旧镜像结果不能继承到这版。后续由root协调GPU并完成这些门，再决定合入和registry/LIS发布。

## 过程中的关键记录

- 37直连Debian下载很慢；改用原签名索引所约束的同版本deb离线安装，未替换版本或搬已安装系统库。
- kubectl stdin短命令曾短写却exit0；传输脚本改为显式stat/hash输出，并独立逐块核SHA后合入，最终再核整个checkpoint。保留失败证据和分块marker。
- metadata标签检查发现decord原包标签不一致；保留真实错误并做容器实测及root审核。
- 构建及完整命令见BUILD_COMMANDS.md，过程见REPORT.md，授权见PRO37_BUILD_AND_CKPT_TASK.md。

## 资源交接

无活跃构建、编译、传输或本任务GPU服务。BuildKit Pod保留，生命周期由root管理。37路径内完整镜像/权重/来源证据保留；34原件、118历史环境、同事Pod和GPU未动。仅删除明确归属本任务的短写块及本机chunk临时文件。
