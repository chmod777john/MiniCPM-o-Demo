# 34 浏览器体验服务

2026-10-14 08:21 UTC由root启动。**当前已停止**：2026-10-14 09:33 UTC按用户授权停止gateway与worker，释放双卡给pip/wheel镜像验证；8008体验入口暂不可用。

入口：`https://34.66.244.63:8008/`。自签名证书；首次需在浏览器确认继续访问并授权麦克风/摄像头。

| 页面 | 路径 |
|---|---|
| 桌面单工 | `/turnbased` |
| 普通音频双工 | `/audio_duplex` |
| 普通音视频双工 | `/omni` |
| 移动单工/音频双工 | `/mobile/` |
| 移动视频双工 | `/mobile-omni/` |
| 新版应用入口 | `/` |

## 运行身份

固定镜像沿用 `DOCKER_LIS_SPLIT_ACCEPTANCE.md` 的v11 worker/gateway，HF/accel/demo无代码修改。
34两张PRO6000使用Pro5000 TP2预设；原始FC checkpoint为job1054182 iter1275。

- owner：root，服务用途为用户浏览器验收。
- worker：`o5-browser-worker-20261014`，容器ID `7cab0fe74f9fd6b8f118f541cb2e9b8e5d34ee2c04e3cb9054756ca77a7d0d77`。
- gateway：`o5-browser-gateway-20261014`，容器ID `ac0dd45aecc9f6abcea88dc87e770a38731b049e7dd6692bf3ecd7ef1a1eb66b`。
- network：`o5-browser-20261014`。
- 公网8008映射容器HTTPS18006；registry18025仅绑定host localhost。
- 现场命令和状态：`/home/weihongliang/o5-docker-lis-20261014/browser-20261014/{commands.json,service-state.json}`。
- gateway启用页面 `GATEWAY_API_ONLY=0`，平台只读配置保留recording=false。

初次使用18024端口公网超时，随后仅重建本任务gateway映射到公网可达8008；worker保持运行。
无云防火墙修改，未触碰原有443/8006等旧服务，未推送镜像或部署LIS。

## 本轮检查

外部HTTPS下6个页面、health/status及所引用19个JS/CSS资源HTTP200，worker idle且queue0。
经公网WSS完成文字和真实音频两项API；文字返回“你好。”，音频88320bytes、finite/nonzero。
这些是部署检查；用户自行做浏览器、麦克风、摄像头交互验收。

原始检查：`/user/weihongliang/analysis-runs/20261014-host34-browser/`。
停止时已核对上述完整容器身份，两个容器均exit0；两卡各3MiB且无GPU计算进程。其他服务保持运行，容器、镜像和模型文件均保留。
停止证据：`/user/weihongliang/analysis-runs/20261014-docker-pip-install/host34-release.stdout.json`。
