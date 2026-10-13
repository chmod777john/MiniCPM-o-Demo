# 普通双工页面迁移记录

日期：2026-10-13。Owner：Codex/root。

## 用户要求与边界

1. 把旧普通音视频双工页面功能迁入当前外部 HF adapter。
2. 接通选项 API。
3. 保留新版 demos，不搬旧 FC applications。
4. 保留 `modeling/external_o5_facade.py` 桥接方式；本轮暂缓单工。

## 固定来源

- 目标：`/user/weihongliang/worktrees/o5-fc-dev-external-hf-20261013`。
- 本轮起点：`20ed37ae7299839055d1086bc2d05e05b4081a43`，上游基线
  `o5-fc-dev@732b075841f837d67d9f512835cbf1aec174911e`。
- 页面来源：`/user/weihongliang/demo-adapter-fc-no-fc-3d-rope-20261009-batch-tts-20261010-exact-window-20261010-window-events-20261010`，
  `4df255f2698900bf17ec16b2b680b8d07882aeea`。
- 外部 HF：`minicpm-o-5-fix-3d-rope-o5fc-20261013@335bfddaf1c32d455f87d2e5f699b0f71c6d65a4`。
- 外部 accel：`…-direct-table-20261010-o5fc@6e33931cdf7e1fcff2f896d706bb73b949fc4a39`；全路径见 DEPLOYMENT.md。

## 迁移方式

用固定 source commit 的 `git show <sha>:<path>` 复制普通页面及其配套组件，
再做新 transport 的适配；没有整仓或旧 gateway 覆盖。

复制的普通 UI 范围：

- `static/omni/{omni.html,omni-app.js,omni.css}`
- `static/audio-duplex/{audio_duplex.html,audio-duplex-app.js,audio-duplex.css}`
- `static/duplex/ui/duplex-ui.js`、`static/duplex/duplex-shared.css`
- `static/duplex/lib/{session-video-recorder.js,recording-settings.js}`

其后增量修改：当前 `realtime-session.js` 保留新版事件协议，连接新 window metrics、
字幕增量、可选恢复；共享 i18n 增补对应字段；音频设备选择器支持禁用偏好持久化。
视频页面每次重载回到页面/服务默认，音频保留原常用设置但 MaxKV/窗口模式重置。
移除当前 HF 无实现的 Draft / veto / Length Penalty 控件。

## 后端改动

- `ExternalO5DuplexConfig` 增加固定/策略 HD，水位校验与 256K 页面范围对应。
- `DuplexView` 和 `PyTorchBackend` 传递 optional slices、LS 输出、普通会话配置和指标。
- facade 把 window/LS/TTS 参数交给 HF；策略 HD 仅选择下一 unit 的 processor 参数。
- 滑窗 observer 包装已安装的 decoder 方法，只观察计数/前后KV/host时间；
  不复制或改写裁剪算法，不增加 GPU 同步。保留 wraps 链。
- `py_backend/server.py` 在普通 finalize 后回传单独指标；FC runtime 保持现有分支。
- gateway 的 `ordinary_reconnect.py` 保留原 relay 的 worker 连接，支持普通页面重绑；
  未收到 session.created token 时断线不保留无主队列，恢复时锁保护 socket 替换。
  worker 已分配后的 queue_done 通知也纳入原有 finally 释放范围。
- 普通页只在 finalize 后判断 MaxKV，避免先于裁剪自动停止。

HF、accel、旧页面来源目录未修改。本轮没有操作 GPU、服务或环境。

## 源码审查结论与尚未覆盖

已沿页面 → session.init/input.append → gateway → worker → backend → View → facade → HF
阅读参数和事件流；也检查了 FC 分支、会话复位、策略切片、finalize 顺序和恢复资源释放。
TTS 源码虽保留初始化时的 processors 字段，但当前 generate_chunk 会按传入 penalty
重新生成，因此没有引入额外的 TTS 缓存补丁。

上述是源码审查，**不是运行验收**。本轮未新增/执行测试，未启动浏览器、API 服务或 GPU。`git diff --check` 无空白错误。
未覆盖：真实音视频生成、Basic/KV surgery 裁剪、切片策略和TTS开关、断网恢复/到期释放、
长文件与连续会话运行。此前验收文件仅说明本轮起点的状态。

断线恢复不是音频无损续传：中断期间不补发新输入；恢复只保留已被 gateway 接收的
会话状态及有限输出。多进程 gateway 需要同进程路由。性能没有本轮实测结论。
