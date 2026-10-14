# 移动端 no-FC 对齐验收记录

日期：2026-10-14。基准 Demo `24d02d68f3bf6b8875ffc70dd5efb92476c77c8e`。
实现范围及构建方法见 `frontend/mobile/README.md`。

## 修复

- 单工流式与非流式均走现有 chat WebSocket，发送完整历史；补齐视频 omni、HD slice、do_sample 与 TTS 参数。
- React 音频双工采用 mode=audio、服务默认滑窗参数、独立 TTS 音色、KV上限/指标和普通断线重连。
- 视频继续共用桌面 Omni 运行时；移动 HTML 改为可重复生成，React 设置面板映射到相同 DOM 和音色 controller。
- 修复移动音色 custom event 无接收者和异步默认音色覆盖手选音色的问题。双工隐藏无效 length penalty；单工无新增 thinking+TTS 行为。
- build:static 同时构建 React App、移动 Omni 外壳和设置 widget；Vite增加/v1 WebSocket代理。

## 本分支验证（CPU）

环境：项目内 `frontend/mobile/node_modules`；Node22.23.3、TypeScript6、Vite8.3.4；未全局安装。

- `npm run build:static`：通过，含TypeScript/App/widget/HTML生成。
- `npm run test:contracts`：15/15通过，覆盖完整历史、流/非流payload、图片/视频、TTS/独立音色、滑窗/HD、共享RealtimeSession+mock WS音视频发送/指标/关闭、参考音色异步生命周期、桌面/移动HTML ID一致性。
- 新模块ESLint：通过。全项目ESLint仍有基准遗留46项，按文件/rule/message比较无新增。
- JavaScript语法与`git diff --check`：通过。

原始日志：`/user/weihongliang/analysis-runs/20261014-hf-simplex/mobile/{build.log,contracts.log,lint-current.json,lint-baseline.json,REPORT.md}`。

不包含真实手机麦克风/摄像头、浏览器交互或GPU实测；推理API端对端由root在HF单工主任务统一验收。没有修改HF/accel/后端或运行中的服务。

`static/mobile/` 按仓库惯例忽略，合入/部署后需运行build:static。`static/mobile-omni/index.html`和widget产物随代码提交。
