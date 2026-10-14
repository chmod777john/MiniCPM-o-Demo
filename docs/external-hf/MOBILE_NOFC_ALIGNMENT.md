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

## Root集成验证

原提交 `c0259866` 合入为 `47cb5315`，后端桥接 `1b761e12`，HF `5cccbf7`。合入后重新执行 build:static 和15项CPU契约均通过。

双A100 TP job1218128 对helper实际生成的音频/视频payload各跑2个unit，均收到unit_finalized；session_config确认Basic8000/6000、binary0.3、视频slice2。单工12项API、取消/错误后的新会话也通过。实际手机浏览器仍未验，API及CPU契约不能替代设备交互验收。精简结果在 `evidence/20261014-simplex/final/`。

### 主动取消补充

移动补充提交 `add32c8e` 合入为 `08c22855`：已建立的聊天会话停止时发送session.close/client_cancel，立即停止播放并保存中断状态；收到ACK后关闭，1.5秒兜底。排队/连接阶段直接取消。

最终重新构建通过；24项CPU行为检查通过，其中9项执行真实App函数检查取消、迟到消息、重试清理、切换及卸载。最终构建后的20个页面/资源HTTP200（含桌面单工和移动入口），静态资源SHA256在final-static-pages.json。浏览器真机验收仍由用户完成。
