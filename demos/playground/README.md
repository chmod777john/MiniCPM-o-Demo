# MiniCPM-o Playground

独立的 React / TypeScript 桌面体验应用。八种体验在同一页面切换：单工、音频双工、视频双工、Board、UI Card、SubAgent、Rejection 和 Standalone Event。

## 启动

在这个目录使用 Node.js 22.12+：

```bash
npm ci
npm run dev
```

默认监听 `127.0.0.1:5186`。远程开发可以通过 SSH 转发该端口，用本机的 `http://localhost:5186` 打开；音视频权限需要 HTTPS 或 localhost。

```bash
npm run build
npm run preview
```

`dist/` 是可独立托管的静态应用，采用相对资源路径。构建时需要完整仓库，因为协议客户端和部分场景通过相对路径复用现有模块。

## 连接模型和工具

在页面右上的“开发者模式”中设置：

- **模型 API 地址**：API 根地址，或完整的 `/v1/realtime` 地址。推理 WebSocket 由浏览器直接连接。单工使用 `mode=chat`；普通双工使用 `mode=audio|video`；FC 使用 v5 `mode=audio|omni`。应填写支持相应协议的端点。
- **工具服务地址**：现有 Demo Gallery 服务根地址。FC 场景从它读取预设、参考音频并执行工具；它与模型 API 是两个独立服务。
- **场景预设**：加载后可查看 system segments、tools 和参考音色；也可以上传自己的参考音频。应用保留服务提供的预设内容，不自动改写模型提示词。

工具服务地址为空时使用页面同源。在本地开发中，可以配置 `.env.local`：

```dotenv
PLAYGROUND_GALLERY_TARGET=http://127.0.0.1:10032
```

该开发代理只转发 `/api/demos/**` 与 `/demos/**`，不转发推理请求。修改环境变量后重启开发服务。生产静态托管需单独配置这些路由，或填写允许跨域访问的工具服务地址。`npm run preview` 仅用于查看构建后的页面，不包含工具代理。

当前连接表单支持仓库原生 WebSocket 协议，尚未接入 MAP 的 API Key 子协议鉴权。页面不会自动选择线上端点，也不会在打开时启动模型会话或 SubAgent 任务。

## 代码组织

```text
playground/
  app/                 # 应用入口、导航、统一体验布局与样式
  shared/
    ui/                # 消息、输入框、设置、弹窗和场景容器
    session/           # 不依赖 React 的会话状态、协议桥接和生命周期
    media/             # 附件、录音与参考音色转换
  turnbased/           # 单工配置与原生传输
  audio-duplex/
  video-duplex/
  board/
  uicard/
  subagent/
  rejection/
  standalone/
```

整个目录是一个应用；各体验目录是内部模块。SubAgent 和外部事件面板使用 React；Board、UI Card、Rejection 的专用展示暂通过隔离容器复用现有场景实现。协议、麦克风、摄像头和音频播放模块只读复用仓库已有实现。

切换体验或新建对话会结束当前会话并清空本页记录；有会话或内容时先确认。停止会话释放浏览器媒体与连接，SubAgent 停止本地轮询；已提交到工具服务的后台任务可能继续运行。仅连接地址及普通选项保存在本机 localStorage，对话、音频和 system 内容不持久化。

视觉素材和出处记录在 `shared/assets/README.md`；文字颜色、主视觉与麦克风按钮采用设计稿，导航按现有八个体验组织。当前交付范围为桌面布局。会话状态与媒体处理独立于布局，后续移动端可复用；旧桌面页面、旧移动端和服务端均未修改。本分支已完成类型检查与生产构建；真实模型、设备权限及各工具的端到端联调仍待进行。
