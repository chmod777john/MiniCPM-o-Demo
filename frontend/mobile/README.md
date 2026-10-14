# Mobile Frontend

`frontend/mobile` 是这次移动端适配的独立子前端，技术栈默认是：

- `React 19`
- `TypeScript 6`
- `Vite 8`
- `bun` 作为默认包管理器

## 推荐用法

先安装依赖：

```bash
bun install
```

本机如果要直接跑最新 `vite`，推荐显式带上 `--bun`：

```bash
bun run --bun dev
bun run --bun build
bun run --bun build:static
bun run --bun preview
bun run --bun lint
```

如果要把构建产物发布到仓库现有的 `static/` 目录，供 gateway 通过 `/mobile` 入口直接访问，用：

```bash
bun run --bun build:static
```

执行后会把 `dist/` 同步到 `static/mobile/`。

## 本地联调代理

开发预览默认会把以下接口代理到：

```text
https://127.0.0.1:8025
```

包括：

- `/api/*`
- `/status`
- `/health`
- `/workers`
- `/ws/*`
- `/s/*`

如果你要切到别的 gateway，可以在启动前覆盖：

```bash
VITE_PROXY_TARGET=https://127.0.0.1:8025 bun run --bun dev
```

之所以保留这层 dev proxy，是为了让 `frontend/mobile` 独立跑在 `8032` 时也能直接连真实后端；而真正发布到 `/mobile` 后又可以继续走同源路径，不需要改业务代码。

## npm 兼容路径

脚本本身仍然保持标准的 `vite` / `tsc` 形式，没有为了 bun 做很重的魔改。
如果后续有人更习惯 `npm`，也可以继续走常规命令：

```bash
npm install
npm run dev
npm run build
npm run build:static
npm run preview
npm run lint
```

但 `npm` 路径需要本机 `Node >= 20.19` 或 `>= 22.12`，因为 `Vite 8` 的官方要求就是这个版本线。

## 当前已知问题

这次已经遇到两个和 bun 相关、值得明确记录的问题：

1. `bun create vite@latest` 在当前环境下会因为 `node:util` 的 `styleText` 导出报错，不能顺利完成脚手架初始化。
2. 在这台机器上，直接执行 `bun run dev` 仍可能让 `vite` 落回系统 `Node 16`，从而触发版本不满足。显式使用 `bun run --bun <script>` 可以绕开这个问题。

所以当前策略是：

- 依赖管理和本地开发优先使用 `bun`
- 保留 `npm` 兼容脚本，避免项目结构过度偏离常见前端习惯
- 对 bun 的已知兼容问题直接记录，不把它藏在工程细节里

## no-FC API 对齐（2026-10-14）

移动端只组装请求、采集媒体和播放结果。单工/双工推理均经现有 API 使用服务启动时指定的外部 HF；移动端没有 modeling 或推理循环。

| 入口 | 协议 | 与桌面共用部分 |
| --- | --- | --- |
| `/mobile/` 单工 | `/v1/realtime?mode=chat`；每轮完整 messages；`streaming` 控制输出 | HF chat API，文字/音频 delta 与 response.done |
| `/mobile/` 音频通话 | `/v1/realtime?mode=audio` | `RealtimeSession`、音频采集 worklet、播放器 |
| `/mobile-omni/` 视频通话 | `/v1/realtime?mode=video` | 桌面 Omni DOM、`omni-app.js`、RealtimeSession；React 设置面板和移动 CSS |

单工的流式/非流式都使用 chat WebSocket；不再调用已移除的 `/api/chat`。两者均发送完整历史，保持 `generation.max_new_tokens/do_sample/length_penalty`、TTS 音色和 `use_tts_template`。图片 HD 为 2 slices，普通图片为 1；包含视频时 `omni_mode=true`、slice=1、TTS mode=omni。未增加思考开关，`enable_thinking=false` 与桌面默认一致；不提供单工 KV/滑窗控件。

双工设置包括：TTS 开关与独立 TTS 音色、Basic/Off/KV surgery（服从服务允许列表）、Basic 高低阈值、KV surgery 参数、Listen/Speak 二选一与阈值、播放缓冲、最大 KV，以及视频固定/策略 HD。最大 KV 是客户端停止会话的上限，最高 262144；Basic 高低阈值才是模型滑窗参数，两者分别生效。`/api/frontend_defaults` 提供模式/阈值/播放缓冲默认值。会话开始后参数锁定，重启会话使用新选项。旧 length_penalty 对 HF 普通双工无效，因此双工不再展示/发送该参数。

音频页显示真实 KV 长度和滑窗次数；视频页镜像共享桌面运行时的 KV、滑窗次数与最近裁剪长度。普通断线重连沿用桌面客户端策略。slot 协议继续由服务/HF 与 checkpoint 决定，移动端没有独立 slot 开关。

### 构建与检查

Node >=22.12（或项目声明的 Bun）下：

```bash
npm install --ignore-scripts --no-package-lock
npm run build:static
npm run test:contracts
```

所有依赖位于 `frontend/mobile/node_modules`，不做全局安装。生产部署必须执行 `build:static`：`static/mobile/` 是 Git 忽略的构建产物。该命令也重新生成移动 Omni 外壳和 React 设置组件，防止只编译手机聊天页却漏掉视频设置。

`sync:omni` 从 `static/omni/omni.html` 生成 `static/mobile-omni/index.html`，保留全部桌面 DOM ID，仅添加移动壳。请修改桌面 HTML 或移动 CSS/bridge 后重新生成，勿手改生成 HTML。共享引用音频状态通过 Omni UI bridge 读取/写入，不再发没有监听者的自定义事件。

本地 Vite 代理也包含 `/v1/*` WebSocket。契约检查覆盖完整历史、单工视频/图片/TTS、双工准备参数、真实 RealtimeSession 与 mock WebSocket 的发送/指标/关闭、默认音色异步覆盖，以及桌面/移动 DOM ID 一致性；它们不等于真实手机麦克风/摄像头或 GPU API 验收。
