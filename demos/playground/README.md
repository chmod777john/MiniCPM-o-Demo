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

所有体验只配置 API Base 和 Model ID。填写 API 地址后自动读取 `/v1/models`，切换模型直接使用同一目录中的详情，无需手动发现或读取。空闲时每 30 秒刷新目录；会话期间锁定连接设置。服务不再提供所选 ID 时保留选择并提示，不静默切换到另一个模型。

模型信息与体验预设分开：

- **模型服务**：`/v1/models` 返回模型 ID、`loaded_model`、能力及默认配置。本次实际模型和生效参数以 `session.created` / `session.resumed` 为准。各体验共用模型目录，新打开的体验可继承已有连接。
- **单工／普通双工预设**：本应用的 `server/cases.json` 提供三个 case 的设置 schema、默认值、语言预设及参考音频。`GET /api/demos/playground/cases/<id>/defaults` 返回配置；`/api/demos/playground/audio/<id>` 提供音频。
- **FC 预设与工具**：Gallery 的 `/api/demos/<id>/defaults` 提供原场景预设与工具。场景执行继续调用原工具接口。
- **分享服务**：独立配置，负责接收录制和生成分享链接。
- **UI Card 外部工具**：应用服务通过 `UICARD_TOOLS_CONFIG` 读取部署者的私有 JSON 配置，格式见 `../uicard/README.md`。如果该 case 单独运行应用后端，可用 `PLAYGROUND_UICARD_TARGET` 指定其地址，仅代理 `/api/demos/uicard`。搜索、天气、地图的服务端 key 不下发浏览器；缺少的供应商配置仍会明确提示。

推理 WebSocket 始终由浏览器直接连接 `/v1/realtime?model=<ID>&mode=chat|audio|video|omni`。Gateway 将 Model ID 映射到内部池再调度实例；不提供 `/e/...` 兼容路由。单工、普通双工与 FC 可以选择同一模型；具体能力以目录声明为准。

配置步骤：

1. 复制 `.env.example` 为 `.env.local`，填写 Gallery 应用服务；可设置 `VITE_PLAYGROUND_DEFAULT_API` 为此次部署的 API Base，`VITE_PLAYGROUND_DEFAULT_MODEL` 为默认 Model ID。
2. 启动后等待预设加载，在右上角选择模型；浏览器遇到可信开发服务的自签证书时，先直接打开该服务确认访问。
3. 点击开始并允许麦克风／摄像头访问。开发者模式可查看模型身份、当前 preset 和会话实际返回的配置。

```dotenv
PLAYGROUND_GALLERY_TARGET=http://127.0.0.1:10032
# 可选：默认 API Base、模型 ID、录制分享服务
# VITE_PLAYGROUND_DEFAULT_API=https://api.example.com
# VITE_PLAYGROUND_DEFAULT_MODEL=minicpm-o5
# PLAYGROUND_SESSION_TARGET=http://127.0.0.1:9000
# PLAYGROUND_PROXY_TLS_INSECURE=1
```

`VITE_PLAYGROUND_DEFAULT_API` 改变后，会在下次加载时更新浏览器保存的连接地址一次，其他体验参数保留。之后手动选择的地址仍可记忆。该变量只有公开端点地址，不含凭证。

开发与预览由 Vite 挂载本应用配置后端；独立托管静态文件时，运行 `npm run serve:app`（默认 `127.0.0.1:5187`），反向代理 `/api/demos/playground/` 到它，其他 `/api/demos/` 路由到 Gallery。配置后端只提供设置和音频，不加载模型、不转发推理。生产部署需一并提供仓库中的参考音频，或设置 `PLAYGROUND_AUDIO_ROOT`。

预设、工具及会话分享统一通过同源应用接口访问，由部署配置上游地址；这些地址不在页面中填写，也不从旧浏览器设置恢复。首次进入加载应用默认值，再保留用户可记忆的设置；切换 preset 仅替换该预设声明的字段；恢复默认重新加载当前应用配置。设备选择保留在浏览器端。未成功加载配置前不能开始会话。

## 配置和行为

- 单工：流式音频从首块起提供播放、暂停、拖动和下载，停止生成保留已输出的部分回答。支持文本与图片／音频／视频附件，保留对话历史；用户消息发送后不提供编辑或重发入口，AI 回答下方横向排列重新生成和复制。System 保留多模态内容设置。回答中不展示 TTS 控制标记、耗时、token 统计或请求编号。默认流式、1024 tokens、关闭 TTS；TTS 可独立指定音色或从 System 提取。非流式语音由播放器手动播放。
- 普通双工：区分 LLM 参考音频与 TTS 音色；支持滑窗、Listen/Speak 阈值、实时／文件／混音输入、设备选择、暂停和强制聆听。Max KV 是客户端停止阈值。视频保留原始分辨率，区分 Fixed HD 和 Strategy HD。
- FC：使用各场景原预设及其 tools，显示端点模型身份和会话实际配置。预算、调度、tts_pad 沿用端点默认值。文件回放限 Audio 模式；检查点只有服务报告可用时才能使用。
- 录制：普通音频双工输出双声道 WAV；视频支持浏览器可用的录制格式、码率及录入字幕。分享依赖原应用服务提供的会话存储接口。

FC 主体的“协议与指标”和普通体验设置里的“协议事件”可筛选、展开、回放音频并导出最近 200 条记录（事件本身约 24 MiB 上限）；单工另外支持 `.msgpack` 对话导出。界面配置和源码映射已对照旧页面；构建通过不代表真实模型、浏览器设备及外部工具已完成端到端验收。

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

整个目录是一个应用；各体验目录是内部模块。SubAgent 和外部事件面板使用 React；Board、UI Card、Rejection 的专用展示暂通过隔离容器复用现有场景实现。协议和 FC 媒体模块只读复用仓库已有实现；普通双工文件处理与视频录制在本目录适配 DOM 和构建路径，保留原有截帧、混音、归一化和字幕算法。

切换体验或新建对话会结束当前会话并清空本页记录；有会话或内容时先确认。停止会话释放浏览器媒体与连接，SubAgent 停止本地轮询；已提交到工具服务的后台任务可能继续运行。连接地址和原页面支持记忆的参数按体验分别保存到本机 localStorage；音频双工包括自定义文本提示词；视频双工刷新后恢复页面／服务默认值。Max KV、滑窗策略从当前页面／服务默认值重新初始化。对话、上传文件、音频和 FC system segments 不持久化。

视觉素材和出处记录在 `shared/assets/README.md`；文字颜色、主视觉与麦克风按钮采用设计稿，导航按现有八个体验组织。当前交付范围为桌面布局。会话状态与媒体处理独立于布局，后续移动端可复用；旧桌面页面和旧移动端源码保留；本分支 Gateway 已切换到 Model ID 接口，这些旧页面需迁移调用方式后才能接入新 Gateway。本分支已完成类型检查与生产构建；真实模型、设备权限及各工具的端到端联调仍待进行。

## 行为核对记录 · 2026-10-22

原版基准为 `a3e7c70` 所在仓库的原页面。三个独立审计线程先枚举原行为，再逐项映射候选：单工 66 项、普通音视频双工 65 项、FC 55 项，共 186 个契约条目，**不是通过测试的数量**。原发现保留，后续复核追加关闭记录。

详细清单、修复处置与构建记录放在工作区之外：`/user/weihongliang/analysis-runs/20261022-o5-react-playground/parity-audit/`；以 `ROOT_RESOLUTION.md` 和各报告末尾复核为准。此处只保留简短索引，避免将截图、运行日志和重复审计记录加入应用仓库。

这一阶段完成参数／来源展示、媒体与分享生命周期、流式音频与导出、FC 场景及恢复行为的源码修复。已通过类型检查、生产构建与静态桌面渲染；尚未执行真实模型、设备或工具服务的端到端验收。

尚未完整对齐：英文界面与原 FAQ、生成期间起草下一轮、部分录音／附件快捷操作、普通双工文件预检查与部分调试展示。具体缺口保留在清单，不能将此版本视作旧页面的完全替代。原页面继续保留。

## 端点选择接入 · 2026-10-23

- FC 顶部端点选择器复用原 Gallery 的发现和合并逻辑；配置、端点列表错误会显示具体原因。
- 默认配置读取通过部署者配置的 Gallery 路由；开发代理可显式允许可信自签上游证书，推理保持浏览器直连。
- 初版仅在相同协议间继承；后续统一为模型端点目录，协议入口由各体验分别选择。
- 验收范围：配置接口只读检查、源码核对、类型检查与生产构建；不包含真实会话、麦克风与工具调用端到端验收。详细记录见工作区外 `analysis-runs/20261022-o5-react-playground/endpoint-discovery.md`。

## FC 主体布局 · 2026-10-23

- 场景结果位于主体上方；AI Speech 与 Think / Tool-Call 在下方并排、默认展开，进入页面即可看到空状态。可用宽度较小时上下排列。
- 轨迹复用原 `NsSegmentView` 的 Unit、Token、听说 lane、pending 和预算截断展示；预算与最新 used 放在面板顶部。面板可收起或全屏，收起保留事件状态。
- 原始协议事件与运行指标从主体“协议与指标”打开；配置、Loaded model、runtime 和 checkpoint 仍在设置栏。
- 验收范围：类型检查、生产构建和桌面空状态截图；没有启动模型会话或工具任务，实时 token／工具执行和全屏交互尚未实测。详细记录位于工作区外 `analysis-runs/20261022-o5-react-playground/fc-main-layout.md`。
