> 📍 **本 CLAUDE.md** — 所属目录 `/user/sunweiyue/lib/MiniCPM-o-Demo-modelbest/demos/` · 项目级（Demo Gallery 开发规范）

# Demo Gallery：怎么新加 / 修改一个 Demo

**Demo 与 API 是两件事。** Demo 服务（`demo_server.py`）只有页面与各 Demo 自己的后端，不加载模型；
页面上选一个 API 端点，浏览器按协议 v4 直接连它（`docs/fc-duplex/semantic-realtime-api-v4.md`）。
Demo 侧代码**不许 import API 侧任何模块**（`core/`、`py_backend/`、`gateway*`、`control_plane/`…），
有守卫测试 `tests/test_api_demo_boundary.py`。

**核心性质：一个 Demo = `demos/<id>/` 一个文件夹。加、改、删 Demo 都不碰任何共享文件。**
这条有测试强制（`tests/test_gallery_pluggability.py`），不是靠自觉。

API 细节（`@demo` 参数、场景契约字段、工具调用链路）的单一事实源是同目录的
**`README.md`**，动手前先读它。本文件只写硬规则和流程。

## [MUST NOT] 共享层不许动

新 Demo 的开发**不得修改**以下任何文件：

```text
demo_server.py              Demo 服务主程序
demo_gallery/               后端 core（注册 / 发现 / 挂载 / 页面模板）
static/demo-gallery/        前端 core（shell、realtime client、通用组件）
```

- 共享层里**不允许出现任何具体 Demo 的名字**（有 `git grep` 守卫测试）。
- Demo 的 API（包括工具接口）只能挂在自己的 `router` 上，由 Gallery 挂到
  `/api/demos/<id>/**`。**禁止**往 Demo 服务主路由加接口。
- 如果你觉得「不改共享层就做不出来」，那是共享层缺了一个通用扩展点。**停下来问用户**，
  不要在共享层里写 `if demo_id == ...` 这种特判。

## 新建 Demo 的流程

1. **复制脚手架**（`_template` 以 `_` 开头，不会出现在索引页）：
   ```bash
   cp -r demos/_template demos/<id>
   cd demos/<id>
   sed -i 's/_template/<id>/g' __init__.py demo.py web/index.js web/_template.css
   mv web/_template.css web/<id>.css
   ```
2. **改 `demo.py`**：`title` / `summary` / `order`；需要后端接口就往 `router` 上加。
3. **写 `web/scene.js`**：右上主面板的全部内容。会话、麦克风、播放、system 编辑、
   think 账本都由 shell 负责，场景不要重做。
4. **写 `web/<id>.css`**：每条顶层选择器以 `[data-demo="<id>"]` 开头。
5. **测试 + 浏览器自检**（见下文「交付前自检」）。

现成范例：`demos/board/`（工具调用 + 搜图 + 额外静态目录）、`demos/rejection/`（纯 think 解析）、`demos/standalone/`（向模型注入 standalone_event、展示 Unit 回执，不带工具）、`demos/subagent/`（工具调用交给 Claude Code，每会话独立工作区；部署要求见下文「部署 subagent」）。
做类似功能时先照抄它们的写法。

## [MUST] 硬约束

- **id 三处一致**：目录名 = `@demo(demo_id=...)` = 前端 `registerDemo({ id })`。id 同时是
  URL 段（`/demos/<id>`），不一致在注册时就会抛错。
- **前端 import 共享模块用相对路径** `../../../static/demo-gallery/core/xxx.js`，
  不要写绝对 URL，也不要改挂载前缀。URL 与磁盘路径同构，同一条相对路径在浏览器和
  node 测试里才会解析到同一个文件。
- **调自己的后端用 `api.url('<path>')`**，不写死 `/api/demos/<id>/...`。
- **`render()` 必须自己画空状态**：页面骨架里没有任何场景内容。
- **CSS 不得污染别人**：不加命名空间的规则会被测试拦下。`@keyframes` 名以 `<id>-` 开头。
  不要改 `stageEl` 的 class 去「解除」shell 布局。
- **通用的东西放共享层由用户决定**：别把麦克风提示、模型信息栏这类通用样式/逻辑复制进
  Demo，也别自己挪进共享层。
- **代码里不许出现任何人的私人路径或凭证默认值**（家目录、`~/.claude/settings.json`、某人装的程序、
  API key）。会动用账号 / 密钥 / 计费资源的配置一律走环境变量，**没配就拒绝服务**，不许回退到某人的默认值。
  原因：集群容器都以 root 运行，别人的 `/user/<name>` 对你可读——默认值写的是谁，别人克隆部署后就悄悄用谁的账号。
- **模型口径用端点默认值**：预算、调度方式、tts_pad 协议上是 Session 参数，但 Demo **不发送**，
  直接用 API 端点给的默认值（`<端点>/v1/info` 的 `model.session_defaults`，建会话后以
  `session.created.session_config` 为准）。换模型 = 换端点或改端点的 Worker 模板，不在 Demo 里改。
  preset 只描述 system segments 与 TTS prompt 音频，不要再做「只差 tts_pad / 只差 budget」的成对 preset。
- **音频写 Demo 服务所在机器上的绝对路径**，建会话时由 shell 读成 base64（`/api/demos/_audio`）。
  只能读 `DEMO_AUDIO_ROOTS` 白名单内的 16 kHz 单声道 WAV；新素材放不进白名单就加白名单目录，别改代码绕过。
- **推理忠实**：Demo 只负责展示，**不许为了效果改推理语义**（加 logit bias、改 prompt、
  补写模型没生成的 token 等）。完整规则见 `.cursor/rules/model-faithful-inference.mdc`。

## 交付前自检

```bash
cd /user/sunweiyue/lib/MiniCPM-o-Demo-modelbest
/user/sunweiyue/runtime/o5_eval/venv-o5-accel/bin/python -m pytest -q \
  tests/test_gallery_pluggability.py demos -p no:cacheprovider
```

- 新 Demo 自动被守卫测试覆盖（id 一致、场景契约、CSS 命名空间、路由不逃逸、
  共享层无 Demo 名），不需要登记。
- Demo 自己的逻辑测试放 `demos/<id>/tests/`。前端纯逻辑抽成独立 JS 模块，
  用 `tests/conftest.py` 的 `run_node` 直接 import 测。**不要用正则从源码里抠函数**。
- **浏览器实际打开 `/demos/<id>` 看一遍**：零 JS 报错、零 404、空状态和布局正常。
  只跑单元测试不算完成。

> 全量 `pytest tests demos` 里有一批测试需要真实模型权重或已过期
> （`test_chat` / `test_duplex` / `test_e2e` / `test_integration` / `test_streaming` /
> `test_queue`），本机必然失败，与 Demo 开发无关。

## ⚠️ 线上 Demo 服务跑的就是这个目录

线上 Demo 服务（公网 `https://47.95.219.248:7032/demos`）是开发机上的一个进程，直接从
`/user/sunweiyue/lib/MiniCPM-o-Demo-modelbest` 启动（`scripts/start_demo_server.sh <env 文件>`）。
它**不加载模型**；页面里连的 API 是另一套服务（公网 7029，见仓库根 `CLAUDE.md`「部署」）。所以：

- **前端文件（JS / CSS / 页面模板）一保存，线上立刻变**；Python 改动要重启 Demo 服务进程才生效
  （秒级，不涉及模型）。前后端不同步时页面可能直接坏掉。
- 开发半成品时**不要直接在这个工作区里改**，用 `git worktree` 另开一份，在别的端口起一个 Demo 服务自测。
- 新 Demo 上线要重启 Demo 服务：Gallery 只在启动时扫描 `demos/`。

## 部署 subagent：必须用你自己的 Claude Code 与网关账号

subagent 会在后台起 Claude Code（联网、消耗网关额度；跑在 Demo 服务所在机器上），部署者必须在**自己的**
Demo 服务 env 文件里设两个环境变量，
缺任何一个，subagent 的 API 全部返回 503（其余 Demo 不受影响）：

| 变量 | 含义 |
|---|---|
| `SUBAGENT_CLAUDE_BIN` | **你自己的** `claude` 可执行文件路径 |
| `SUBAGENT_CLAUDE_SETTINGS` | **你自己的** Claude Code settings 文件；只取其中 `env` 段（网关地址、API key 等）传给后台 |

**[MUST NOT] 不要把这两个路径指向别人的目录**——哪怕读得到。那等于用别人的 API key 替你付费。
**key 用项目 key，不要用个人会话 key**：给 subagent 单独建一份 settings 文件（只含 `env`：网关地址、项目 key、
网关必需的 disable 开关），放在仓库外、600 权限。个人 CloudCLI 会话 key 有 per-key 限速（llm-center 实测 50 RPM），
且用量记在个人名下。
不想开放 subagent 就两个都不设，它会保持禁用。
本 Demo 不设访问密码：打开页面即可用，后台调用记在部署者配置的那把 key 上。

## 删除 Demo

删掉 `demos/<id>/` 整个文件夹，重启服务即可。
