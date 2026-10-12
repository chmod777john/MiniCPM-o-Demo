> 📍 **本 CLAUDE.md** — 所属目录 `/user/sunweiyue/lib/MiniCPM-o-Demo-modelbest/demos/` · 项目级（Demo Gallery 开发规范）

# Demo Gallery：怎么新加 / 修改一个 Demo

**核心性质：一个 Demo = `demos/<id>/` 一个文件夹。加、改、删 Demo 都不碰任何共享文件。**
这条有测试强制（`tests/test_gallery_pluggability.py`），不是靠自觉。

API 细节（`@demo` 参数、场景契约字段、工具调用链路）的单一事实源是同目录的
**`README.md`**，动手前先读它。本文件只写硬规则和流程。

## [MUST NOT] 共享层不许动

新 Demo 的开发**不得修改**以下任何文件：

```text
gateway.py                  Gateway 主程序
demo_gallery/               后端 core（注册 / 发现 / 挂载 / 页面模板）
static/demo-gallery/        前端 core（shell、realtime client、通用组件）
```

- 共享层里**不允许出现任何具体 Demo 的名字**（有 `git grep` 守卫测试）。
- Demo 的 API（包括工具接口）只能挂在自己的 `router` 上，由 Gallery 挂到
  `/api/demos/<id>/**`。**禁止**往 Gateway 主路由加接口。
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

## ⚠️ 线上部署跑的就是这个目录

廊坊 deploy 池的 Demo 服务（公网 `https://47.95.219.248:7029/demos`）直接从
`/user/sunweiyue/lib/MiniCPM-o-Demo-modelbest` 启动。所以：

- **前端文件（JS / CSS / 页面模板）一保存，线上立刻变**；Python 改动要重启 job 才生效。
  前后端不同步时页面可能直接坏掉。
- 开发半成品时**不要直接在这个工作区里改**，用 `git worktree` 另开一份。要上线时再合回来，
  并按需重启 deploy job。
- 新 Demo 上线要重启服务：Gallery 只在启动时扫描 `demos/`。

## 部署 subagent：必须用你自己的 Claude Code 与网关账号

subagent 会在后台起 Claude Code（联网、消耗网关额度），部署者必须在**自己的**入口脚本里设三个环境变量，
缺任何一个，subagent 的 API 全部返回 503（其余 Demo 不受影响）：

| 变量 | 含义 |
|---|---|
| `SUBAGENT_PASSWORD` | 页面进入 subagent 时要输的访问密码（公网端口无鉴权，必须设） |
| `SUBAGENT_CLAUDE_BIN` | **你自己的** `claude` 可执行文件路径 |
| `SUBAGENT_CLAUDE_SETTINGS` | **你自己的** Claude Code settings 文件；只取其中 `env` 段（网关地址、API key 等）传给后台 |

**[MUST NOT] 不要把这两个路径指向别人的目录**——哪怕读得到。那等于用别人的 API key 替你付费。
不想开放 subagent 就三个都不设，它会保持禁用。

## 删除 Demo

删掉 `demos/<id>/` 整个文件夹，重启服务即可。
