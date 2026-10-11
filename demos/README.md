# Demo Gallery 开发规范

> **一句话**：加一个 Demo = 复制 `demos/_template/` 成 `demos/<id>/`，改三处名字。
> **不需要改任何共享文件**（`gateway.py`、`demo_gallery/`、`static/demo-gallery/`）——
> 这一条不是约定，是测试：`tests/test_gallery_pluggability.py` 会按本文的步骤逐字复制一遍脚手架并挂载验证。

上线后的入口：

| 页面 | URL |
|---|---|
| Gallery 索引页（自动列出所有 Demo） | `/demos` |
| 某个 Demo | `/demos/<id>` |
| 该 Demo 的 API（含工具接口） | `/api/demos/<id>/**` |
| 该 Demo 的前端资源 | `/demos/<id>/web/**` |

---

## 1. 新建一个 Demo（三步）

```bash
cp -r demos/_template demos/my_demo
cd demos/my_demo
sed -i 's/_template/my_demo/g' __init__.py demo.py web/index.js web/_template.css
mv web/_template.css web/my_demo.css
```

改完的三处：

1. **`demo.py`** 的 `demo_id` / `title` / `summary` / `order`；
2. **`__init__.py`** 的 import 路径（`demos.my_demo.demo`）；
3. **`web/<id>.css`** 的文件名，以及 CSS 里的 `[data-demo="<id>"]` 命名空间。

重启 Gateway，`/demos` 上就会出现它。

> ⚠️ **id 的硬约束**：目录名 = `@demo(demo_id=...)` = 前端 `registerDemo({ id })`，三者必须完全一致
> （它同时是 URL 段；不一致在注册时直接抛错，并有测试盯着）。
> 以 `_` 开头的目录不会被发现（`_template` 因此不出现在索引页）。

## 2. 一个 Demo 的文件夹长什么样

```text
demos/<id>/
├── __init__.py        # 只 import demo.py —— import 本包即完成注册
├── demo.py            # 与 Gallery 的唯一接口：@demo(...) + 自己的 APIRouter
├── defaults.py        # 可选：defaults() 的实现（board 从训练 case 抽 system/tools）
├── tools/             # 可选：工具的 python 实现（board 的 display_object_on_board）
├── tests/             # 本 Demo 自己的测试（pytest 会扫 demos/）
└── web/
    ├── index.js       # 前端入口：registerDemo({ id, createScene })
    ├── scene.js       # 场景：右上主面板的全部内容与交互
    ├── quick-settings.js  # 可选：本 Demo 的 preset
    └── <id>.css       # 本 Demo 专属样式
```

现有 Demo 就是活范例：`demos/board/`（带工具 + 额外静态目录）、`demos/rejection/`（纯 think 解析）、`demos/standalone/`（输入框注入 standalone_event + Unit 回执，不带工具）。

## 3. 后端：`demo.py`

```python
router = APIRouter()   # 由 Gallery 挂到 /api/demos/<id>/**

@demo(
    demo_id="my_demo",            # = 目录名
    title="MiniCPM-o · My Demo",
    summary="索引页卡片上的一句话",
    order=50,                     # 索引页排序，小的在前
    styles=("my_demo.css",),      # 相对 web/
    router=router,                # 不需要后端接口就去掉
    # extra_mounts=(StaticMount(path="tools/xxx/downloads", create=True),),
)
class MyDemo:
    def defaults(self) -> dict:   # 可选
        return {"system": ..., "tools": ...}
```

| `@demo` 参数 | 说明 |
|---|---|
| `demo_id` | 必填，= 目录名 |
| `title` / `summary` / `order` | 索引页与页面标题 |
| `entry` | 前端入口，默认 `web/index.js` |
| `styles` | 本 Demo 的 CSS，相对 `web/`，注入在共享 `shell.css` 之后；文件不存在注册即报错 |
| `router` | 本 Demo 的 API。**工具接口只能放这里**，不允许挂到 Gateway 主路由（有测试盯着） |
| `extra_mounts` | 额外静态目录（相对 Demo 目录），URL = `/demos/<id>/<path>`；`create=True` 时启动时 mkdir |

**`GET /api/demos/<id>/defaults`** 由 Gallery 自动提供：先放通用运行时字段
（`model_identity`、`checkpoint_profile_id`、`default_quick_setting`、`tts_pad_protocol`、
non-spoken 调度与 budget、`unit_sec` 等，均来自部署环境变量），再用 `defaults()` 的返回值**覆盖合并**。
Demo 不用自己拼这些通用字段。

**一个 Demo import 失败不会拖垮 Gateway**：只记 `[gallery] demo xxx 导入失败，已跳过` 的日志，其余照常。
所以新 Demo 没出现在索引页时，**先看 Gateway 日志**。

运维开关（逗号分隔）：`GALLERY_DEMOS=board,rejection`（白名单）、`GALLERY_DEMOS_DISABLED=xxx`（黑名单）。

## 4. 前端：场景契约

shell（`static/demo-gallery/shell/app.js`）负责一切通用的东西：会话握手、麦克风、播放、
system 段编辑、quick setting 下拉、think/tool-call 账本、模型信息栏。
**场景只负责右上主面板 `stageEl`**。完整契约在 `static/demo-gallery/core/scene-contract.js` 顶部注释。

```js
// web/index.js
import { registerDemo } from '../../../static/demo-gallery/core/registry.js';
import { createScene } from './scene.js';
registerDemo({ id: 'my_demo', createScene });
```

`createScene(context)` 收到：

| 字段 | 用途 |
|---|---|
| `stageEl` | 主面板根节点。shell 永远不写它的 innerHTML，**内容与布局全归场景** |
| `api.url(p)` | 拼本 Demo 的 API 地址：`api.url('tools/xxx')` → `/api/demos/<id>/tools/xxx`。**不要写死绝对路径** |
| `assets.url(p)` | 拼本 Demo 的静态资源地址 |
| `getClient()` | 当前 Realtime client（未连线时 `null`），工具结果用 `client.sendToolResult({ toolCallId, content })` 回灌 |
| `shell.setStatus` | 可选，回写底部状态栏 |

返回的 scene 对象：

| 字段 | 必填 | 说明 |
|---|---|---|
| `render()` | ✅ | 首帧由场景自己画（**包括空状态**），页面骨架里没有任何场景内容 |
| `panelTitle` / `panelHint` / `pageTitle` / `liveStatus` | | 面板标题、提示、页面标题、开麦后的状态文案 |
| `presetOptions` / `buildQuickSettings(defaults)` / `fallbackQuickSetting` | | quick setting 下拉。**没有 preset 是合法的**，shell 会跳过（脚手架就是这样） |
| `tools(defaults, presetId)` | | session 声明的工具列表；`presetId` 是当前选中的 preset（选 Custom 时为 `fallbackQuickSetting`）。工具顺序属于训练口径时按 preset 返回各自的工具 |
| `reset()` | | 新会话开始时清场；缺省等于 `render()` |
| `onThinkBegin/Delta/End`、`onToolCallDone(event)` | | 模型事件钩子 |
| `onUnitStarted(event)` | | 每个被处理 Unit 的 `response.unit.started`，含 `tool_events` / `standalone_events` 回执，用来展示事件落在哪个 Unit |

**工具调用的链路**：session 用 `auto_execute_tools=false`，模型只发 tool_call →
场景在 `onToolCallDone` 里 `fetch(api.url('tools/<name>'))` 调本 Demo 的后端 → 拿到结果后
`getClient().sendToolResult(...)` 经 WS 回灌给模型。范例见 `demos/board/web/scene.js`。

不关联工具调用的外部事件（如后台子任务的进度汇报）用 `getClient().sendStandaloneEvent({ content, eventId })`，
协议见 `docs/fc-duplex/semantic-realtime-api-v2.md` §8.1。

**import 共享模块一律用相对路径** `../../../static/demo-gallery/core/xxx.js`。
Demo 资源的 URL 与磁盘路径同构（`/demos/<id>/web/` ↔ `demos/<id>/web/`），所以同一条相对路径
在浏览器和 node 测试里解析到同一个文件 —— **不要改挂载前缀**。

可复用的共享模块（`static/demo-gallery/core/`）：`quick-settings.js`（preset 拼装）、
`system-prefixes.js`（统一系统提示词前缀）、`ui-utils.js`（`escapeHtml` 等）、
`realtime-client.js`、`ns-segment-view.js` 等。

## 5. CSS 规则

- 每条顶层选择器（含 `@media` 内的）**必须以 `[data-demo="<id>"]` 开头**；`@keyframes` 名以 `<id>-` 开头。
  shell 会给 `<body>` 打上 `data-demo="<id>"`。不加命名空间会污染 shell 和别的 Demo —— 有测试逐条检查。
- 不要改 `stageEl` 的 class 来「解除」shell 的布局；布局写在自己的 CSS 里。
- 通用的东西（麦克风提示、模型信息栏……）属于 `shell.css`，不要复制进 Demo。

## 6. 测试

```bash
python -m pytest -q tests demos -p no:cacheprovider
```

- 新 Demo 自动被这些守卫覆盖，无需登记：id 三处一致、scene 满足契约、CSS 命名空间、
  API 不逃逸到主路由、共享层不出现任何 Demo 名字。
- Demo 自己的逻辑测试放 `demos/<id>/tests/`。前端纯逻辑抽成独立模块（如 `board-state.js`、
  `think-parser.js`），用 `tests/conftest.py` 的 `run_node` 直接 import 测，**不要用正则抠源码**。
- 改了 CSS/布局，用浏览器实际打开 `/demos/<id>` 看一眼（零 JS 报错、零 404、布局正常）。

## 7. 删除一个 Demo

删掉 `demos/<id>/` 整个文件夹即可。
