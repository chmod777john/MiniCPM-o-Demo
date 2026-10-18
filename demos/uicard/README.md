# UI Card Demo（`/demos/uicard`）

ui_card 训练口径的双工 Agent：模型调用天气 / 地图 / 搜索工具，工具**真实执行**，模型用 `show_card` 在对话中
展示卡片。移植自 `cjp/o5-fc-dev` 分支的 `jinpeng_agent_v2` 页面（`demos/agent_tools/` + `static/agent-tools*/`），
按本仓库的 Demo 组织方式重写，并按训练数据修正了工具结果格式。

## 页面

- **主面板左边：卡片区**。大图显示最新一张卡片，下面一排是本会话展示过的卡片，点一下回看。
- **主面板右边：消息区**（默认展开，「收起 ▸」折叠成窄条，状态记在浏览器里）。模型说的话 / think / 工具调用
  （参数逐 token 出现，返回结果默认折叠，带耗时与产出的卡片编号）。与 shell 下方的 AI Speech、
  Think/Tool-Call 面板内容有重叠，两边都能各自折叠。
- **消息区顶部「工具设置」**：`serp_google_search` 的实际后端（SERP.HK / Tavily 转 SERP 格式）、SERP 额外延迟、
  定位 / 天气的模拟延迟；改了下一次工具调用生效，不用重连。

## preset（每个对齐一份训练数据）

| preset | 训练数据（`/user/sunweiyue/dataset/o5_sdk_training_data/ui_card/…`） | 工具 | 对齐的端点 |
|---|---|---|---|
| `uicard_1016_r002` | `agent_duplex_uicard_1016_min2k_max8k_sysalign_train_recipe_002_v3_swy` | 6 个（含 `tavily_search`） | `f5t16-4000step` / `f5t17-4000step`（F5-T16 / T17 data-mix） |
| `uicard_1004_r011` | `agent_duplex_uicard_1004_min2k_max8k_sysalign_train_recipe_011_v3` | 5 个 | `merge1006-128k-300step`（0930 合版，agent 部分 = `family008_t005_f5t8allmix`） |

system = 统一前缀 #2（中文 no-think + 填充句，与三个 data-mix 的 `system_prefix_text` 逐字相同）+ 该份数据里
一条真实样本的 `[参考音频, per-case 正文]` 与 tools（原样，工具顺序也是训练口径）。选样本的条件：`show_card`
收 `card_id`、1016 批带 `tavily_search`、正文里有会话时间。**建会话前把正文里的会话时间刷新成当前时间**
（`web/session-time.js`，经 shell 的 `prepareSystem` 钩子；训练数据每条样本的会话时间本来就各不相同）。

## 工具结果格式（训练数据是唯一事实源）

`tools/executor.py` 的返回与训练数据**逐字段、同字段顺序**，由 `tests/test_uicard_tools.py` 对着
`tests/fixtures/training_tool_results_1016.json`（训练数据里原样抽出的每类工具真实返回）钉住：

- 天气 / 地图卡片由工具拼好，模型只拿编号（`cards: [{card_id:"w1",label:"今天"}]`、
  `[{title, markerCount, card_id:"m1"}]`）；编号整个会话递增（页面维护，`card_index_base`），完整卡片只给页面。
- 路线：`routes[].path` 抽稀到 6 个点给模型（训练口径），页面画图用最多 180 点；公交最多 5 条方案。
- `get_client_location`：固定位置（北京中关村）带 `city`。
- `tavily_search`：`{query, results:[{url,title,content,score(,published_date)}]}`，`content` 截 500 字。
- `serp_google_search`：经 llm-center MCP 网关的 SERP.HK 原始响应。
- `show_card`：`卡片已展示给用户。`；`card_id` 查不到时页面不画、改回错误结果给模型。

## 部署：key 走 Demo 服务的环境变量

Demo 服务 env 里设 `UICARD_TOOLS_CONFIG=<配置文件绝对路径>`（文件放仓库外、600 权限，改了不用重启）：

```json
{
  "serp_google_search": {"endpoint": "https://llm-center.modelbest.co/llm/mcp?mcp=serp-search", "api_key": "…"},
  "tavily": {"endpoint": "https://api.tavily.com/search", "api_key": "…", "search_depth": "basic"},
  "amap_weather": {"api_key": "<高德 Web 服务 key>"},
  "map_search": {"api_key": "<高德 Web 服务 key>"},
  "get_client_location": {"latitude": 39.9593, "longitude": 116.2979, "accuracy": 28, "city": "北京市"},
  "map_display": {"amap_js_api_key": "<高德 JS key>", "amap_js_security_code": "…"}
}
```

**不设就不调用外部接口**：需要 key 的工具如实给模型回 `{"status":"error","message":"… 未配置 api_key …"}`，
页面上方标出未配置的工具；不回退到任何人的默认 key。`map_display` 可省，省了地图卡用 OpenStreetMap 底图。

## 测试

```bash
python -m pytest -q demos/uicard tests/test_gallery_scene_hooks.py -p no:cacheprovider
```

外部接口在测试里一律用 httpx MockTransport / monkeypatch 替身，不碰真实网络与 key。
