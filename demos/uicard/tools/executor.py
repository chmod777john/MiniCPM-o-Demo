"""uicard 的六个工具：真实调用外部接口，返回与 ui_card 训练数据**逐字段、同字段顺序**的工具结果。

训练数据（``ui_card/agent_duplex_uicard_1016_*_recipe_002_v3_swy`` 与 ``1004_*_recipe_011_v3``）是格式的
唯一事实源；工具结果的形状一旦和训练不同，下一次 prefill 就落在分布外。两批的工具结果格式相同，
1016 批多一个 ``tavily_search`` 工具。各工具的训练格式：

* ``get_client_location`` → ``{"status":"ok","message":"","timestamp":毫秒,"latitude","longitude","accuracy","city"}``
* ``amap_weather`` → 高德原始响应按训练字段投影，末尾追加 ``"cards":[{"card_id":"w1","label":"今天"},…]``
* ``map_search`` search/nearby → ``{"provider","operation","query","places":[…],"cards":[{"title","markerCount","card_id"}]}``
* ``map_search`` route → ``{"provider","operation","origin","destination","routes":[{"mode","distanceMeters",
  "durationSeconds","path":[6 个点]}],"cards":[…]}``
* ``serp_google_search`` → SERP.HK 原始响应（经 llm-center MCP 网关）
* ``tavily_search`` → ``{"query","results":[{"url","title","content","score"(,"published_date")}]}``，
  ``content`` 截到 500 字（训练数据最常见的截断长度）
* ``show_card`` → ``卡片已展示给用户。``

天气与地图卡片由工具在这里拼好，模型只拿到编号（``cards``）；完整卡片只给页面（:class:`ToolExecution`
的 ``cards``），模型永远看不到。编号由页面按会话递增分配（``card_index_base``），服务端无状态。

移植自 cjp/o5-fc-dev 分支 ``demos/agent_tools/``（executor.py / v2.py），按训练数据修正了：
路线保留 6 点 ``path``、定位带 ``city``、地图 ``cards`` 字段顺序、独立的 ``tavily_search`` 工具。
"""

from __future__ import annotations

import asyncio
import json
import math
import random
import time
from typing import Any
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, Field

from demos.uicard.tools.config import tool_config
from demos.uicard.tools.coords import gcj02_to_wgs84, haversine_meters, wgs84_to_gcj02
from demos.uicard.tools.serp_mcp import search_serp_hk

SHOW_CARD_RESULT = "卡片已展示给用户。"
WEATHER_CARD_PREFIX = "w"
MAP_CARD_PREFIX = "m"
# 训练数据里每条路线的 path 一律抽稀到 6 个点（含起终点）；公交最多给 5 条方案，其余方式 1 条。
ROUTE_PATH_POINTS = 6
MAX_ROUTES = 5
# 页面画图用的完整路线（不给模型）。
DISPLAY_PATH_POINTS = 180
TAVILY_CONTENT_CHARS = 500
TAVILY_DEFAULT_MAX_RESULTS = 5
SERP_MAX_ATTEMPTS = 4
SERP_RETRY_DELAY_SEC = 0.25
MAX_RESPONSE_DELAY_SEC = 300.0

TOOL_NAMES = (
    "amap_weather",
    "map_search",
    "serp_google_search",
    "tavily_search",
    "get_client_location",
    "show_card",
)

_AMAP_ROUTE_PATHS = {
    "driving": "driving",
    "walking": "walking",
    "transit": "transit/integrated",
    "cycling": "bicycling",
}


class ToolError(Exception):
    """工具拿不到训练形态的结果；消息会如实回给模型。"""


class CardEntry(BaseModel):
    """工具拼好的一张卡片：编号给模型，卡片本体只给页面。"""

    card_id: str
    card: dict[str, Any]


class ToolExecution(BaseModel):
    """一次工具执行的结果。

    ``content`` 原样回灌给模型；其余字段只给页面。
    """

    name: str
    content: str
    cards: list[CardEntry] = Field(default_factory=list)
    card: dict[str, Any] | None = None       # show_card 传入的自写卡片
    card_id: str | None = None               # show_card 引用的编号（由页面查注册表）
    error: str | None = None
    elapsed_ms: int = 0
    provider: str | None = None              # serp_google_search 实际用的搜索后端


def compact(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _api_key(section: str) -> str:
    key = _text(tool_config(section).get("api_key"))
    if not key:
        raise ToolError(f"{section} 未配置 api_key（Demo 服务环境变量 UICARD_TOOLS_CONFIG 指向的配置文件）")
    return key


def _card_index_base(options: dict[str, Any]) -> int:
    value = _as_int(options.get("card_index_base"), 1)
    return value if value >= 1 else 1


async def execute_tool(
    name: str,
    arguments: dict[str, Any],
    options: dict[str, Any] | None = None,
) -> ToolExecution:
    """执行一次工具调用。

    参数:
        name: 工具名（``response.tool_call.done`` 里的 ``call.name``）。
        arguments: 解析后的工具参数。
        options: 页面侧开关：``card_index_base``（本次卡片编号起点）、``search_provider``
            （``serp`` / ``tavily``，只换 serp_google_search 的实际后端，模型看到的格式不变）、
            ``serp_response_delay_sec``、``<工具>_delay_enabled/min_ms/max_ms``。

    返回:
        :class:`ToolExecution`；失败时 ``content`` 是 ``{"status":"error","message":…}``，同样可以回灌。
    """

    started = time.monotonic()
    options = options or {}
    result = ToolExecution(name=name, content="")
    try:
        if name == "get_client_location":
            result.content = _get_client_location()
            await _sleep_local_delay("get_client_location", options)
        elif name == "amap_weather":
            result.content, result.cards = await _amap_weather(arguments, _card_index_base(options))
            await _sleep_local_delay("amap_weather", options)
        elif name == "map_search":
            result.content, result.cards = await _map_search(arguments, _card_index_base(options))
        elif name == "serp_google_search":
            result.provider = _text(options.get("search_provider")).lower() or "serp"
            result.content = await _serp_google_search(arguments, result.provider)
            await _sleep_seconds(options.get("serp_response_delay_sec"))
        elif name == "tavily_search":
            result.content = await _tavily_search(arguments)
        elif name == "show_card":
            result.content, result.card, result.card_id = _show_card(arguments)
        else:
            raise ToolError(f"unknown tool: {name}")
    except Exception as exc:  # noqa: BLE001 - 如实回给模型与页面
        error = str(exc) if isinstance(exc, ToolError) else f"{type(exc).__name__}: {exc}"
        result = ToolExecution(
            name=name,
            content=compact({"status": "error", "message": error}),
            error=error,
            provider=result.provider,
        )
    result.elapsed_ms = int((time.monotonic() - started) * 1000)
    return result


# ------------------------------------------------------------------ 延迟（测试用开关）


async def _sleep_seconds(value: Any) -> None:
    seconds = _as_float(value)
    if seconds is None or not math.isfinite(seconds) or seconds <= 0:
        return
    await asyncio.sleep(min(seconds, MAX_RESPONSE_DELAY_SEC))


async def _sleep_local_delay(name: str, options: dict[str, Any]) -> None:
    """页面打开了本地工具延迟时，在区间内随机等待（模拟真实网络耗时）。"""

    if not options.get(f"{name}_delay_enabled"):
        return
    low = max(0.0, _as_float(options.get(f"{name}_delay_min_ms")) or 0.0)
    high = max(low, _as_float(options.get(f"{name}_delay_max_ms")) or low)
    if high > 0:
        await asyncio.sleep(min(random.uniform(low, high) / 1000.0, MAX_RESPONSE_DELAY_SEC))


# ---------------------------------------------------------------- get_client_location


def _get_client_location() -> str:
    settings = tool_config("get_client_location")
    return compact(
        {
            "status": "ok",
            "message": "",
            "timestamp": int(time.time() * 1000),
            "latitude": _as_float(settings.get("latitude")),
            "longitude": _as_float(settings.get("longitude")),
            "accuracy": _as_int(settings.get("accuracy"), 28),
            "city": _text(settings.get("city")),
        }
    )


# ------------------------------------------------------------------------- show_card


def _show_card(arguments: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str | None]:
    """show_card 只认两种入参：天气 / 地图的 ``card_id``，或模型自写的 ``card`` 对象。

    编号在页面的注册表里，这里只原样回传给页面去查；查不到时由页面改回错误结果。
    """

    card_id = _text(arguments.get("card_id"))
    if card_id:
        return SHOW_CARD_RESULT, None, card_id
    card = arguments.get("card")
    if isinstance(card, str):
        try:
            card = json.loads(card)
        except ValueError as exc:
            raise ToolError(f"card is not valid JSON: {exc}") from exc
    if not isinstance(card, dict) or not _text(card.get("type")):
        raise ToolError("show_card requires card_id, or a card object with a type field")
    return SHOW_CARD_RESULT, card, None


# ---------------------------------------------------------------------- amap_weather

# 训练数据冻结的字段集与顺序（高德后来新增的 temperature_float 等模型没见过，投影掉）。
_WEATHER_LIVE_FIELDS = (
    "province", "city", "adcode", "weather", "temperature",
    "winddirection", "windpower", "humidity", "reporttime",
)
_WEATHER_FORECAST_FIELDS = ("city", "adcode", "province", "reporttime")
_WEATHER_CAST_FIELDS = (
    "date", "week", "dayweather", "nightweather", "daytemp",
    "nighttemp", "daywind", "nightwind", "daypower", "nightpower",
)
# 高德中文天气 → 卡片 condition 枚举；先匹配先赢（「雷阵雨」→ thunder，「阵雨」→ showers）。
_CONDITIONS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("雷",), "thunder"),
    (("阵雨",), "showers"),
    (("雨",), "rain"),
    (("雪",), "snow"),
    (("雾", "霾"), "fog"),
    (("风", "沙", "尘"), "wind"),
    (("阴",), "overcast"),
    (("多云",), "cloudy"),
    (("晴",), "sunny"),
)
_FORECAST_LABELS = ("今天", "明天", "后天")


def _project(source: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: source[field] for field in fields if field in source}


def condition_of(description: Any) -> str:
    text = str(description or "")
    for keys, condition in _CONDITIONS:
        if any(key in text for key in keys):
            return condition
    return "cloudy"


def _wind(direction: Any, power: Any) -> str:
    direction = str(direction or "").strip()
    power = str(power or "").strip()
    if not direction and not power:
        return ""
    if direction and not direction.endswith("风"):
        direction += "风"
    return f"{direction}{power}级" if power else direction


def _int_or_none(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def project_amap_weather(body: dict[str, Any]) -> dict[str, Any] | None:
    """把高德成功响应投影回训练字段集；不是成功响应时返回 None。"""

    if str(body.get("status")) != "1":
        return None
    result = _project(body, ("status", "count", "info", "infocode"))
    lives = body.get("lives")
    forecasts = body.get("forecasts")
    if isinstance(lives, list) and lives:
        result["lives"] = [_project(live, _WEATHER_LIVE_FIELDS) for live in lives if isinstance(live, dict)]
    elif isinstance(forecasts, list) and forecasts:
        projected = []
        for forecast in forecasts:
            if not isinstance(forecast, dict):
                continue
            entry = _project(forecast, _WEATHER_FORECAST_FIELDS)
            entry["casts"] = [
                _project(cast, _WEATHER_CAST_FIELDS)
                for cast in forecast.get("casts") or []
                if isinstance(cast, dict)
            ]
            projected.append(entry)
        result["forecasts"] = projected
    else:
        return None
    return result


def build_weather_cards(body: dict[str, Any], start_index: int) -> list[tuple[str, str, dict[str, Any]]]:
    """每天一张天气卡：``[(card_id, label, card), …]``；实况一张，预报按天。"""

    lives = body.get("lives")
    if isinstance(lives, list) and lives and isinstance(lives[0], dict):
        live = lives[0]
        card: dict[str, Any] = {
            "type": "weather",
            "city": live.get("city"),
            "condition": condition_of(live.get("weather")),
            "unit": "℃",
            "desc": live.get("weather"),
        }
        temp = _int_or_none(live.get("temperature"))
        if temp is not None:
            card["temp"] = temp
        humidity = str(live.get("humidity") or "").strip()
        if humidity:
            card["humidity"] = f"{humidity}%"
        wind = _wind(live.get("winddirection"), live.get("windpower"))
        if wind:
            card["wind"] = wind
        return [(f"{WEATHER_CARD_PREFIX}{start_index}", "今天", card)]

    forecasts = body.get("forecasts")
    if not (isinstance(forecasts, list) and forecasts and isinstance(forecasts[0], dict)):
        return []
    forecast = forecasts[0]
    cards: list[tuple[str, str, dict[str, Any]]] = []
    for index, cast in enumerate(forecast.get("casts") or []):
        if not isinstance(cast, dict):
            continue
        card = {
            "type": "weather",
            "city": forecast.get("city"),
            "condition": condition_of(cast.get("dayweather")),
            "unit": "℃",
            "desc": cast.get("dayweather"),
        }
        high = _int_or_none(cast.get("daytemp"))
        low = _int_or_none(cast.get("nighttemp"))
        if high is not None:
            card["temp"] = high
            card["high"] = high
        if low is not None:
            card["low"] = low
        wind = _wind(cast.get("daywind"), cast.get("daypower"))
        if wind:
            card["wind"] = wind
        label = _FORECAST_LABELS[index] if index < len(_FORECAST_LABELS) else str(cast.get("date") or "")
        cards.append((f"{WEATHER_CARD_PREFIX}{start_index + len(cards)}", label, card))
    return cards


def weather_payload(body: dict[str, Any], start_index: int) -> tuple[str, list[CardEntry]]:
    """高德响应 → (给模型的结果, 给页面的卡片)。高德自己的错误响应原样透传。"""

    projected = project_amap_weather(body)
    if projected is None:
        return compact(body), []
    cards = build_weather_cards(projected, start_index)
    if cards:
        projected["cards"] = [{"card_id": card_id, "label": label} for card_id, label, _ in cards]
    return compact(projected), [CardEntry(card_id=card_id, card=card) for card_id, _, card in cards]


async def _amap_weather(arguments: dict[str, Any], start_index: int) -> tuple[str, list[CardEntry]]:
    settings = tool_config("amap_weather")
    params = {
        "key": _api_key("amap_weather"),
        "city": _text(arguments.get("city")) or "北京",
        "extensions": _text(arguments.get("extensions")) or "base",
    }
    async with httpx.AsyncClient(timeout=float(settings.get("timeout_sec", 10.0))) as client:
        response = await client.get(str(settings.get("endpoint")), params=params)
    try:
        body = response.json()
    except ValueError:
        return response.text, []
    if not isinstance(body, dict):
        return response.text, []
    return weather_payload(body, start_index)


# ------------------------------------------------------------------------ map_search


def _map_card_entry(card_id: str, title: str, marker_count: int) -> dict[str, Any]:
    # 字段顺序按训练数据：title, markerCount, card_id。
    return {"title": title, "markerCount": marker_count, "card_id": card_id}


def _simplify_path(path: list[dict[str, float]], limit: int) -> list[dict[str, float]]:
    """均匀抽 ``limit`` 个点，保留起终点。"""

    if len(path) <= limit:
        return path
    step = (len(path) - 1) / float(limit - 1)
    return [path[round(index * step)] for index in range(limit)]


async def _map_search(arguments: dict[str, Any], start_index: int) -> tuple[str, list[CardEntry]]:
    settings = tool_config("map_search")
    key = _api_key("map_search")
    base_url = str(settings.get("base_url") or "https://restapi.amap.com").rstrip("/")
    operation = _text(arguments.get("operation")).lower()
    if operation not in {"search", "nearby", "route"}:
        raise ToolError("operation must be search, nearby, or route")
    async with httpx.AsyncClient(timeout=float(settings.get("timeout_sec", 10.0))) as client:
        if operation == "route":
            return await _map_route(client, base_url, key, arguments, start_index)
        return await _map_places(client, base_url, key, operation, arguments, start_index)


async def _map_places(
    client: httpx.AsyncClient,
    base_url: str,
    key: str,
    operation: str,
    arguments: dict[str, Any],
    start_index: int,
) -> tuple[str, list[CardEntry]]:
    query = _text(arguments.get("query"))
    if not query:
        raise ToolError("query is required for search/nearby")
    limit = _clamp(_as_int(arguments.get("limit"), 5), 1, 10)
    params: dict[str, Any] = {"key": key, "keywords": query, "offset": limit, "extensions": "base"}
    latitude = _as_float(arguments.get("latitude"))
    longitude = _as_float(arguments.get("longitude"))
    has_center = latitude is not None and longitude is not None
    if operation == "nearby":
        if not has_center:
            raise ToolError("latitude and longitude are required for nearby search")
        gcj_lat, gcj_lon = wgs84_to_gcj02(latitude, longitude)
        params["location"] = f"{gcj_lon:.6f},{gcj_lat:.6f}"
        params["radius"] = _clamp(_as_int(arguments.get("radius"), 5000), 100, 50000)
        params["sortrule"] = "distance"
        endpoint = f"{base_url}/v3/place/around"
    else:
        city = _text(arguments.get("city"))
        if city:
            params["city"] = city
        endpoint = f"{base_url}/v3/place/text"

    body = _amap_json(await client.get(endpoint, params=params))
    places: list[dict[str, Any]] = []
    for poi in body.get("pois") or []:
        point = _parse_lon_lat(poi.get("location"))
        if point is None:
            continue
        wgs_lat, wgs_lon = gcj02_to_wgs84(point[0], point[1])
        place: dict[str, Any] = {
            "id": str(poi.get("id") or ""),
            "name": str(poi.get("name") or ""),
            "latitude": wgs_lat,
            "longitude": wgs_lon,
        }
        address = poi.get("address")
        if isinstance(address, str) and address.strip():
            place["address"] = address
        if operation == "nearby":
            place["distanceMeters"] = round(haversine_meters(latitude, longitude, wgs_lat, wgs_lon))
        places.append(place)
        if len(places) >= limit:
            break
    if not places:
        raise ToolError(f"no places found for {query!r}")

    title = f"附近的{query}" if operation == "nearby" else query
    card_id = f"{MAP_CARD_PREFIX}{start_index}"
    payload = {
        "provider": "amap",
        "operation": operation,
        "query": query,
        "places": places,
        "cards": [_map_card_entry(card_id, title, len(places))],
    }
    card = {"type": "map", "title": title, "provider": "amap", "markers": places}
    return compact(payload), [CardEntry(card_id=card_id, card=card)]


async def _map_route(
    client: httpx.AsyncClient,
    base_url: str,
    key: str,
    arguments: dict[str, Any],
    start_index: int,
) -> tuple[str, list[CardEntry]]:
    mode = _text(arguments.get("mode")).lower() or "driving"
    amap_path = _AMAP_ROUTE_PATHS.get(mode)
    if amap_path is None:
        raise ToolError(f"unsupported route mode: {mode}")
    city = _text(arguments.get("city"))
    origin, origin_gcj = await _resolve_endpoint(client, base_url, key, arguments, "origin", city)
    destination, destination_gcj = await _resolve_endpoint(client, base_url, key, arguments, "destination", city)
    params: dict[str, Any] = {
        "key": key,
        "origin": f"{origin_gcj[1]:.6f},{origin_gcj[0]:.6f}",
        "destination": f"{destination_gcj[1]:.6f},{destination_gcj[0]:.6f}",
        "output": "json",
    }
    if mode == "transit":
        if not city:
            raise ToolError("city is required for transit routes")
        params["city"] = city
    body = _amap_json(await client.get(f"{base_url}/v3/direction/{amap_path}", params=params))
    full_routes = _parse_amap_routes(body, mode)
    if not full_routes:
        raise ToolError("AMap returned no route")

    model_routes = []
    display_routes = []
    for route in full_routes:
        full_path = route.pop("path", [])
        model_routes.append({**route, "path": _simplify_path(full_path, ROUTE_PATH_POINTS)})
        display_routes.append({**route, "path": _simplify_path(full_path, DISPLAY_PATH_POINTS)})

    title = f"{origin['name']} → {destination['name']}"
    card_id = f"{MAP_CARD_PREFIX}{start_index}"
    payload = {
        "provider": "amap",
        "operation": "route",
        "origin": origin,
        "destination": destination,
        "routes": model_routes,
        "cards": [_map_card_entry(card_id, title, 2)],
    }
    card = {
        "type": "map",
        "title": title,
        "provider": "amap",
        "markers": [origin, destination],
        "routes": display_routes,
    }
    return compact(payload), [CardEntry(card_id=card_id, card=card)]


async def _resolve_endpoint(
    client: httpx.AsyncClient,
    base_url: str,
    key: str,
    arguments: dict[str, Any],
    prefix: str,
    city: str,
) -> tuple[dict[str, Any], tuple[float, float]]:
    """起终点：优先用模型给的坐标，否则按名称地理编码。形状按训练数据：``{id, name, latitude, longitude}``。"""

    latitude = _as_float(arguments.get(f"{prefix}_latitude"))
    longitude = _as_float(arguments.get(f"{prefix}_longitude"))
    name = _text(arguments.get(prefix)) or ("起点" if prefix == "origin" else "终点")
    if latitude is not None and longitude is not None:
        return (
            {"id": prefix, "name": name, "latitude": latitude, "longitude": longitude},
            wgs84_to_gcj02(latitude, longitude),
        )
    if not _text(arguments.get(prefix)):
        raise ToolError(f"{prefix} name or coordinates are required")
    params: dict[str, Any] = {"key": key, "address": name}
    if city:
        params["city"] = city
    body = _amap_json(await client.get(f"{base_url}/v3/geocode/geo", params=params))
    for geocode in body.get("geocodes") or []:
        point = _parse_lon_lat(geocode.get("location"))
        if point is None:
            continue
        wgs_lat, wgs_lon = gcj02_to_wgs84(point[0], point[1])
        return {"id": prefix, "name": name, "latitude": wgs_lat, "longitude": wgs_lon}, point
    raise ToolError(f"place not found: {name}")


def _parse_amap_routes(body: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    route = body.get("route")
    if not isinstance(route, dict):
        return []
    routes: list[dict[str, Any]] = []
    for item in (route.get("paths") or route.get("transits") or [])[:MAX_ROUTES]:
        if not isinstance(item, dict):
            continue
        routes.append(
            {
                "mode": mode,
                "distanceMeters": _as_int(item.get("distance"), 0),
                "durationSeconds": _as_int(item.get("duration"), 0),
                "path": _collect_polyline(item),
            }
        )
    return routes


def _collect_polyline(item: dict[str, Any]) -> list[dict[str, float]]:
    """驾车 / 步行 / 骑行的 steps，或公交方案 segments 里步行与公交线路的 polyline。"""

    chunks: list[str] = []
    for step in item.get("steps") or []:
        if isinstance(step, dict):
            chunks.append(str(step.get("polyline") or ""))
    for segment in item.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        for step in (segment.get("walking") or {}).get("steps") or []:
            if isinstance(step, dict):
                chunks.append(str(step.get("polyline") or ""))
        for line in (segment.get("bus") or {}).get("buslines") or []:
            if isinstance(line, dict):
                chunks.append(str(line.get("polyline") or ""))
                break  # 同一段的备选线路只取第一条
    points: list[dict[str, float]] = []
    for chunk in chunks:
        for raw in chunk.split(";"):
            point = _parse_lon_lat(raw)
            if point is None:
                continue
            wgs_lat, wgs_lon = gcj02_to_wgs84(point[0], point[1])
            points.append({"latitude": wgs_lat, "longitude": wgs_lon})
    return points


def _amap_json(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError as exc:
        raise ToolError(f"AMap returned a non-JSON body: {response.text[:200]}") from exc
    if not isinstance(body, dict):
        raise ToolError("AMap returned an unexpected body")
    if str(body.get("status")) != "1":
        raise ToolError(f"AMap: {body.get('info') or 'request failed'}")
    return body


def _parse_lon_lat(value: Any) -> tuple[float, float] | None:
    """高德 ``"lon,lat"`` → ``(latitude, longitude)``。"""

    if not isinstance(value, str) or "," not in value:
        return None
    lon_text, _, lat_text = value.strip().partition(",")
    longitude = _as_float(lon_text)
    latitude = _as_float(lat_text)
    if latitude is None or longitude is None:
        return None
    return latitude, longitude


# --------------------------------------------------------------- serp_google_search

# 带这些模块之一才算真有结果；全空是供应商的软失败，值得重试。
_SERP_SUBSTANTIVE_MODULES = ("organic", "top_stories", "answer_box", "knowledge_graph", "people_also_ask")


async def _serp_google_search(arguments: dict[str, Any], provider: str) -> str:
    query = _text(arguments.get("query"))
    if not query:
        raise ToolError("serp_google_search requires query")
    if provider == "tavily":
        return await _tavily_as_serp(query, arguments)
    if provider not in {"serp", "serp_hk"}:
        raise ToolError(f"unknown search provider: {provider}")
    settings = tool_config("serp_google_search")
    key = _api_key("serp_google_search")
    timeout = float(settings.get("timeout_sec", 20.0))
    retry_budget = float(settings.get("retry_budget_sec", 9.0))
    endpoint = str(settings.get("endpoint"))
    started = time.monotonic()
    body = ""
    for attempt in range(1, SERP_MAX_ATTEMPTS + 1):
        try:
            body = await asyncio.to_thread(
                search_serp_hk, query, key, 5, timeout_seconds=timeout, url=endpoint
            )
        except Exception as exc:  # noqa: BLE001 - 如实回给模型
            body = compact({"code": -1, "msg": f"serp request failed: {exc}", "result": None})
        if attempt == SERP_MAX_ATTEMPTS or not _serp_is_retryable(body):
            break
        delay = SERP_RETRY_DELAY_SEC * attempt
        if time.monotonic() - started + delay >= retry_budget:
            break
        await asyncio.sleep(delay)
    return body


def _serp_is_retryable(body: str) -> bool:
    try:
        parsed = json.loads(body)
    except ValueError:
        return False
    if not isinstance(parsed, dict):
        return False
    code = parsed.get("code")
    if isinstance(code, int) and (code == 3102 or code >= 500):
        return True
    if code is not None and code != 0:
        return False
    result = parsed.get("result") if code is not None else parsed
    if not isinstance(result, dict):
        return True
    return not any(result.get(module) for module in _SERP_SUBSTANTIVE_MODULES)


async def _tavily_request(payload: dict[str, Any]) -> dict[str, Any]:
    settings = tool_config("tavily")
    key = _api_key("tavily")
    payload = {"search_depth": _text(settings.get("search_depth")) or "basic", **payload}
    async with httpx.AsyncClient(timeout=float(settings.get("timeout_sec", 15.0))) as client:
        response = await client.post(
            str(settings.get("endpoint")),
            json=payload,
            headers={"Authorization": f"Bearer {key}"},
        )
    try:
        parsed = response.json()
    except ValueError as exc:
        raise ToolError(f"tavily returned a non-JSON body: {response.text[:200]}") from exc
    if not isinstance(parsed, dict):
        raise ToolError("tavily returned an unexpected body")
    if response.status_code >= 400:
        detail = parsed.get("detail") or parsed.get("error") or response.text[:200]
        raise ToolError(f"tavily: {detail}")
    return parsed


async def _tavily_as_serp(query: str, arguments: dict[str, Any]) -> str:
    """serp_google_search 的备选后端：Tavily 结果投影成 SERP.HK 外形（cjp 原实现）。"""

    max_results = _clamp(_as_int(arguments.get("num"), 10), 1, 20)
    parsed = await _tavily_request({"query": query, "max_results": max_results, "include_answer": False})
    organic = []
    for rank, item in enumerate(parsed.get("results") or [], start=1):
        if not isinstance(item, dict):
            continue
        link = _text(item.get("url"))
        if not link:
            continue
        organic.append(
            {
                "display_link": _display_link(link),
                "link": link,
                "rank": rank,
                "snippet": _text(item.get("content")),
                "title": _text(item.get("title")),
            }
        )
    result = {
        "general": {"engine": "google", "page": 1, "query": query, "type": "search"},
        "top_stories": [],
        "organic": organic,
        "people_also_ask": [],
        "people_also_search_for": [],
    }
    return json.dumps({"code": 0, "msg": "success", "result": result}, ensure_ascii=False, indent=2)


def _display_link(url: str) -> str:
    parsed = urlsplit(url)
    host = f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme else parsed.netloc
    segments = [segment for segment in parsed.path.split("/") if segment][:2]
    return " › ".join([host, *segments]) if segments else host


# --------------------------------------------------------------------- tavily_search

_TAVILY_PASSTHROUGH = ("topic", "time_range", "include_domains", "exclude_domains")


def tavily_payload(query: str, body: dict[str, Any]) -> str:
    """Tavily 响应 → 训练形态 ``{"query","results":[{url,title,content,score(,published_date)}]}``。"""

    results = []
    for item in body.get("results") or []:
        if not isinstance(item, dict):
            continue
        entry: dict[str, Any] = {
            "url": item.get("url"),
            "title": item.get("title"),
            "content": str(item.get("content") or "")[:TAVILY_CONTENT_CHARS],
            "score": item.get("score"),
        }
        if item.get("published_date"):
            entry["published_date"] = item["published_date"]
        results.append(entry)
    return compact({"query": query, "results": results})


async def _tavily_search(arguments: dict[str, Any]) -> str:
    query = _text(arguments.get("query"))
    if not query:
        raise ToolError("tavily_search requires query")
    payload: dict[str, Any] = {
        "query": query,
        "max_results": _clamp(_as_int(arguments.get("max_results"), TAVILY_DEFAULT_MAX_RESULTS), 1, 10),
        "include_answer": False,
    }
    for key in _TAVILY_PASSTHROUGH:
        value = arguments.get(key)
        if value not in (None, "", []):
            payload[key] = value
    return tavily_payload(query, await _tavily_request(payload))
