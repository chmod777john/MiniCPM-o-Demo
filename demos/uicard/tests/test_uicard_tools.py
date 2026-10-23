"""uicard 工具结果必须与 ui_card 训练数据同形：同样的字段、同样的字段顺序。

夹具 ``fixtures/training_tool_results_1016.json`` 是从训练数据里原样抽出的每类工具真实返回。
外部接口一律用 httpx MockTransport / monkeypatch 替身，测试不碰真实网络与 key。
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from demos.uicard.tools import config as config_module
from demos.uicard.tools import executor
from demos.uicard.tools.executor import execute_tool

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "training_tool_results_1016.json").read_text("utf-8"))
TRAINING = FIXTURE["results"]


def signature(value: Any) -> Any:
    """有序的字段结构：dict 保留键顺序，list 取首元素，叶子只记类型。"""

    if isinstance(value, dict):
        return [(key, signature(item)) for key, item in value.items()]
    if isinstance(value, list):
        return ["list", signature(value[0])] if value else ["list"]
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    return type(value).__name__


def training(key: str) -> Any:
    return json.loads(TRAINING[key]["content"])


@pytest.fixture(autouse=True)
def fake_config(monkeypatch, tmp_path):
    path = tmp_path / "uicard_tools.json"
    path.write_text(json.dumps({
        "serp_google_search": {"api_key": "test-serp"},
        "tavily": {"api_key": "test-tavily", "endpoint": "https://tavily.test/search"},
        "amap_weather": {"api_key": "test-amap", "endpoint": "https://amap.test/weather"},
        "map_search": {"api_key": "test-amap", "base_url": "https://amap.test"},
    }), encoding="utf-8")
    monkeypatch.setenv(config_module.CONFIG_ENV, str(path))


def install_transport(monkeypatch, handler):
    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real(*args, **kwargs)

    monkeypatch.setattr(executor.httpx, "AsyncClient", factory)


def run(name, arguments, options=None):
    return asyncio.run(execute_tool(name, arguments, options or {"card_index_base": 1}))


AMAP_LIVE = {
    "status": "1", "count": "1", "info": "OK", "infocode": "10000",
    "lives": [{
        "province": "广东", "city": "深圳市", "adcode": "440300", "weather": "晴", "temperature": "27",
        "winddirection": "南", "windpower": "≤3", "humidity": "70", "reporttime": "2026-10-18 22:30:43",
        "temperature_float": "27.0", "humidity_float": "70.0",
    }],
}
AMAP_FORECAST = {
    "status": "1", "count": "1", "info": "OK", "infocode": "10000",
    "forecasts": [{
        "city": "上海市", "adcode": "310000", "province": "上海", "reporttime": "2026-10-18 22:33:01",
        "casts": [
            {"date": f"2026-10-{18 + i}", "week": str(i + 1), "dayweather": "小雨", "nightweather": "阴",
             "daytemp": "24", "nighttemp": "18", "daywind": "北", "nightwind": "北", "daypower": "1-3",
             "nightpower": "1-3", "daytemp_float": "24.0", "nighttemp_float": "18.0"}
            for i in range(4)
        ],
    }],
}
POIS = {"status": "1", "info": "OK", "pois": [
    {"id": f"B{i}", "name": f"咖啡{i}", "location": f"116.30{i},39.96{i}", "address": f"地址{i}", "type": "x"}
    for i in range(3)
]}
STEPS = [{"polyline": ";".join(f"116.{300 + j},39.{960 - j}" for j in range(20))}]
ROUTE = {"status": "1", "info": "OK", "route": {"paths": [{"distance": "11054", "duration": "1185", "steps": STEPS}]}}
TRANSIT = {"status": "1", "info": "OK", "route": {"transits": [
    {"distance": str(5000 + i), "duration": str(2000 + i),
     "segments": [{"walking": {"steps": STEPS}, "bus": {"buslines": [{"polyline": "116.31,39.95;116.32,39.94"}]}}]}
    for i in range(6)
]}}


def amap_handler(request: httpx.Request) -> httpx.Response:
    path = request.url.path
    if path.endswith("/weather"):
        body = AMAP_FORECAST if request.url.params.get("extensions") == "all" else AMAP_LIVE
    elif path.endswith("/v3/place/text") or path.endswith("/v3/place/around"):
        body = POIS
    elif path.endswith("/transit/integrated"):
        body = TRANSIT
    elif "/v3/direction/" in path:
        body = ROUTE
    else:
        body = {"status": "0", "info": "unexpected"}
    return httpx.Response(200, json=body)


def test_get_client_location_matches_training():
    result = run("get_client_location", {})
    assert result.error is None
    assert signature(json.loads(result.content)) == signature(training("get_client_location"))


@pytest.mark.parametrize("extensions,key,count", [("base", "amap_weather_base", 1), ("all", "amap_weather_all", 4)])
def test_weather_matches_training_and_numbers_cards(monkeypatch, extensions, key, count):
    install_transport(monkeypatch, amap_handler)
    result = run("amap_weather", {"city": "上海", "extensions": extensions}, {"card_index_base": 5})
    assert result.error is None
    payload = json.loads(result.content)
    assert signature(payload) == signature(training(key))
    assert [card["card_id"] for card in payload["cards"]] == [f"w{5 + i}" for i in range(count)]
    assert [entry.card_id for entry in result.cards] == [f"w{5 + i}" for i in range(count)]
    assert all(entry.card["type"] == "weather" for entry in result.cards)
    # 高德新增的 *_float 字段模型没见过，必须投影掉。
    assert "_float" not in result.content
    if extensions == "all":
        assert [card["label"] for card in payload["cards"]] == ["今天", "明天", "后天", "2026-10-21"]


@pytest.mark.parametrize("arguments,key", [
    ({"operation": "search", "query": "咖啡", "city": "北京", "limit": 3}, "map_search_search"),
    ({"operation": "nearby", "query": "咖啡", "latitude": 39.96, "longitude": 116.3, "radius": 3000, "limit": 3},
     "map_search_nearby"),
])
def test_places_match_training(monkeypatch, arguments, key):
    install_transport(monkeypatch, amap_handler)
    result = run("map_search", arguments, {"card_index_base": 7})
    assert result.error is None
    payload = json.loads(result.content)
    assert signature(payload) == signature(training(key))
    assert payload["cards"] == [{"title": payload["cards"][0]["title"], "markerCount": 3, "card_id": "m7"}]
    assert result.cards[0].card["type"] == "map"


@pytest.mark.parametrize("mode", ["driving", "walking", "cycling", "transit"])
def test_route_matches_training_with_six_point_path(monkeypatch, mode):
    install_transport(monkeypatch, amap_handler)
    arguments = {
        "operation": "route", "origin": "中关村", "origin_latitude": 39.9593, "origin_longitude": 116.2979,
        "destination": "北京西站", "destination_latitude": 39.8946, "destination_longitude": 116.3218,
        "mode": mode, "city": "北京",
    }
    result = run("map_search", arguments)
    assert result.error is None
    payload = json.loads(result.content)
    assert signature(payload) == signature(training(f"map_search_route_{mode}"))
    assert all(len(route["path"]) == 6 for route in payload["routes"])
    assert len(payload["routes"]) == (5 if mode == "transit" else 1)
    # 页面画图用的路线点更密，且不进模型的结果。
    assert len(result.cards[0].card["routes"][0]["path"]) > 6


def test_tavily_search_matches_training_and_truncates(monkeypatch):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(json.loads(request.content))
        assert request.headers["Authorization"] == "Bearer test-tavily"
        return httpx.Response(200, json={
            "query": "q", "answer": None, "images": [], "follow_up_questions": None,
            "results": [
                {"url": "https://a", "title": "A", "content": "长" * 900, "score": 0.5, "raw_content": None},
                {"url": "https://b", "title": "B", "content": "短", "score": 0.1, "raw_content": None},
            ],
        })

    install_transport(monkeypatch, handler)
    arguments = {"query": "北京 故宫", "max_results": 2, "include_domains": ["dpm.org.cn"]}
    result = run("tavily_search", arguments)
    assert result.error is None
    payload = json.loads(result.content)
    assert signature(payload) == signature(training("tavily_search_general"))
    assert len(payload["results"][0]["content"]) == 500
    assert seen["include_domains"] == ["dpm.org.cn"] and seen["max_results"] == 2
    assert seen["search_depth"] == "advanced"
    assert seen["chunks_per_source"] == 1
    assert seen["country"] == "china"
    assert seen["include_raw_content"] is False
    assert seen["topic"] == "general"
    assert "raw_content" not in payload["results"][0]


def test_tavily_news_keeps_published_date(monkeypatch):
    install_transport(monkeypatch, lambda request: httpx.Response(200, json={"results": [
        {"url": "https://a", "title": "A", "content": "x", "score": 0.5, "published_date": "Mon, 13 Oct 2026"},
    ]}))
    payload = json.loads(run("tavily_search", {"query": "q", "topic": "news"}).content)
    assert signature(payload) == signature(training("tavily_search_news"))


def test_tavily_total_cap_shares_unused_budget(monkeypatch):
    """总量 3000 字按剩余条数平摊；前面短的，后面能多用。无网址的条目丢掉。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "query": "q",
            "results": [
                {"url": "https://a", "title": "A", "content": "短", "score": 0.2},
                {"title": "no-url", "content": "丢", "score": 0.1},
                *[{"url": f"https://{i}", "title": str(i), "content": "长" * 900, "score": 0.5} for i in range(9)],
            ],
        })

    install_transport(monkeypatch, handler)
    payload = json.loads(run("tavily_search", {"query": "english query", "max_results": 10}).content)
    contents = [item["content"] for item in payload["results"]]
    assert len(contents) == 10
    assert sum(len(text) for text in contents) <= 3000
    assert len(contents[0]) == 1
    assert len(contents[1]) > 1


def test_tavily_rejects_unknown_topic():
    result = run("tavily_search", {"query": "q", "topic": "finance"})
    assert result.error and "topic" in result.error


def test_serp_passes_vendor_body_through(monkeypatch):
    body = TRAINING["serp_google_search"]["content"]
    monkeypatch.setattr(executor, "search_serp_hk", lambda *args, **kwargs: body)
    result = run("serp_google_search", {"query": "x"})
    assert result.error is None and result.content == body and result.provider == "serp"


def test_show_card_by_id_and_by_object():
    by_id = run("show_card", {"card_id": "m2"})
    assert (by_id.content, by_id.card_id, by_id.card) == (TRAINING["show_card"]["content"], "m2", None)
    by_card = run("show_card", {"card": {"type": "hint", "title": "t", "body": "b"}})
    assert by_card.card == {"type": "hint", "title": "t", "body": "b"} and by_card.card_id is None
    bad = run("show_card", {})
    assert bad.error and json.loads(bad.content)["status"] == "error"


def test_missing_key_is_reported_not_defaulted(monkeypatch):
    monkeypatch.delenv(config_module.CONFIG_ENV)
    result = run("amap_weather", {"city": "北京"})
    assert result.error and "UICARD_TOOLS_CONFIG" in result.error
    assert json.loads(result.content) == {"status": "error", "message": result.error}
    # 不需要 key 的工具照常可用。
    assert run("get_client_location", {}).error is None
