"""通过 LLMCenter MCP 网关调用 SERP.HK Google 高级搜索。"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List


SERP_HK_URL = "https://llm-center.modelbest.co/llm/mcp?mcp=serp-search"
MAX_RESPONSE_BYTES = 1 << 20
USER_AGENT = "OmniExperience/1.0 (+https://modelbest.cn)"
MCP_ACCEPT = "application/json, text/event-stream"
MCP_TOOL_NAME = "serp_google_search"


def search_serp_hk(
    query: str,
    api_key: str,
    num: int = 5,
    *,
    timeout_seconds: float = 20.0,
    url: str | None = None,
) -> str:
    """调用 MCP 搜索并返回供应商原始 JSON 文本。"""

    if not query.strip():
        raise ValueError("query is required")
    if not api_key.strip():
        raise ValueError("serp_google_search api key is empty")
    if isinstance(num, bool) or not isinstance(num, int) or num <= 0:
        raise ValueError("num must be a positive integer")

    endpoint = (url or SERP_HK_URL).strip() or SERP_HK_URL
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": MCP_TOOL_NAME,
                    "arguments": {"q": query},
                },
            },
            ensure_ascii=False,
        ).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": "Bearer " + api_key,
            "Content-Type": "application/json",
            "Accept": MCP_ACCEPT,
            "User-Agent": USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = response.read(MAX_RESPONSE_BYTES).decode("utf-8", errors="replace")
            content_type = _header_value(response, "Content-Type")
    except urllib.error.HTTPError as error:
        error_body = error.read(MAX_RESPONSE_BYTES).decode("utf-8", errors="replace")
        raise RuntimeError(f"http {error.code}: {error_body[:1000]}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(str(error.reason)) from error
    return _vendor_body(body, content_type)


def _header_value(response: object, name: str) -> str:
    headers = getattr(response, "headers", None)
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return ""
    value = getter(name) or getter(name.lower())
    return value if isinstance(value, str) else ""


def _vendor_body(body: str, content_type: str) -> str:
    payload = _mcp_message(body, content_type)
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message") or json.dumps(error, ensure_ascii=False)
        raise RuntimeError(f"mcp error: {message}")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("mcp result is missing")
    texts = _text_contents(result.get("content"))
    if result.get("isError"):
        detail = texts[0] if texts else json.dumps(result, ensure_ascii=False)
        raise RuntimeError(f"mcp tool error: {detail[:1000]}")
    if len(texts) == 1:
        return texts[0]
    if texts:
        return json.dumps(texts, ensure_ascii=False)
    raise RuntimeError("mcp tool returned no text content")


def _mcp_message(body: str, content_type: str) -> Dict[str, Any]:
    if "text/event-stream" in content_type.lower():
        messages = _messages_from_sse(body)
        if not messages:
            raise RuntimeError("empty mcp event stream")
        for message in reversed(messages):
            if "result" in message or "error" in message:
                return message
        return messages[-1]
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"mcp response is not json: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("mcp response is not a json object")
    return payload


def _messages_from_sse(body: str) -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    data_lines: List[str] = []

    def flush() -> None:
        if not data_lines:
            return
        raw = "\n".join(data_lines).strip()
        data_lines.clear()
        if not raw or raw == "[DONE]":
            return
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            messages.append(parsed)

    for line in body.splitlines():
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
        elif line == "":
            flush()
    flush()
    return messages


def _text_contents(content: object) -> List[str]:
    if not isinstance(content, list):
        return []
    texts: List[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        if item.get("type") not in (None, "text"):
            continue
        text = item.get("text")
        if isinstance(text, str) and text:
            texts.append(text)
    return texts
