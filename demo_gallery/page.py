"""Shell HTML 渲染：把当前 demo 的入口模块 URL 等元信息注入页面骨架。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from demo_gallery.registry import DemoSpec

_TEMPLATES = Path(__file__).resolve().parent / "templates"
_STYLES_SLOT = "<!--GALLERY_STYLES-->"
_BOOTSTRAP_SLOT = "<!--GALLERY_BOOTSTRAP-->"


def render_shell(spec: DemoSpec) -> str:
    """渲染某个 demo 的页面。

    参数:
        spec: 已注册的 Demo。

    返回:
        注入了样式 <link> 与 bootstrap JSON 的完整 HTML。

    异常:
        RuntimeError: 骨架缺少占位符（宁可直接报错，也不要静默产出半成品页面 ——
            旧实现用 ``html.replace("<body>", ...)`` 依赖骨架恰好是裸 <body>，很脆）。
    """

    html = (_TEMPLATES / "shell.html").read_text(encoding="utf-8")
    for slot in (_STYLES_SLOT, _BOOTSTRAP_SLOT):
        if slot not in html:
            raise RuntimeError(f"shell.html 缺少占位符 {slot}")

    bootstrap = {
        "demoId": spec.demo_id,
        "title": spec.title,
        "summary": spec.summary,
        "entry": spec.entry_url,
        "assetsBase": spec.assets_url,
        "api": {
            "base": spec.api_prefix,
            "defaults": f"{spec.api_prefix}/defaults",
        },
    }
    links = "\n  ".join(
        f'<link rel="stylesheet" href="{spec.assets_url}/{style}" />'
        for style in spec.styles
    )
    return (
        html.replace(_STYLES_SLOT, links)
        .replace(_BOOTSTRAP_SLOT, _bootstrap_script(bootstrap))
        .replace("{{DEMO_TITLE}}", spec.title)
        .replace("{{DEMO_ID}}", spec.demo_id)
    )


def render_index(specs: Sequence[DemoSpec]) -> str:
    """渲染 Gallery 索引页：列出全部已注册 demo。

    参数:
        specs: 已按 order 排好的 Demo 列表。

    返回:
        完整 HTML；新 demo 注册后自动出现，不需要改任何页面。
    """

    html = (_TEMPLATES / "index.html").read_text(encoding="utf-8")
    cards = "\n".join(
        f'''      <a class="demo-card" href="{spec.page_url}">
        <strong>{_escape(spec.title)}</strong>
        <span class="demo-id">{_escape(spec.demo_id)}</span>
        <p>{_escape(spec.summary)}</p>
      </a>'''
        for spec in specs
    )
    if not cards:
        cards = '      <p class="empty">还没有注册任何 Demo。</p>'
    return html.replace("<!--GALLERY_CARDS-->", cards)


def _bootstrap_script(bootstrap: dict) -> str:
    # `</` 必须转义：demo 的 summary 里一个尖括号就能把 <script> 提前关掉。
    payload = json.dumps(bootstrap, ensure_ascii=False).replace("</", "<\\/")
    return f'<script id="gallery-bootstrap" type="application/json">{payload}</script>'


def _escape(value: str) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
