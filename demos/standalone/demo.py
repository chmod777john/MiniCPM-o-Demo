"""Standalone Event Demo：用户在输入框里输入任意文本，回车即作为 ``input.standalone_event`` 注入。

这是注入外部内容最通用的方式：不关联任何 tool_call，模型在下一个被处理的 Unit 的
input_event 槽里看到 ``<standalone_event>正文</standalone_event>``。

本 Demo 只做注入，**与系统里有什么工具无关**：preset 是 ``[统一前缀, 参考音频]``，不带 tools。
preset 全部在前端（``web/quick-settings.js``），后端除 Gallery 通用 defaults 外不提供任何东西。
"""

from __future__ import annotations

from demo_gallery import demo


@demo(
    demo_id="standalone",
    title="MiniCPM-o · Standalone Event",
    summary="在输入框里输入任意文本、回车，作为 standalone input_event 注入模型（不带工具）",
    order=30,
    styles=("standalone.css",),
)
class StandaloneDemo:
    """无专属 defaults。"""
