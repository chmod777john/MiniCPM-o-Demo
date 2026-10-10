"""Demo 脚手架。复制整个 demos/_template/ 成 demos/<你的 id>/，把三处 _template 改掉即可。

三处要改的地方：
  1. 本文件的 demo_id / title / summary；
  2. __init__.py 里的 import 路径；
  3. web/<id>.css 文件名、以及 CSS 里的 [data-demo="<id>"] 命名空间。

规范详见 demos/README.md。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from demo_gallery import demo

# 这个 Demo 自己的 API。由 Gallery 挂到 /api/demos/<id>/**，不会碰 gateway 主路由。
# 不需要后端接口的纯前端 Demo 可以整段删掉，并去掉下面的 router= 参数。
router = APIRouter()


@demo(
    demo_id="_template",          # 必须等于目录名
    title="Demo 模板",
    summary="复制我，改三处名字，就有一个新 Demo",
    order=999,
    styles=("_template.css",),    # 相对 web/
    router=router,
)
class TemplateDemo:
    """可选：实现 defaults()，返回值会合并进 /api/demos/<id>/defaults。

    Gallery 的通用运行时字段（模型路径、budget、tts_pad 等）会自动带上，不用自己拼。
    """

    def defaults(self) -> dict[str, Any]:
        return {}


@router.get("/example")
async def example() -> dict[str, Any]:
    """示例接口：前端用 ctx.api.url('example') 拼出完整地址。"""

    return {"ok": True}
