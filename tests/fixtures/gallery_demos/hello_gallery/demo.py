"""Fixture demo：结构与真实 demo 完全一致，用来守住可插拔这条性质本身。"""

from __future__ import annotations

from fastapi import APIRouter

from demo_gallery import demo

router = APIRouter()


@demo(
    demo_id="hello_gallery",
    title="Hello Gallery",
    summary="fixture demo，用于测试",
    order=1,
    styles=("hello_gallery.css",),
    router=router,
)
class HelloGalleryDemo:
    def defaults(self) -> dict:
        return {"hello": "world"}


@router.get("/ping")
async def ping() -> dict:
    return {"ok": True}
