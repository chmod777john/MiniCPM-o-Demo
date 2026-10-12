"""Demo Gallery 独立服务：页面 + 各 Demo 自己的后端，经协议 v4 访问任意 API 端点。

Demo 与 API 是两件事：本进程不加载模型、不认识 Worker，只在浏览器里让用户选一个 API 端点
（``https://<host>:<port>/e/<端点>``），之后浏览器直接连那个端点的 ``/v1/realtime``。

启动::

    DEMO_API_ENDPOINTS='[{"name":"f9t4","url":"https://47.95.219.248:7029/e/f9t4"}]' \\
    DEMO_AUDIO_ROOTS=/path/to/ref_audio:/path/to/dataset \\
    python demo_server.py --port 10032 --https

Demo 侧配置见 ``demo_gallery/runtime.py``（端点列表、默认 preset）与 ``demo_gallery/audio.py``
（允许读音频的目录）。各 Demo 自己的配置（例如 subagent 的 ``SUBAGENT_CLAUDE_*``）见其目录。
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from demo_gallery import install_gallery

ROOT = Path(__file__).resolve().parent

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

app = FastAPI(title="MiniCPM-o Demo Gallery", docs_url=None, redoc_url=None)
# 共享前端：URL 与磁盘路径同构（demos/<id>/web/*.js 用 ../../../static/demo-gallery/... 相对 import）。
app.mount("/static/demo-gallery", StaticFiles(directory=str(ROOT / "static" / "demo-gallery")), name="demo-gallery")
install_gallery(app)


@app.get("/", include_in_schema=False)
async def index() -> RedirectResponse:
    return RedirectResponse("/demos")


@app.get("/health", include_in_schema=False)
async def health() -> dict:
    return {"status": "healthy"}


def main() -> None:
    parser = argparse.ArgumentParser(description="MiniCPM-o Demo Gallery（独立于 API）")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10032)
    parser.add_argument("--https", action="store_true", help="启用 HTTPS（浏览器麦克风需要安全上下文）")
    parser.add_argument("--ssl-certfile", default=str(ROOT / "certs" / "cert.pem"))
    parser.add_argument("--ssl-keyfile", default=str(ROOT / "certs" / "key.pem"))
    args = parser.parse_args()
    ssl = {"ssl_certfile": args.ssl_certfile, "ssl_keyfile": args.ssl_keyfile} if args.https else {}
    uvicorn.run(app, host=args.host, port=args.port, **ssl)


if __name__ == "__main__":
    main()
