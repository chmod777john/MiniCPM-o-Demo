"""扫描 ``demos/`` 并 import 每个子包，触发其 ``@demo`` 装饰器。"""

from __future__ import annotations

import importlib
import logging
import os
import pkgutil

logger = logging.getLogger("demo_gallery")

DEMOS_PACKAGE = "demos"


def _env_set(name: str) -> set[str]:
    return {item.strip() for item in os.environ.get(name, "").split(",") if item.strip()}


def discover_demos(package: str = DEMOS_PACKAGE) -> list[str]:
    """import 每个 demo 子包。

    参数:
        package: demo 所在的包名（测试用 fixture 包时传别的值）。

    返回:
        成功 import 的模块名列表。

    说明:
        - 只扫子包，跳过 ``_`` 开头（``_template`` 因此天然不参与）。
        - **单个 demo import 失败只记日志并跳过** —— 一个坏 demo 不能让整个 Gateway 起不来。
        - 运维开关：``GALLERY_DEMOS``（白名单）/ ``GALLERY_DEMOS_DISABLED``（黑名单），逗号分隔。
    """

    allow, deny = _env_set("GALLERY_DEMOS"), _env_set("GALLERY_DEMOS_DISABLED")
    root = importlib.import_module(package)
    imported: list[str] = []
    for info in sorted(pkgutil.iter_modules(root.__path__), key=lambda i: i.name):
        if not info.ispkg or info.name.startswith("_"):
            continue
        if (allow and info.name not in allow) or info.name in deny:
            logger.info("[gallery] demo %s 被环境变量禁用，跳过", info.name)
            continue
        name = f"{package}.{info.name}"
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001
            logger.exception("[gallery] demo %s 导入失败，已跳过", name)
            continue
        imported.append(name)
    logger.info("[gallery] 发现 %d 个 demo: %s", len(imported), ", ".join(imported) or "(none)")
    return imported
