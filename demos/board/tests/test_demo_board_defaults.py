"""Board demo 的 defaults：Gallery 通用运行时 + board 自己的训练 case 投影。"""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_board_defaults_merge_gallery_config_and_board_specific_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Gallery 的 Demo 侧配置（可连的 API 端点）透传给页面；board 专属字段在其之上合并。

    模型口径（预算、调度、tts_pad）不在这里：页面向所选 API 端点的 /v1/info 查询。
    case 目录不可用时仍要给出可用的兜底（空 segments + 工具定义），不能把整个 defaults 拖垮。
    """

    import demo_server
    from fastapi.testclient import TestClient

    monkeypatch.setenv("DEMO_API_ENDPOINTS", '[{"name": "f9t4", "url": "https://h:7029/e/f9t4"}]')
    monkeypatch.setenv("FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_LISTENING", "30")    # API 侧变量：不能再泄露到 Demo
    # 指向一个空目录：走「case 不可用」的兜底分支。
    monkeypatch.setattr(
        "demos.board.defaults.CASE_FOLDER_CANDIDATES", (str(tmp_path),)
    )

    defaults = TestClient(demo_server.app).get("/api/demos/board/defaults").json()

    assert defaults["api_endpoints"] == [{"name": "f9t4", "url": "https://h:7029/e/f9t4"}]
    assert defaults["default_api_endpoint"] == "https://h:7029/e/f9t4"
    assert not any(key.startswith("non_spoken") for key in defaults)
    assert defaults["default_system"]["segments"] == []
    assert defaults["default_system"]["tools"]
    assert defaults["default_tts_prompt_audio"] is None


@pytest.mark.asyncio
async def test_rejection_defaults_carry_no_board_fields() -> None:
    """拒识 demo 没有自己的 defaults，只应拿到 Gallery 通用字段。"""

    import demo_server
    from fastapi.testclient import TestClient

    defaults = TestClient(demo_server.app).get("/api/demos/rejection/defaults").json()
    assert "default_system" not in defaults
    assert "case_folder" not in defaults
    assert "api_endpoints" in defaults
