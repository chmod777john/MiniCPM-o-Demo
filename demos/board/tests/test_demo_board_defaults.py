"""Board demo 的 defaults：Gallery 通用运行时 + board 自己的训练 case 投影。"""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_board_defaults_merge_runtime_and_board_specific_fields(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """launcher 注入的运行时参数透传给页面；board 专属字段在其之上合并。

    case 目录不可用时仍要给出可用的兜底（空 segments + 工具定义），不能把整个 defaults 拖垮。
    """

    import gateway
    from fastapi.testclient import TestClient

    monkeypatch.setenv("CHECKPOINT_PROFILE_ID", "profile_test")
    monkeypatch.setenv("FC_DUPLEX_NON_SPOKEN_SCHEDULING", "quality")
    monkeypatch.setenv("FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_LISTENING", "30")
    monkeypatch.setenv("FC_DUPLEX_NON_SPOKEN_BUDGET_WHILE_SPEAKING", "15")
    monkeypatch.setenv("FC_DUPLEX_UNIT_SEC", "1.0")
    # 指向一个空目录：走「case 不可用」的兜底分支。
    monkeypatch.setattr(
        "demos.board.defaults.CASE_FOLDER_CANDIDATES", (str(tmp_path),)
    )

    defaults = TestClient(gateway.app).get("/api/demos/board/defaults").json()

    assert defaults["checkpoint_profile_id"] == "profile_test"
    assert defaults["non_spoken_scheduling"] == "quality"
    assert defaults["non_spoken_budget_while_listening"] == 30
    assert defaults["non_spoken_budget_while_speaking"] == 15
    assert defaults["unit_sec"] == 1.0
    assert defaults["default_system"]["segments"] == []
    assert defaults["default_system"]["tools"]
    assert defaults["default_tts_prompt_audio"] is None


@pytest.mark.asyncio
async def test_rejection_defaults_carry_no_board_fields() -> None:
    """拒识 demo 没有自己的 defaults，只应拿到通用运行时字段。

    （改造前它无条件调 board 的 endpoint，再把用不上的 default_system 丢掉。）
    """

    import gateway
    from fastapi.testclient import TestClient

    defaults = TestClient(gateway.app).get("/api/demos/rejection/defaults").json()
    assert "default_system" not in defaults
    assert "case_folder" not in defaults
    assert "model_identity" in defaults
