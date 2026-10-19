"""uicard preset 从训练数据原样取 system 与工具（数据不在本机时跳过）。"""

from __future__ import annotations

import pytest

from demos.uicard.presets import PRESET_BUNDLES, UI_CARD_ROOT, uicard_presets

pytestmark = pytest.mark.skipif(not UI_CARD_ROOT.is_dir(), reason="ui_card 训练数据不在本机")


def test_presets_load_training_case_verbatim():
    presets = uicard_presets()
    assert set(presets) == set(PRESET_BUNDLES)
    for preset_id, preset in presets.items():
        assert "error" not in preset, preset.get("error")
        kinds = [segment["kind"] for segment in preset["system"]["segments"]]
        assert kinds == ["audio", "text"], preset_id
        names = [tool["function"]["name"] for tool in preset["system"]["tools"]]
        assert "show_card" in names
        show_card = next(t for t in preset["system"]["tools"] if t["function"]["name"] == "show_card")
        assert "card_id" in show_card["function"]["parameters"]["properties"]
        assert "服务端本地时间：" in preset["system"]["segments"][1]["text"]
        assert preset["tts_prompt_audio"]["file_path"].endswith(".wav")
    assert "tavily_search" in [t["function"]["name"] for t in presets["uicard_1016_r002"]["system"]["tools"]]
    assert "tavily_search" not in [t["function"]["name"] for t in presets["uicard_1004_r011"]["system"]["tools"]]
