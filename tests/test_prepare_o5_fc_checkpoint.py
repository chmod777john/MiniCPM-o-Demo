"""O5 FC checkpoint 一键制备流程的轻量回归测试。"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

import pytest
import torch

from tools.checkpoint import prepare_o5_fc_checkpoint as preparation


def _write_checkpoint(path: Path, *, rows: int, include_tts: bool) -> None:
    """写入可供制备流程检查的最小 PT。"""

    state = {
        "llm.model.embed_tokens.weight": torch.zeros(rows, 2),
        "llm.lm_head.weight": torch.zeros(rows, 2),
    }
    if include_tts:
        state["tts.fake.weight"] = torch.ones(1)
    torch.save(state, path)


def _make_request(
    tmp_path: Path,
    *,
    frozen_tts_base_pt: Path | None,
) -> preparation.O5CheckpointPreparationRequest:
    """构造使用测试临时路径的制备请求。"""

    source = tmp_path / "source"
    source.mkdir()
    base_model = tmp_path / "base-model"
    base_model.mkdir()
    model_path = tmp_path / "model-path"
    model_path.mkdir()
    converter = tmp_path / "canonical_converter.py"
    converter.write_text("# fake\n", encoding="utf-8")
    extractor = tmp_path / "extract_backbone.py"
    extractor.write_text("# fake\n", encoding="utf-8")
    backbone_python = tmp_path / "backbone_python"
    backbone_python.write_text("# fake\n", encoding="utf-8")
    backbone_pythonpath = tmp_path / "backbone-pythonpath"
    backbone_pythonpath.mkdir()
    return preparation.O5CheckpointPreparationRequest(
        source_checkpoint_dir=str(source.resolve()),
        output_dir=str((tmp_path / "output").resolve()),
        base_model_dir=str(base_model.resolve()),
        model_path=str(model_path.resolve()),
        converter_script=str(converter.resolve()),
        backbone_extractor_script=str(extractor.resolve()),
        backbone_python=str(backbone_python.resolve()),
        backbone_pythonpath=str(backbone_pythonpath.resolve()),
        tts_policy=preparation.TtsPolicy.AUTO.value,
        frozen_tts_base_pt=(
            str(frozen_tts_base_pt.resolve())
            if frozen_tts_base_pt is not None
            else None
        ),
        dtype="bfloat16",
        mtp_layer=0,
        expected_rows=4,
        expected_tensor_count=3,
        expected_tts_tensor_count=1,
        expected_backbone_key_count=2,
        expected_backbone_shard_count=1,
        keep_intermediate=False,
    )


def _argument(command: Sequence[str], name: str) -> Path:
    """读取命令行选项后的路径参数。"""

    return Path(command[command.index(name) + 1])


def _install_fake_stages(
    monkeypatch: pytest.MonkeyPatch,
    *,
    converted_has_tts: bool,
    executed_scripts: list[str],
) -> None:
    """用轻量 PT 和 backbone 文件替代三个重型子进程。"""

    def fake_run(
        command: Sequence[str],
        *,
        cwd: Path,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        del cwd
        script_name = Path(command[1]).name
        executed_scripts.append(script_name)
        if script_name == "canonical_converter.py":
            _write_checkpoint(
                _argument(command, "--output-path"),
                rows=4,
                include_tts=converted_has_tts,
            )
            return
        if script_name == "merge_frozen_tts.py":
            model_pt = _argument(command, "--model-pt")
            output_pt = _argument(command, "--output-pt")
            state = torch.load(model_pt, map_location="cpu", weights_only=True)
            state["tts.fake.weight"] = torch.ones(1)
            torch.save(state, output_pt)
            output_pt.with_suffix(output_pt.suffix + ".manifest.json").write_text(
                "{}\n",
                encoding="utf-8",
            )
            return
        if script_name == "extract_backbone.py":
            assert extra_env is not None
            output_dir = Path(extra_env["BACKBONE_DIR"])
            output_dir.mkdir()
            (output_dir / "config.json").write_text(
                json.dumps({"vocab_size": 4}),
                encoding="utf-8",
            )
            (output_dir / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.language_model.embed_tokens.weight": (
                                "model-00001.safetensors"
                            ),
                            "lm_head.weight": "model-00001.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )
            return
        raise AssertionError(f"未预期的脚本: {script_name}")

    monkeypatch.setattr(preparation, "_run_command", fake_run)
    monkeypatch.setattr(
        preparation,
        "_read_safetensor_rows",
        lambda backbone_dir, weight_map, tensor_key: 4,
    )


def test_llm_only_checkpoint_merges_frozen_tts_and_extracts_backbone(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM-only DCP 应补冻结 TTS，且不默认保留巨大的中间 PT。"""

    frozen_base = tmp_path / "frozen-base.pt"
    _write_checkpoint(frozen_base, rows=4, include_tts=True)
    request = _make_request(tmp_path, frozen_tts_base_pt=frozen_base)
    executed_scripts: list[str] = []
    _install_fake_stages(
        monkeypatch,
        converted_has_tts=False,
        executed_scripts=executed_scripts,
    )

    manifest = preparation.prepare_o5_fc_checkpoint(request)

    assert executed_scripts == [
        "canonical_converter.py",
        "merge_frozen_tts.py",
        "extract_backbone.py",
    ]
    assert manifest.frozen_tts_merged is True
    assert manifest.tts_tensor_count == 1
    assert manifest.embedding_rows == 4
    assert not (Path(request.output_dir) / "canonical_llm_only.pt").exists()
    assert (Path(request.output_dir) / "model.pt").is_file()
    assert (Path(request.output_dir) / "backbone").is_dir()


def test_checkpoint_with_trained_tts_skips_frozen_merge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """已包含 TTS 的 checkpoint 必须原样成为部署 PT，不得混入 frozen base。"""

    request = _make_request(tmp_path, frozen_tts_base_pt=None)
    executed_scripts: list[str] = []
    _install_fake_stages(
        monkeypatch,
        converted_has_tts=True,
        executed_scripts=executed_scripts,
    )

    manifest = preparation.prepare_o5_fc_checkpoint(request)

    assert executed_scripts == [
        "canonical_converter.py",
        "extract_backbone.py",
    ]
    assert manifest.frozen_tts_merged is False
    assert manifest.tts_tensor_count == 1


def test_resume_reuses_completed_assets_without_running_heavy_stages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """同一请求的 --resume 应复用完整 PT/backbone，并保持合并血缘。"""

    frozen_base = tmp_path / "frozen-base.pt"
    _write_checkpoint(frozen_base, rows=4, include_tts=True)
    request = _make_request(tmp_path, frozen_tts_base_pt=frozen_base)
    executed_scripts: list[str] = []
    _install_fake_stages(
        monkeypatch,
        converted_has_tts=False,
        executed_scripts=executed_scripts,
    )
    preparation.prepare_o5_fc_checkpoint(request)

    def reject_run(
        command: Sequence[str],
        *,
        cwd: Path,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        raise AssertionError(f"续跑不应执行子进程: {command}, cwd={cwd}")

    monkeypatch.setattr(preparation, "_run_command", reject_run)
    resumed = preparation.prepare_o5_fc_checkpoint(request, resume=True)

    assert resumed.frozen_tts_merged is True
    assert resumed.tts_tensor_count == 1


def test_non_empty_output_requires_explicit_resume(
    tmp_path: Path,
) -> None:
    """默认不得覆盖未知输出目录。"""

    request = _make_request(tmp_path, frozen_tts_base_pt=None)
    output_dir = Path(request.output_dir)
    output_dir.mkdir()
    (output_dir / "foreign.txt").write_text("do not overwrite\n", encoding="utf-8")

    try:
        preparation.prepare_o5_fc_checkpoint(request)
    except FileExistsError as error:
        assert "--resume" in str(error)
    else:
        raise AssertionError("非空目录未被拒绝")


def test_resume_reuses_completed_canonical_part_after_tts_policy_correction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """策略门禁纠正后应复用已完整写出的 canonical PT，不重复转换。"""

    frozen_base = tmp_path / "frozen-base.pt"
    _write_checkpoint(frozen_base, rows=4, include_tts=True)
    previous_request = replace(
        _make_request(tmp_path, frozen_tts_base_pt=frozen_base),
        tts_policy=preparation.TtsPolicy.MERGE_FROZEN.value,
    )
    output_dir = Path(previous_request.output_dir)
    output_dir.mkdir()
    (output_dir / preparation.REQUEST_FILENAME).write_text(
        json.dumps(asdict(previous_request), ensure_ascii=False),
        encoding="utf-8",
    )
    _write_checkpoint(
        output_dir / ".canonical.pt.part",
        rows=4,
        include_tts=True,
    )
    corrected_request = replace(
        previous_request,
        tts_policy=preparation.TtsPolicy.CHECKPOINT_ONLY.value,
    )
    executed_scripts: list[str] = []
    _install_fake_stages(
        monkeypatch,
        converted_has_tts=True,
        executed_scripts=executed_scripts,
    )

    manifest = preparation.prepare_o5_fc_checkpoint(
        corrected_request,
        resume=True,
    )

    assert executed_scripts == ["extract_backbone.py"]
    assert manifest.frozen_tts_merged is False
    assert manifest.tts_tensor_count == 1
