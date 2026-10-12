"""Worker 模板、job 规格与就绪文件的契约。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from control_plane.specs import WorkerJobSpec, WorkerTemplate, read_ready, write_ready

TEMPLATE = {"name": "f9t4-rejection-iter2000", "model_dir": "/abs/ckpt", "tts_pad_protocol": "legacy"}


def test_template_defaults_match_current_models() -> None:
    """只填 checkpoint 目录即可：Session 默认值取当前模型的口径（revised、45/25、quality）。"""

    template = WorkerTemplate.model_validate({"name": "m", "model_dir": "/abs/ckpt"})
    assert template.tts_pad_protocol == "revised"
    assert (template.non_spoken_budget_while_listening, template.non_spoken_budget_while_speaking) == (45, 25)
    assert template.non_spoken_scheduling == "quality" and template.gpus == 2


def test_template_drops_retired_unit_sec() -> None:
    """旧 state.json 里的模板带 unit_sec（已改为纯 Session 参数），读入时丢弃而不是报错。"""

    template = WorkerTemplate.model_validate({**TEMPLATE, "unit_sec": 1.0})
    assert "unit_sec" not in template.model_dump()


@pytest.mark.parametrize("patch", [
    {"name": "Has Space"}, {"name": "-lead"}, {"model_dir": "relative/dir"},
    {"tts_pad_protocol": "auto"}, {"unknown_field": 1},
])
def test_template_rejects_bad_input(patch: dict) -> None:
    with pytest.raises(ValidationError):
        WorkerTemplate.model_validate({**TEMPLATE, **patch})


def test_ready_file_roundtrip(tmp_path: Path) -> None:
    assert read_ready(tmp_path) is None
    write_ready(tmp_path, worker_id="f9t4-1", host="10.0.0.5", port=22410, model_info={"profile_id": "p"})
    ready = read_ready(tmp_path)
    assert (ready.worker_id, ready.host, ready.port, ready.model_info) == ("f9t4-1", "10.0.0.5", 22410, {"profile_id": "p"})


def test_job_spec_ids_are_validated(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        WorkerJobSpec(worker_id="Bad/ID", endpoint="f9t4", template=WorkerTemplate(**TEMPLATE), run_dir=str(tmp_path))
