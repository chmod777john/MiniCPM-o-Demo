"""Worker 模板、job 规格与就绪文件的契约。"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from control_plane.specs import WorkerJobSpec, WorkerTemplate, read_ready, write_ready

TEMPLATE = {"name": "f9t4-rejection-iter2000", "model_dir": "/abs/ckpt", "tts_pad_protocol": "legacy"}


def test_template_defaults_match_demo_deployments() -> None:
    template = WorkerTemplate.model_validate(TEMPLATE)
    assert (template.non_spoken_budget_while_listening, template.non_spoken_budget_while_speaking) == (45, 25)
    assert template.unit_sec == 1.0 and template.gpus == 2


@pytest.mark.parametrize("patch", [
    {"name": "Has Space"}, {"name": "-lead"}, {"model_dir": "relative/dir"},
    {"tts_pad_protocol": "auto"}, {"unknown_field": 1},
])
def test_template_rejects_bad_input(patch: dict) -> None:
    with pytest.raises(ValidationError):
        WorkerTemplate.model_validate({**TEMPLATE, **patch})


def test_tts_pad_protocol_has_no_default() -> None:
    """口径必须显式给：按目录日期推断曾经错配（0930_final_merge 实为 revised）。"""

    with pytest.raises(ValidationError):
        WorkerTemplate.model_validate({k: v for k, v in TEMPLATE.items() if k != "tts_pad_protocol"})


def test_ready_file_roundtrip(tmp_path: Path) -> None:
    assert read_ready(tmp_path) is None
    write_ready(tmp_path, worker_id="f9t4-1", host="10.0.0.5", port=22410, model_info={"profile_id": "p"})
    ready = read_ready(tmp_path)
    assert (ready.worker_id, ready.host, ready.port, ready.model_info) == ("f9t4-1", "10.0.0.5", 22410, {"profile_id": "p"})


def test_job_spec_ids_are_validated(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        WorkerJobSpec(worker_id="Bad/ID", endpoint="f9t4", template=WorkerTemplate(**TEMPLATE), run_dir=str(tmp_path))
