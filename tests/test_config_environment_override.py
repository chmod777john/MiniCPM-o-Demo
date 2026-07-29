"""服务配置文件路径环境覆盖的契约测试。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def test_o5_demo_config_path_overrides_repository_config(
    tmp_path: Path,
) -> None:
    """一键启动器应能把 Session 数据目录隔离到用户存储目录。"""

    data_dir = tmp_path / "user-storage" / "data"
    config_path = tmp_path / "service_config.json"
    config_path.write_text(
        f'{{"service": {{"data_dir": "{data_dir}"}}}}',
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["O5_DEMO_CONFIG_PATH"] = str(config_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from config import get_config; print(get_config().data_dir)",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.stdout.strip() == str(data_dir)
