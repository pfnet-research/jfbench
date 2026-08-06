from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from jfbench._data import DATA_DIR


def test_importing_constraints_does_not_import_prompt_data(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import jfbench.constraints._group; print('jfbench.prompts' in sys.modules)",
        ],
        cwd=tmp_path,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"


def test_data_dir_points_to_package_data() -> None:
    assert (DATA_DIR / "ifbench_ja_translated.jsonl").is_file()
    assert (DATA_DIR / "ja_stackoverflow_train.jsonl.zst").is_file()
