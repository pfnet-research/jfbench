from __future__ import annotations

import os
from pathlib import Path


def _find_data_dir() -> Path:
    env_data_dir = os.environ.get("JFBENCH_DATA_DIR")
    if env_data_dir:
        return Path(env_data_dir)

    package_data_dir = Path(__file__).resolve().parent / "data"
    if package_data_dir.exists():
        return package_data_dir

    d = Path(__file__).resolve().parent
    while d != d.parent:
        if (d / "pyproject.toml").exists():
            return d / "data"
        d = d.parent
    return package_data_dir


DATA_DIR = _find_data_dir()
