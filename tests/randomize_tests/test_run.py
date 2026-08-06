from __future__ import annotations

from typing import Any
from typing import ClassVar
from typing import TYPE_CHECKING

from jfbench.randomize.run import _parse_args
from jfbench.randomize.run import randomize_dataset


if TYPE_CHECKING:
    from pathlib import Path


def test_parse_args_supports_input_and_output_dirs(tmp_path: Path) -> None:
    args = _parse_args(
        [
            "--input-dir",
            str(tmp_path / "input"),
            "--output-dir",
            str(tmp_path / "output"),
            "--seed",
            "7",
        ]
    )

    assert args.input_dir == tmp_path / "input"
    assert args.output_dir == tmp_path / "output"
    assert args.seed == 7


def test_randomize_dataset_uses_specified_directories(monkeypatch: Any, tmp_path: Path) -> None:
    class _StubDataset:
        load_calls: ClassVar[list[str]] = []
        save_calls: ClassVar[list[str]] = []
        last_instance: ClassVar[_StubDataset | None] = None

        def __init__(self) -> None:
            self.map_calls: list[dict[str, Any]] = []

        @classmethod
        def load_from_disk(cls, path: str) -> "_StubDataset":
            cls.load_calls.append(path)
            instance = cls()
            cls.last_instance = instance
            return instance

        def map(self, func: Any, desc: str) -> "_StubDataset":
            self.map_calls.append({"desc": desc, "func": func})
            return self

        def save_to_disk(self, path: str) -> None:
            type(self).save_calls.append(path)

    class _StubConstraintInventory:
        @property
        def all_wrappers(self) -> tuple[Any, ...]:
            return ()

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()

    monkeypatch.setattr("jfbench.randomize.run.Dataset", _StubDataset)
    monkeypatch.setattr("jfbench.randomize.run.ConstraintInventory", _StubConstraintInventory)

    randomize_dataset(input_dir=input_dir, output_dir=output_dir)

    assert _StubDataset.load_calls == [str(input_dir)]
    assert _StubDataset.save_calls == [str(output_dir)]
    assert _StubDataset.last_instance is not None
    assert _StubDataset.last_instance.map_calls[0]["desc"] == "Randomizing constraints"
