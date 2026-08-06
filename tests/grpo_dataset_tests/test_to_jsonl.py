from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

from datasets import Dataset
import yaml

from jfbench.grpo_dataset import to_jsonl


if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def test_to_jsonl_writes_prompt_and_metadata_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = Dataset.from_dict(
        {
            "prompt_source": ["ja_stackoverflow"],
            "instruction": ["質問です"],
            "response": ["回答です"],
            "reasoning_content": ["考えました"],
            "constraint_count": [4],
        }
    )
    input_dir = tmp_path / "input"
    dataset.save_to_disk(str(input_dir))
    output_dir = tmp_path / "output"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "to_jsonl.py",
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--allow-overwrite",
        ],
    )

    to_jsonl.main()

    with (output_dir / "train.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle]

    assert rows == [
        {
            "prompt": [{"role": "user", "content": "質問です"}],
            "metadata": '{"prompt_source":"ja_stackoverflow","instruction":"質問です","response":"回答です","reasoning_content":"考えました","constraint_count":4}',
        }
    ]

    with (output_dir / "README.md").open() as handle:
        readme = yaml.safe_load(handle.read().strip("-\n"))

    assert readme == {
        "dataset_info": {
            "features": [
                {
                    "name": "prompt",
                    "list": [
                        {"name": "content", "dtype": "string"},
                        {"name": "role", "dtype": "string"},
                    ],
                },
                {"name": "metadata", "dtype": "string"},
            ]
        }
    }
    captured = capsys.readouterr()
    assert "Loaded records: 1" in captured.out
    assert "Removed records: 0" in captured.out
    assert "Written records: 1" in captured.out


def test_to_jsonl_filters_rows_with_empty_required_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    dataset = Dataset.from_dict(
        {
            "instruction": ["keep", "drop-a", "drop-b"],
            "response": ["ok", "", "ok"],
            "reasoning_content": ["ok", "ok", None],
            "constraint_count": [1, 2, 4],
            "constraint_types": [
                ["WordsLengthConstraint"],
                ["NegationLogicConstraint"],
                ["DoubleNegationLogicConstraint"],
            ],
            "constraints": [
                [{"name": "WordsLengthConstraint"}],
                [{"name": "NegationLogicConstraint"}],
                [{"name": "DoubleNegationLogicConstraint"}],
            ],
        }
    )
    input_dir = tmp_path / "input"
    dataset.save_to_disk(str(input_dir))
    output_dir = tmp_path / "output"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "to_jsonl.py",
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--allow-overwrite",
        ],
    )

    to_jsonl.main()

    with (output_dir / "train.jsonl").open() as handle:
        rows = [json.loads(line) for line in handle]

    assert len(rows) == 1
    assert rows[0]["prompt"] == [{"role": "user", "content": "keep"}]
    captured = capsys.readouterr()
    assert "Loaded records: 3" in captured.out
    assert "Removed records: 2" in captured.out
    assert "Written records: 1" in captured.out
