from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from jfbench.dpo_dataset.screening import screen_dataset


if TYPE_CHECKING:
    from pathlib import Path


def _write_jsonl(path: Path, records: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def test_screen_dataset_filters_records(tmp_path: Path) -> None:
    input_file = tmp_path / "train.jsonl"
    records = [
        {
            "prompt_source": "ja_stackoverflow",
            "instruction": "write",
            "chosen": "good",
            "rejected": "bad",
        },
        {
            "prompt_source": "ja_stackoverflow",
            "instruction": "write",
            "chosen": "unknown",
            "rejected": "bad",
        },
        {
            "prompt_source": "ja_stackoverflow",
            "instruction": "write",
            "chosen": "good",
            "rejected": "pass",
        },
    ]
    _write_jsonl(input_file, records)
    analysis = tmp_path / "analysis.txt"
    analysis.write_text(
        """
Constraint satisfaction by record
---------------------------------
  - [0] prompt_id=sample-0: chosen=pass, rejected=fail
  - [1] prompt_id=sample-1: chosen=unknown, rejected=fail
  - [2] prompt_id=sample-2: chosen=pass, rejected=pass
""",
        encoding="utf-8",
    )
    output_file = tmp_path / "output.jsonl"

    screen_dataset(
        input_path=input_file,
        analysis_result_path=analysis,
        output_path=output_file,
    )

    with output_file.open(encoding="utf-8") as handle:
        filtered_records = [json.loads(line) for line in handle]
    assert filtered_records == [records[0]]


def test_screen_dataset_detects_missing_statuses(tmp_path: Path) -> None:
    input_file = tmp_path / "train.jsonl"
    records = [
        {
            "prompt_source": "ja_stackoverflow",
            "instruction": "write",
            "chosen": "a",
            "rejected": "b",
        },
        {
            "prompt_source": "ja_stackoverflow",
            "instruction": "write",
            "chosen": "c",
            "rejected": "d",
        },
    ]
    _write_jsonl(input_file, records)
    analysis = tmp_path / "analysis.txt"
    analysis.write_text(
        """
Constraint satisfaction by record
---------------------------------
  - [0] prompt_id=sample-0: chosen=pass, rejected=fail
""",
        encoding="utf-8",
    )
    output_file = tmp_path / "output.jsonl"

    with pytest.raises(ValueError, match="fewer entries"):
        screen_dataset(
            input_path=input_file,
            analysis_result_path=analysis,
            output_path=output_file,
        )
