from __future__ import annotations

from typing import TYPE_CHECKING

from datasets import Dataset
from datasets import DatasetDict

from jfbench.sft_dataset import screening


if TYPE_CHECKING:
    from pathlib import Path


def test_merge_dataset_directories_combines_multiple_inputs(tmp_path: Path) -> None:
    dataset_one = Dataset.from_dict({"instruction": ["hello"], "response": ["world"]})
    dataset_two = Dataset.from_dict({"instruction": ["hola"], "response": ["mundo"]})
    input_dir_one = tmp_path / "input_one"
    input_dir_two = tmp_path / "input_two"
    dataset_one.save_to_disk(str(input_dir_one))
    dataset_two.save_to_disk(str(input_dir_two))

    merged = screening.merge_dataset_directories([input_dir_one, input_dir_two])

    assert merged["train"]["instruction"] == ["hello", "hola"]


def test_collect_duplicate_rows_reports_occurrences() -> None:
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "instruction": ["ping", "pong", "ping", "ping", "pong"],
                    "response": ["a", "b", "c", "d", "e"],
                }
            )
        }
    )

    report = screening.collect_duplicate_rows(dataset, "instruction")

    duplicates = {entry.instruction: entry for entry in report["train"]}
    assert duplicates["ping"].occurrences == 3
    assert duplicates["ping"].indices == [0, 2, 3]
    assert duplicates["pong"].occurrences == 2


def test_deduplicate_dataset_removes_later_occurrences() -> None:
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "instruction": ["keep", "drop", "keep", "drop", "solo"],
                    "response": ["a", "b", "c", "d", "e"],
                }
            )
        }
    )

    deduplicated, removed = screening.deduplicate_dataset(dataset, "instruction")

    assert deduplicated["train"]["instruction"] == ["keep", "drop", "solo"]
    assert removed["train"] == 2
