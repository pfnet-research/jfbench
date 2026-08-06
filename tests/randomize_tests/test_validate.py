from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

from datasets import Dataset
import pytest

from jfbench.randomize.validate import validate_randomization


if TYPE_CHECKING:
    from pathlib import Path


def _save_dataset(records: list[dict[str, Any]], path: Path) -> None:
    dataset = Dataset.from_list(records)
    dataset.save_to_disk(str(path))


def test_validate_randomization_accepts_matching_sft_records(tmp_path: Path) -> None:
    before = [
        {
            "data_id": "sample-0",
            "prompt_document": "Document text",
            "constraint_types": ["FormatConstraint"],
            "evaluation": {"FormatConstraint": True},
            "response": "Answer",
        }
    ]
    after = [
        {
            "data_id": "sample-0",
            "prompt_document": "Document text",
            "constraint_types": ["FormatConstraint"],
            "evaluation": {"FormatConstraint": True},
            "response": "Answer",
            "instruction": "Updated instruction",
        }
    ]
    before_dir = tmp_path / "before"
    after_dir = tmp_path / "after"
    _save_dataset(before, before_dir)
    _save_dataset(after, after_dir)

    validate_randomization(before_dir=before_dir, after_dir=after_dir)


def test_validate_randomization_detects_prompt_differences(tmp_path: Path) -> None:
    before = [
        {
            "data_id": "sample-0",
            "prompt_document": "Original",
            "constraint_types": [],
            "evaluation": {},
        }
    ]
    after = [
        {
            "data_id": "sample-0",
            "prompt_document": "Modified",
            "constraint_types": [],
            "evaluation": {},
        }
    ]
    before_dir = tmp_path / "before"
    after_dir = tmp_path / "after"
    _save_dataset(before, before_dir)
    _save_dataset(after, after_dir)

    with pytest.raises(ValueError) as excinfo:
        validate_randomization(before_dir=before_dir, after_dir=after_dir)
    assert "Prompt mismatch" in str(excinfo.value)


def test_validate_randomization_compares_dpo_evaluations(tmp_path: Path) -> None:
    before = [
        {
            "data_id": "sample-1",
            "prompt_document": "",
            "prompt": "Prompt text",
            "constraint_types": ["FormatConstraint"],
            "chosen": {"evaluation": {"FormatConstraint": True}, "response": "good"},
            "rejected": {"evaluation": {"FormatConstraint": False}, "response": "bad"},
        }
    ]
    after = [
        {
            "data_id": "sample-1",
            "prompt_document": "",
            "prompt": "Prompt text",
            "constraint_types": ["FormatConstraint"],
            "chosen": {"evaluation": {"FormatConstraint": False}, "response": "good"},
            "rejected": {"evaluation": {"FormatConstraint": False}, "response": "bad"},
        }
    ]
    before_dir = tmp_path / "before"
    after_dir = tmp_path / "after"
    _save_dataset(before, before_dir)
    _save_dataset(after, after_dir)

    with pytest.raises(ValueError) as excinfo:
        validate_randomization(before_dir=before_dir, after_dir=after_dir)
    assert "Constraint evaluations mismatch" in str(excinfo.value)


def test_validate_randomization_accepts_multiple_directories(tmp_path: Path) -> None:
    before_one = [
        {
            "data_id": "sample-0",
            "prompt_document": "Doc",
            "constraint_types": ["FormatConstraint"],
            "evaluation": {"FormatConstraint": True},
            "response": "ok",
        }
    ]
    before_two = [
        {
            "data_id": "sample-2",
            "prompt_document": "Doc2",
            "constraint_types": ["Constraint"],
            "evaluation": {"Constraint": False},
            "response": "bad",
        }
    ]
    after_one = [
        {
            "data_id": "sample-0",
            "prompt_document": "Doc",
            "constraint_types": ["FormatConstraint"],
            "evaluation": {"FormatConstraint": True},
            "response": "ok",
        }
    ]
    after_two = [
        {
            "data_id": "sample-2",
            "prompt_document": "Doc2",
            "constraint_types": ["Constraint"],
            "evaluation": {"Constraint": False},
            "response": "bad",
        }
    ]
    before_dir_one = tmp_path / "before1"
    before_dir_two = tmp_path / "before2"
    after_dir_one = tmp_path / "after1"
    after_dir_two = tmp_path / "after2"
    _save_dataset(before_one, before_dir_one)
    _save_dataset(before_two, before_dir_two)
    _save_dataset(after_one, after_dir_one)
    _save_dataset(after_two, after_dir_two)

    validate_randomization(
        before_dirs=[before_dir_one, before_dir_two],
        after_dirs=[after_dir_one, after_dir_two],
    )
