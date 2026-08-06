from __future__ import annotations

import json
from typing import TYPE_CHECKING

import datasets
from datasets import Dataset

from jfbench.grpo_dataset import postprocess


if TYPE_CHECKING:
    from pathlib import Path


def _build_row(
    *,
    constraint_types: list[str],
    constraint_kwargs: list[dict[str, object]],
    prompt_document: str = "Sample document",
    seed: int = 123,
) -> dict[str, object]:
    return {
        "prompt_source": "ja_stackoverflow",
        "prompt_index": 0,
        "prompt_id": "ja_stackoverflow-0",
        "prompt_document": prompt_document,
        "instruction": "instruction",
        "prompt": "prompt",
        "model": "model",
        "model_name": "model",
        "attempt": 1,
        "response": "response",
        "reasoning_content": "reasoning",
        "evaluation": {"constraints_passed": True},
        "constraint_count": len(constraint_types),
        "constraint_types": constraint_types,
        "constraint_groups": ["Length" for _ in constraint_types],
        "constraint_instructions": ["instruction" for _ in constraint_types],
        "constraint_kwargs": constraint_kwargs,
        "constraints": [
            {"name": name, "group": "Length", "instructions": "instruction"}
            for name in constraint_types
        ],
        "data_id": "data-0",
        "seed": seed,
    }


def test_postprocess_rewrites_constraints_to_json_strings(tmp_path: Path) -> None:
    row = _build_row(
        constraint_types=["SentencesLengthConstraint"],
        constraint_kwargs=[
            {
                "sentences_length_constraint_min_sentences": 3,
                "sentences_length_constraint_max_sentences": 4,
            }
        ],
    )
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    Dataset.from_list([row]).save_to_disk(str(input_dir))

    loaded, written = postprocess.postprocess_datasets(
        [input_dir], output_path=output_dir, allow_overwrite=True
    )

    assert loaded == 1
    assert written == 1
    saved = datasets.load_from_disk(str(output_dir))
    assert isinstance(saved, datasets.DatasetDict)
    train = saved["train"]
    assert len(train) == 1
    assert train["constraint_types"][0] == ["SentencesLengthConstraint"]
    assert "constraint_kwargs" not in train.column_names
    payload = json.loads(train["constraints"][0][0])
    assert payload["name"] == "SentencesLengthConstraint"
    assert payload["kwargs"] == {"min_sentences": 3, "max_sentences": 4}


def test_postprocess_reconstructs_nested_logic_constraint(tmp_path: Path) -> None:
    nested = json.dumps(
        {
            "name": "PrefixProcessingConstraint",
            "kwargs": {"prefix": "<これは文頭です>"},
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    row = _build_row(
        constraint_types=["DoubleNegationLogicConstraint"],
        constraint_kwargs=[
            {
                "double_negation_logic_constraint_positive_constraint": nested,
            }
        ],
    )
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    Dataset.from_list([row]).save_to_disk(str(input_dir))

    postprocess.postprocess_datasets([input_dir], output_path=output_dir, allow_overwrite=True)

    saved = datasets.load_from_disk(str(output_dir))
    assert isinstance(saved, datasets.DatasetDict)
    assert "constraint_kwargs" not in saved["train"].column_names
    payload = json.loads(saved["train"]["constraints"][0][0])
    positive = payload["kwargs"]["positive_constraint"]
    assert positive["__type__"] == "constraint"
    assert positive["value"]["name"] == "PrefixProcessingConstraint"
    assert positive["value"]["kwargs"] == {"prefix": "<これは文頭です>"}
