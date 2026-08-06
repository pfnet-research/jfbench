from __future__ import annotations

import json
import random
from typing import Any
from typing import TYPE_CHECKING

from datasets import Dataset

from jfbench.dpo_dataset import to_jsonl


if TYPE_CHECKING:
    from pathlib import Path

    import pytest


def _patch_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
            return list(text)

    class DummyAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> DummyTokenizer:
            return DummyTokenizer()

    monkeypatch.setattr(to_jsonl, "AutoTokenizer", DummyAutoTokenizer)


def test_convert_to_jsonl_writes_expected_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_tokenizer(monkeypatch)
    records = [
        {
            "prompt_source": "ja_stackoverflow",
            "instruction": "Write a haiku",
            "chosen": {"response": "A calm river flows"},
            "rejected": {"response": "I refuse"},
            "chosen_reasoning": "Reasoning for haiku",
        },
        {
            "prompt_source": "ifbench",
            "instruction": "List steps",
            "chosen": {"response": "Step response"},
            "rejected": None,
            "chosen_reasoning": "Reasoning for steps",
        },
    ]
    dataset = Dataset.from_list(records)
    input_dir = tmp_path / "input"
    dataset.save_to_disk(str(input_dir))
    output_dir = tmp_path / "output"

    to_jsonl.convert_to_jsonl(
        input_paths=[input_dir],
        output_path=output_dir,
        allow_overwrite=True,
    )

    output_file = output_dir / "train.jsonl"
    assert output_file.exists()
    with output_file.open() as handle:
        lines = [json.loads(line) for line in handle]
    assert lines == [
        {
            "prompt_source": "ja_stackoverflow",
            "prompt": [{"role": "user", "content": "Write a haiku"}],
            "chosen": [
                {
                    "role": "assistant",
                    "content": "A calm river flows",
                    "reasoning_content": "Reasoning for haiku",
                }
            ],
            "rejected": [
                {
                    "role": "assistant",
                    "content": "I refuse",
                    "reasoning_content": "",
                }
            ],
        }
    ]
    captured = capsys.readouterr().out.splitlines()
    assert "Loaded records: 2" in captured
    assert "Empty rejected response: 1" in captured
    readme = output_dir / "README.md"
    assert readme.exists()


def test_convert_to_jsonl_accepts_plain_text_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_tokenizer(monkeypatch)
    records = [
        {
            "prompt_source": "ja_stackoverflow",
            "instruction": "Write a poem",
            "chosen_text": "line a",
            "rejected_text": "line b",
            "chosen_reasoning": "Reasoning for poem",
        }
    ]
    dataset = Dataset.from_list(records)
    input_dir = tmp_path / "text_input"
    dataset.save_to_disk(str(input_dir))
    output_dir = tmp_path / "text_output"

    to_jsonl.convert_to_jsonl(
        input_paths=[input_dir],
        output_path=output_dir,
        allow_overwrite=True,
        chosen_column="chosen_text",
        rejected_column="rejected_text",
    )

    with (output_dir / "train.jsonl").open() as handle:
        payload = json.loads(handle.readline())
    assert payload == {
        "prompt_source": "ja_stackoverflow",
        "prompt": [{"role": "user", "content": "Write a poem"}],
        "chosen": [
            {
                "role": "assistant",
                "content": "line a",
                "reasoning_content": "Reasoning for poem",
            }
        ],
        "rejected": [{"role": "assistant", "content": "line b", "reasoning_content": ""}],
    }


def test_convert_to_jsonl_uses_reasoning_columns_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_tokenizer(monkeypatch)
    records = [
        {
            "instruction": "Explain steps",
            "chosen_text": {"response": "Done", "reasoning_content": "analysis"},
            "rejected_text": {"response": "No", "reasoning_content": ""},
            "chosen_reason": "override chosen",
        }
    ]
    dataset = Dataset.from_list(records)
    input_dir = tmp_path / "reasoning_input"
    dataset.save_to_disk(str(input_dir))
    output_dir = tmp_path / "reasoning_output"

    to_jsonl.convert_to_jsonl(
        input_paths=[input_dir],
        output_path=output_dir,
        allow_overwrite=True,
        chosen_column="chosen_text",
        rejected_column="rejected_text",
        chosen_reasoning_column="chosen_reason",
        rejected_reasoning_column="rejected_reason",
    )

    with (output_dir / "train.jsonl").open() as handle:
        payload = json.loads(handle.readline())
    assert payload["chosen"][0]["reasoning_content"] == "override chosen"
    assert payload["rejected"][0]["reasoning_content"] == ""


def test_convert_to_jsonl_skips_empty_records_and_prints_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_tokenizer(monkeypatch)
    records = [
        {
            "instruction": "Instruction 1",
            "chosen": {"response": "Chosen 1"},
            "rejected": {"response": "Rejected 1"},
            "chosen_reasoning": "Reasoning 1",
            "rejected_reasoning": "Reasoning 1b",
        },
        {
            "instruction": "",
            "chosen": {"response": "Chosen 2"},
            "rejected": {"response": "Rejected 2"},
            "chosen_reasoning": "Reasoning 2",
            "rejected_reasoning": "Reasoning 2b",
        },
        {
            "instruction": "Instruction 3",
            "chosen": {"response": ""},
            "rejected": {"response": "Rejected 3"},
            "chosen_reasoning": "Reasoning 3",
            "rejected_reasoning": "Reasoning 3b",
        },
        {
            "instruction": "Instruction 4",
            "chosen": {"response": "Chosen 4"},
            "rejected": {"response": "Rejected 4"},
            "chosen_reasoning": "",
            "rejected_reasoning": "Reasoning 4b",
        },
    ]
    dataset = Dataset.from_list(records)
    input_dir = tmp_path / "input"
    dataset.save_to_disk(str(input_dir))
    output_dir = tmp_path / "output"

    to_jsonl.convert_to_jsonl(
        input_paths=[input_dir],
        output_path=output_dir,
        allow_overwrite=True,
        chosen_reasoning_column="chosen_reasoning",
        rejected_reasoning_column="rejected_reasoning",
    )

    with (output_dir / "train.jsonl").open() as handle:
        lines = [json.loads(line) for line in handle]

    assert lines == [
        {
            "prompt": [{"role": "user", "content": "Instruction 1"}],
            "chosen": [
                {
                    "role": "assistant",
                    "content": "Chosen 1",
                    "reasoning_content": "Reasoning 1",
                }
            ],
            "rejected": [
                {
                    "role": "assistant",
                    "content": "Rejected 1",
                    "reasoning_content": "Reasoning 1b",
                }
            ],
            "prompt_source": "",
        }
    ]

    captured = capsys.readouterr().out.splitlines()
    assert "Loaded records: 4" in captured
    assert "Empty instruction: 1" in captured
    assert "Empty chosen response: 1" in captured
    assert "Empty rejected response: 0" in captured
    assert "Empty chosen reasoning_content: 1" in captured
    assert "Empty rejected reasoning_content: 0" in captured
    total_tokens = (
        len("Instruction 1")
        + len("Chosen 1")
        + len("Rejected 1")
        + len("Reasoning 1")
        + len("Reasoning 1b")
    )
    assert f"Total tokens: {total_tokens}" in captured


def test_convert_to_jsonl_uses_tqdm_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_tokenizer(monkeypatch)
    progress_calls: list[dict[str, Any]] = []

    class DummyTqdm:
        def __init__(self, iterable: Any, **kwargs: Any) -> None:
            self._iterable = iterable
            progress_calls.append(kwargs)

        def __iter__(self) -> Any:
            return iter(self._iterable)

        def close(self) -> None:
            return None

    monkeypatch.setattr(to_jsonl, "tqdm", DummyTqdm)

    records = [
        {
            "instruction": "Progress check",
            "chosen": {"response": "ok"},
            "rejected": {"response": "ng"},
            "chosen_reasoning": "Reasoning for progress check",
        }
    ]
    dataset = Dataset.from_list(records)
    input_dir = tmp_path / "tqdm_input"
    dataset.save_to_disk(str(input_dir))
    output_dir = tmp_path / "tqdm_output"

    to_jsonl.convert_to_jsonl(
        input_paths=[input_dir],
        output_path=output_dir,
        allow_overwrite=True,
    )

    assert progress_calls == [{"total": 1, "desc": "Converting train", "unit": "record"}]


def test_convert_to_jsonl_applies_sampling_ratio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_tokenizer(monkeypatch)
    random_values = [0.9, 0.1]

    def _next_random() -> float:
        return random_values.pop(0)

    monkeypatch.setattr(random, "random", _next_random)

    records = [
        {
            "instruction": "First",
            "chosen": {"response": "Keep?"},
            "rejected": {"response": "No"},
            "chosen_reasoning": "Reasoning for first",
        },
        {
            "instruction": "Second",
            "chosen": {"response": "Keep"},
            "rejected": {"response": "Nope"},
            "chosen_reasoning": "Reasoning for second",
        },
    ]
    dataset = Dataset.from_list(records)
    input_dir = tmp_path / "sampling_input"
    dataset.save_to_disk(str(input_dir))
    output_dir = tmp_path / "sampling_output"

    to_jsonl.convert_to_jsonl(
        input_paths=[input_dir],
        output_path=output_dir,
        allow_overwrite=True,
        sampling_ratio=0.5,
    )

    with (output_dir / "train.jsonl").open() as handle:
        lines = [json.loads(line) for line in handle]

    assert [line["prompt"][0]["content"] for line in lines] == ["Second"]


def test_convert_to_jsonl_reports_token_distribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_tokenizer(monkeypatch)
    captured: dict[str, Any] = {}

    def _capture_plot(
        output_path: Path,
        chosen_token_counts: list[int],
        rejected_token_counts: list[int],
        *,
        allow_overwrite: bool,
    ) -> None:
        captured["output_path"] = output_path
        captured["chosen"] = chosen_token_counts
        captured["rejected"] = rejected_token_counts
        captured["allow_overwrite"] = allow_overwrite

    monkeypatch.setattr(to_jsonl, "_write_token_distribution_plot", _capture_plot)

    records = [
        {
            "instruction": "abc",
            "chosen": {"response": "de"},
            "rejected": {"response": "ij"},
            "chosen_reasoning": "fgh",
            "rejected_reasoning": "k",
        }
    ]
    dataset = Dataset.from_list(records)
    input_dir = tmp_path / "distribution_input"
    dataset.save_to_disk(str(input_dir))
    output_dir = tmp_path / "distribution_output"

    to_jsonl.convert_to_jsonl(
        input_paths=[input_dir],
        output_path=output_dir,
        allow_overwrite=True,
        chosen_reasoning_column="chosen_reasoning",
        rejected_reasoning_column="rejected_reasoning",
    )

    assert captured["output_path"] == output_dir / "token_count_distribution.html"
    assert captured["chosen"] == [8]
    assert captured["rejected"] == [6]
    assert captured["allow_overwrite"] is True


def test_convert_to_jsonl_prints_token_distribution_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_tokenizer(monkeypatch)
    records = [
        {
            "instruction": "a" * 500,
            "chosen": {"response": "b" * 700},
            "rejected": {"response": "c" * 900},
            "chosen_reasoning": "r" * 10,
        },
        {
            "instruction": "a" * 1500,
            "chosen": {"response": "b" * 1000},
            "rejected": {"response": "c" * 500},
            "chosen_reasoning": "r" * 10,
        },
    ]
    dataset = Dataset.from_list(records)
    input_dir = tmp_path / "table_input"
    dataset.save_to_disk(str(input_dir))
    output_dir = tmp_path / "table_output"

    to_jsonl.convert_to_jsonl(
        input_paths=[input_dir],
        output_path=output_dir,
        allow_overwrite=True,
    )

    captured = capsys.readouterr().out
    assert "Token count distribution table:" in captured
    assert "1-1k" in captured
    assert "1k-2k" in captured
    assert "chosen_count" in captured
    assert "rejected_count" in captured


def test_convert_to_jsonl_skips_tokenize_option(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _fail_tokenizer(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Tokenizer should not be initialized when skip_tokenize is set.")

    class DummyAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> None:
            _fail_tokenizer(*args, **kwargs)

    monkeypatch.setattr(to_jsonl, "AutoTokenizer", DummyAutoTokenizer)

    records = [
        {
            "instruction": "Skip tokens",
            "chosen": {"response": "Ok"},
            "rejected": {"response": "No"},
        }
    ]
    dataset = Dataset.from_list(records)
    input_dir = tmp_path / "skip_input"
    dataset.save_to_disk(str(input_dir))
    output_dir = tmp_path / "skip_output"

    to_jsonl.convert_to_jsonl(
        input_paths=[input_dir],
        output_path=output_dir,
        allow_overwrite=True,
        skip_tokenize=True,
    )

    output_file = output_dir / "train.jsonl"
    assert output_file.exists()
    captured = capsys.readouterr().out
    assert "Total tokens:" not in captured
    assert "Token count distribution table:" not in captured
    assert "Token distribution plot:" not in captured


def test_convert_to_jsonl_handles_none_chosen_without_crashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_tokenizer(monkeypatch)
    records = [
        {
            "instruction": "Skip because chosen is None",
            "chosen": None,
            "rejected": {"response": "No"},
            "chosen_reasoning": "has reasoning",
        },
        {
            "instruction": "Keep",
            "chosen": {"response": "Yes"},
            "rejected": {"response": "No"},
            "chosen_reasoning": "reasoning",
        },
    ]
    dataset = Dataset.from_list(records)
    input_dir = tmp_path / "none_chosen_input"
    dataset.save_to_disk(str(input_dir))
    output_dir = tmp_path / "none_chosen_output"

    to_jsonl.convert_to_jsonl(
        input_paths=[input_dir],
        output_path=output_dir,
        allow_overwrite=True,
        chosen_reasoning_column="chosen_reasoning",
    )

    with (output_dir / "train.jsonl").open() as handle:
        lines = [json.loads(line) for line in handle]

    assert [line["prompt"][0]["content"] for line in lines] == ["Keep"]
    captured = capsys.readouterr().out.splitlines()
    assert "Empty chosen response: 1" in captured
