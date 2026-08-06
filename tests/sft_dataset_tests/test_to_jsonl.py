from __future__ import annotations

import json
import sys
from typing import Any
from typing import TYPE_CHECKING

from datasets import Dataset

from jfbench.sft_dataset import to_jsonl


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


def test_to_jsonl_builds_chat_messages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_tokenizer(monkeypatch)
    dataset = Dataset.from_dict(
        {
            "prompt": ["こんにちは"],
            "answer": ["やあ"],
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
            "--instruction-column",
            "prompt",
            "--response-column",
            "answer",
        ],
    )

    to_jsonl.main()

    with (output_dir / "train.jsonl").open() as f:
        rows = [json.loads(line) for line in f]

    assert rows == [
        {
            "chat": [
                {"role": "user", "content": "こんにちは"},
                {"role": "assistant", "content": "やあ"},
            ]
        }
    ]


def test_to_jsonl_appends_reasoning_content_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_tokenizer(monkeypatch)
    dataset = Dataset.from_dict(
        {
            "prompt": ["こんにちは"],
            "answer": ["やあ"],
            "reasoning_content": ["ゆっくり考えました"],
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
            "--instruction-column",
            "prompt",
            "--response-column",
            "answer",
        ],
    )

    to_jsonl.main()

    with (output_dir / "train.jsonl").open() as f:
        rows = [json.loads(line) for line in f]

    assert rows == [
        {
            "chat": [
                {"role": "user", "content": "こんにちは"},
                {
                    "role": "assistant",
                    "content": "やあ",
                    "reasoning_content": "ゆっくり考えました",
                },
            ]
        }
    ]


def test_to_jsonl_combines_multiple_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_tokenizer(monkeypatch)
    dataset_one = Dataset.from_dict({"prompt": ["こんにちは"], "answer": ["やあ"]})
    dataset_two = Dataset.from_dict({"prompt": ["元気?"], "answer": ["ばっちり"]})
    input_dir_one = tmp_path / "input_one"
    input_dir_two = tmp_path / "input_two"
    output_dir = tmp_path / "output"
    dataset_one.save_to_disk(str(input_dir_one))
    dataset_two.save_to_disk(str(input_dir_two))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "to_jsonl.py",
            "--input",
            str(input_dir_one),
            str(input_dir_two),
            "--output",
            str(output_dir),
            "--allow-overwrite",
            "--instruction-column",
            "prompt",
            "--response-column",
            "answer",
        ],
    )

    to_jsonl.main()

    with (output_dir / "train.jsonl").open() as f:
        rows = [json.loads(line) for line in f]

    assert rows == [
        {
            "chat": [
                {"role": "user", "content": "こんにちは"},
                {"role": "assistant", "content": "やあ"},
            ]
        },
        {
            "chat": [
                {"role": "user", "content": "元気?"},
                {"role": "assistant", "content": "ばっちり"},
            ]
        },
    ]


def test_to_jsonl_skips_empty_records_and_prints_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_tokenizer(monkeypatch)
    dataset = Dataset.from_dict(
        {
            "prompt": ["こんにちは", "", "こんにちは2", "こんにちは3", "こんにちは4"],
            "answer": ["やあ", "やあ2", "", "やあ3", "やあ4"],
            "reasoning_content": ["考えた", "考えた2", "考えた3", "", "考えた4"],
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
            "--instruction-column",
            "prompt",
            "--response-column",
            "answer",
        ],
    )

    to_jsonl.main()

    with (output_dir / "train.jsonl").open() as f:
        rows = [json.loads(line) for line in f]

    assert rows == [
        {
            "chat": [
                {"role": "user", "content": "こんにちは"},
                {
                    "role": "assistant",
                    "content": "やあ",
                    "reasoning_content": "考えた",
                },
            ]
        },
        {
            "chat": [
                {"role": "user", "content": "こんにちは4"},
                {
                    "role": "assistant",
                    "content": "やあ4",
                    "reasoning_content": "考えた4",
                },
            ]
        },
    ]

    captured = capsys.readouterr().out.splitlines()
    assert "Loaded records: 5" in captured
    assert "Empty instruction: 1" in captured
    assert "Empty response: 1" in captured
    assert "Empty reasoning_content: 1" in captured
    total_tokens = (
        len("こんにちは")
        + len("やあ")
        + len("考えた")
        + len("こんにちは4")
        + len("やあ4")
        + len("考えた4")
    )
    assert f"Total tokens: {total_tokens}" in captured
