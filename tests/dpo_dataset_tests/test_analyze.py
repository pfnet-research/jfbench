from __future__ import annotations

import argparse
import asyncio
import math
from typing import TYPE_CHECKING

from datasets import Dataset


if TYPE_CHECKING:
    from pathlib import Path

    from _pytest.monkeypatch import MonkeyPatch

from jfbench.dpo_dataset import analyze as analyze_module
from jfbench.dpo_dataset.analyze import _build_combined_answer_via_chat_template
from jfbench.dpo_dataset.analyze import _completions_prompt_logprobs_chunked
from jfbench.dpo_dataset.analyze import _extract_token_lengths
from jfbench.dpo_dataset.analyze import compute_constraint_stats
from jfbench.dpo_dataset.analyze import compute_numeric_distribution
from jfbench.dpo_dataset.analyze import ConstraintStats
from jfbench.dpo_dataset.analyze import format_generated_records
from jfbench.dpo_dataset.analyze import format_numeric_distribution
from jfbench.dpo_dataset.analyze import format_stats
from jfbench.dpo_dataset.analyze import load_datasets
from jfbench.dpo_dataset.analyze import load_jsonl
from jfbench.dpo_dataset.analyze import run_extra_analyses
from jfbench.dpo_dataset.analyze import write_numeric_distribution_histogram


def test_compute_constraint_stats_counts_prompt_sources() -> None:
    rows: list[dict[str, object]] = [
        {"prompt_source": "alpha", "constraint_count": 1, "response": "first"},
        {"prompt_source": "beta", "constraint_count": 2},
        {"prompt_source": "alpha", "constraint_count": 2},
        {"constraint_count": 1},
    ]

    stats = compute_constraint_stats(rows)

    assert stats.total_records == 4
    assert stats.counts_by_constraint_total == {1: 2, 2: 2}
    assert stats.counts_by_prompt_and_constraint_count == {
        "alpha": {1: 1, 2: 1},
        "beta": {2: 1},
    }


def test_format_stats_includes_prompt_source_counts() -> None:
    stats = ConstraintStats(
        total_records=2,
        counts_by_constraint_total={1: 1, 2: 1},
        counts_by_prompt_and_constraint_count={"alpha": {1: 1}, "beta": {2: 1}},
        per_constraint_counts={},
    )

    formatted = format_stats(stats)

    assert "Total data per prompt source and constraint count:" in formatted
    assert "  alpha:" in formatted
    assert "    1: 1" in formatted
    assert "  beta:" in formatted
    assert "    2: 1" in formatted


def test_format_generated_records_includes_prompt_response_reasoning() -> None:
    rows = [
        {
            "prompt_source": "alpha",
            "constraint_count": 2,
            "constraint_types": ["length", "style"],
            "prompt": "Prompt text",
            "chosen": {"response": "Response text", "reasoning_content": "Reasoning text"},
            "rejected": {"response": "Rejected text", "reasoning_content": "Rejected reasoning"},
        }
    ]

    lines = list(format_generated_records(rows))

    assert "Prompt source:" in lines
    assert "alpha" in lines
    assert "Constraint count:" in lines
    assert "2" in lines
    assert "Constraint types:" in lines
    assert "['length', 'style']" in lines
    assert "Prompt:" in lines
    assert "Prompt text" in lines
    assert "Chosen response:" in lines
    assert "Response text" in lines
    assert "Chosen reasoning:" in lines
    assert "Reasoning text" in lines
    assert "Rejected response:" in lines
    assert "Rejected text" in lines
    assert "Rejected reasoning:" in lines
    assert "Rejected reasoning" in lines


def test_load_jsonl_reads_hf_dataset(tmp_path: Path) -> None:
    path = tmp_path / "dataset"
    rows = [
        {"prompt_source": "a", "constraint_count": 1, "constraint_types": []},
        {"prompt_source": "b", "constraint_count": 2, "constraint_types": ["X"]},
    ]
    Dataset.from_list(rows).save_to_disk(str(path))

    loaded = load_jsonl(path)

    assert loaded == rows


def test_load_jsonl_raises_for_missing_directory(tmp_path: Path) -> None:
    path = tmp_path / "missing"

    try:
        load_jsonl(path)
    except FileNotFoundError as exc:
        assert str(path) in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("Expected FileNotFoundError")


def test_load_datasets_merges_multiple_directories(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    Dataset.from_list([{"prompt_source": "a", "constraint_count": 1}]).save_to_disk(str(first))
    Dataset.from_list([{"prompt_source": "b", "constraint_count": 2}]).save_to_disk(str(second))

    records, details = load_datasets([str(first), str(second)])

    assert len(records) == 2
    assert details == [(first, 1), (second, 1)]


def test_compute_numeric_distribution_returns_expected_stats() -> None:
    stats = compute_numeric_distribution([1, 2, 3, 4], bucket_size=2)

    assert stats.count == 4
    assert stats.minimum == 1
    assert stats.maximum == 4
    assert stats.mean == 2.5
    assert stats.p50 == 3
    assert stats.bucket_counts == {"[0, 2)": 1, "[2, 4)": 2, "[4, 6)": 1}


def test_compute_numeric_distribution_ignores_non_finite_values() -> None:
    stats = compute_numeric_distribution([1, math.nan, 2, math.inf, -math.inf], bucket_size=1)

    assert stats.count == 2
    assert stats.minimum == 1
    assert stats.maximum == 2
    assert stats.mean == 1.5
    assert stats.bucket_counts == {"[1, 2)": 1, "[2, 3)": 1}


def test_compute_numeric_distribution_returns_empty_for_only_non_finite_values() -> None:
    stats = compute_numeric_distribution([math.nan, math.inf, -math.inf], bucket_size=1)

    assert stats.count == 0
    assert stats.minimum == 0.0
    assert stats.maximum == 0.0
    assert stats.mean == 0.0
    assert stats.bucket_counts == {}


def test_format_numeric_distribution_handles_empty_data() -> None:
    text = format_numeric_distribution("sample", [], bucket_size=1)

    assert "sample" in text
    assert "(no data)" in text


def test_write_numeric_distribution_histogram_writes_html(tmp_path: Path) -> None:
    output_path = tmp_path / "histogram.html"

    result = write_numeric_distribution_histogram(
        title="sample distribution",
        values=[1.0, 2.0, 3.0, 4.0],
        bucket_size=1.0,
        output_path=output_path,
    )

    assert result == output_path
    assert output_path.exists()
    html = output_path.read_text(encoding="utf-8")
    assert "count: 4" in html
    assert "sample distribution" in html


def test_run_extra_analyses_creates_plots_for_numeric_sections(tmp_path: Path) -> None:
    args = argparse.Namespace(analysis_concurrency=1)
    rows = [
        {
            "chosen_token_length": 10,
            "rejected_token_length": 20,
            "chosen_reasoning_token_length": 30,
            "rejected_reasoning_token_length": 40,
        }
    ]

    sections = asyncio.run(run_extra_analyses(args, rows, plot_output_dir=tmp_path))

    assert "Token length distributions" in sections
    assert len(list(tmp_path.glob("*.html"))) == 2
    html = (tmp_path / "token_length_chosen_vs_rejected_token_lengths.html").read_text(
        encoding="utf-8"
    )
    assert '"name":"chosen"' in html
    assert '"name":"rejected"' in html
    assert "chosen min:" in html
    assert "rejected min:" in html
    assert any(str(tmp_path) in section for section in sections)


def test_run_extra_analyses_creates_difference_plot_for_constraint_pass_count(
    tmp_path: Path,
) -> None:
    args = argparse.Namespace(analysis_concurrency=1)
    rows = [
        {"chosen_all_constraints_pass_count": 3, "rejected_al_constraints_pass_count": 1},
        {"chosen_all_constraints_pass_count": 2, "rejected_al_constraints_pass_count": 2},
    ]

    sections = asyncio.run(run_extra_analyses(args, rows, plot_output_dir=tmp_path))

    assert "Constraint-pass-count distributions" in sections
    output_path = (
        tmp_path
        / "constraint_pass_count_all_constraints_pass_count_difference_chosen_rejected.html"
    )
    assert output_path.exists()
    html = output_path.read_text(encoding="utf-8")
    assert "all-constraints-pass count difference (chosen - rejected)" in html
    assert "count: 2" in html


def test_run_extra_analyses_creates_reward_difference_plots(tmp_path: Path) -> None:
    args = argparse.Namespace(analysis_concurrency=1)
    rows = [
        {
            "chosen_reward": 2.0,
            "rejected_reward": 1.0,
            "chosen_reasoning_reward": 0.5,
            "rejected_reasoning_reward": 0.25,
            "sum_of_chosen_and_chosen_reasoning_reward": 2.5,
            "sum_of_rejected_and_rejected_reasoning_reward": 1.25,
            "concatenated_chosen_and_chosen_reasoning_reward": 2.25,
            "concatenated_rejected_and_rejected_reasoning_reward": 1.0,
        },
        {
            "chosen_reward": 1.5,
            "rejected_reward": 1.0,
            "chosen_reasoning_reward": 0.6,
            "rejected_reasoning_reward": 0.1,
            "sum_of_chosen_and_chosen_reasoning_reward": 2.1,
            "sum_of_rejected_and_rejected_reasoning_reward": 1.1,
            "concatenated_chosen_and_chosen_reasoning_reward": 2.0,
            "concatenated_rejected_and_rejected_reasoning_reward": 1.2,
        },
    ]

    sections = asyncio.run(run_extra_analyses(args, rows, plot_output_dir=tmp_path))

    assert "Reward distributions" in sections
    output_path = tmp_path / "reward_reward_difference_chosen_rejected.html"
    assert output_path.exists()
    html = output_path.read_text(encoding="utf-8")
    assert "reward difference (chosen - rejected)" in html
    assert "count: 2" in html


class _DummyTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text == "":
            return []
        return list(range(len(text.split())))

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> str:
        _ = (tokenize, add_generation_prompt)
        return " | ".join(f"{m['role']}:{m['content']}" for m in messages)


def test_extract_token_lengths_uses_extracted_fields() -> None:
    rows = [
        {
            "chosen": {"response": "a b c"},
            "rejected": {"response": "x y"},
            "chosen_reasoning": "r1 r2 r3 r4",
            "rejected_reasoning": "k1",
        }
    ]
    tokenizer = _DummyTokenizer()

    lengths = _extract_token_lengths(rows, tokenizer=tokenizer)

    assert lengths["chosen"] == [3]
    assert lengths["rejected"] == [2]
    assert lengths["chosen_reasoning"] == [4]
    assert lengths["rejected_reasoning"] == [1]


def test_build_combined_answer_via_chat_template() -> None:
    tokenizer = _DummyTokenizer()

    combined = _build_combined_answer_via_chat_template(
        tokenizer,
        prompt="prompt",
        response="response",
        reasoning="reasoning",
    )

    assert combined == "user:prompt | assistant:response | assistant:reasoning"


class _DummyChunkTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        return [ord(ch) for ch in text]

    def decode(self, ids: list[int], clean_up_tokenization_spaces: bool = False) -> str:
        _ = clean_up_tokenization_spaces
        return "".join(chr(i) for i in ids)


class _DummyCompletions:
    def __init__(self) -> None:
        self.calls = 0

    async def create(self, **kwargs: object) -> object:
        _ = kwargs
        self.calls += 1
        if self.calls == 1:
            raise TimeoutError("simulated timeout")
        return type(
            "DummyResponse",
            (),
            {
                "choices": [
                    type(
                        "DummyChoice",
                        (),
                        {"logprobs": {"content": [{"token": "A", "logprob": -0.5}]}},
                    )()
                ]
            },
        )()


class _DummyClient:
    def __init__(self, completions: _DummyCompletions) -> None:
        self.completions = completions


def test_completions_prompt_logprobs_chunked_retries_on_timeout(
    monkeypatch: MonkeyPatch,
) -> None:
    completions = _DummyCompletions()
    client = _DummyClient(completions)
    tokenizer = _DummyChunkTokenizer()

    warning_messages: list[str] = []
    monkeypatch.setattr(analyze_module.logger, "warning", lambda msg: warning_messages.append(msg))

    result = asyncio.run(
        _completions_prompt_logprobs_chunked(
            client,
            model="dummy-model",
            text="A",
            tokenizer=tokenizer,
            chunk_size=16,
            overlap=0,
        )
    )

    assert completions.calls == 2
    assert result["tokens"] == ["A"]
    assert result["token_logprobs"] == [-0.5]
    assert warning_messages
