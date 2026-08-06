from __future__ import annotations

from jfbench.sft_dataset.analyze import compute_constraint_stats
from jfbench.sft_dataset.analyze import ConstraintStats
from jfbench.sft_dataset.analyze import format_dataset_preview
from jfbench.sft_dataset.analyze import format_generated_records
from jfbench.sft_dataset.analyze import format_stats
from jfbench.sft_dataset.analyze import summarize_dataset_preview


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


def test_compute_constraint_stats_fills_zero_for_all_constraints() -> None:
    rows: list[dict[str, object]] = [
        {
            "prompt_source": "alpha",
            "constraint_count": 1,
            "constraints": [{"group": "length", "name": "SectionsLengthConstraint"}],
        }
    ]

    stats = compute_constraint_stats(rows)

    assert stats.per_constraint_counts[1]["length"]["SectionsLengthConstraint"] == 1
    assert stats.per_constraint_counts[1]["length"]["WordsLengthConstraint"] == 0


def test_compute_constraint_stats_normalizes_data_group_to_lowercase() -> None:
    rows: list[dict[str, object]] = [
        {
            "constraint_count": 1,
            "constraints": [{"group": "LENGTH", "name": "SectionsLengthConstraint"}],
        }
    ]

    stats = compute_constraint_stats(rows)

    assert stats.per_constraint_counts[1]["length"]["SectionsLengthConstraint"] == 1
    assert "LENGTH" not in stats.per_constraint_counts[1]


def test_compute_constraint_stats_normalizes_camel_group_to_snake_case() -> None:
    rows: list[dict[str, object]] = [
        {
            "constraint_count": 1,
            "constraints": [{"group": "MetaOutput", "name": "SectionsLengthConstraint"}],
        }
    ]

    stats = compute_constraint_stats(rows)

    assert stats.per_constraint_counts[1]["meta_output"]["SectionsLengthConstraint"] == 1
    assert "MetaOutput" not in stats.per_constraint_counts[1]


def test_format_generated_records_includes_prompt_response_reasoning() -> None:
    rows = [
        {
            "prompt_source": "alpha",
            "constraint_count": 2,
            "constraint_types": ["length", "style"],
            "prompt": "Prompt text",
            "response": "Response text",
            "reasoning_content": "Reasoning text",
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
    assert "Response:" in lines
    assert "Response text" in lines
    assert "Reasoning:" in lines
    assert "Reasoning text" in lines


def test_summarize_dataset_preview_limits_columns_and_rows() -> None:
    rows = [
        {"a": 1, "b": 2, "c": 3, "d": 4},
        {"a": 5, "b": 6, "c": 7, "d": 8},
        {"a": 9, "b": 10, "c": 11, "d": 12},
        {"a": 13, "b": 14, "c": 15, "d": 16},
    ]

    columns, sampled_rows = summarize_dataset_preview(rows)

    assert columns == ["a", "b", "c", "d"]
    assert len(sampled_rows) == 3
    assert sampled_rows[0]["a"] == 1
    assert sampled_rows[2]["c"] == 11


def test_format_dataset_preview_includes_columns_and_values() -> None:
    columns = ["a", "b", "c"]
    rows = [{"a": "x", "b": "", "c": None}]

    formatted = format_dataset_preview(columns, rows)

    assert "Dataset columns:" in formatted
    assert "  - a" in formatted
    assert "Sample values (up to 3 rows):" in formatted
    assert "    a: x" in formatted
    assert "    b: <empty>" in formatted
    assert "    c: <missing>" in formatted
