import json
from pathlib import Path
from typing import Any

from jfbench.benchmark.analyze import _format_summary
from jfbench.benchmark.analyze import filter_records_by_constraint
from jfbench.benchmark.analyze import format_generated_responses
from jfbench.benchmark.analyze import GroupKey
from jfbench.benchmark.analyze import GroupStats
from jfbench.benchmark.analyze import load_records
from jfbench.benchmark.analyze import summarize_records


def test_summarize_records_groups_by_prompt_source_and_counts() -> None:
    records: list[dict[str, Any]] = [
        {
            "prompt_source": "ifbench",
            "n_constraints": 1,
            "model_short": "Alpha",
            "response": "draft",
            "reasoning_content": "",
            "results": None,
        },
        {
            "prompt_source": "ifbench",
            "n_constraints": 1,
            "model_short": "Alpha",
            "response": "final",
            "reasoning_content": "  ",
            "results": {"ConstraintA": True},
        },
        {
            "prompt_source": "ifbench",
            "n_constraints": 1,
            "model_short": "Alpha",
            "response": "retry",
            "reasoning_content": "analysis",
            "results": {"ConstraintA": True, "ConstraintB": False},
        },
        {
            "prompt_source": None,
            "n_constraints": None,
            "model": "BetaModel",
            "response": "",
            "reasoning_content": None,
            "results": {"ConstraintX": True},
        },
    ]

    summary = summarize_records(records)

    alpha_key = GroupKey(prompt_source="ifbench", n_constraints=1, model_short="Alpha")
    assert alpha_key in summary
    alpha_stats = summary[alpha_key]
    assert alpha_stats.total_count == 3
    assert alpha_stats.generated_count == 3
    assert alpha_stats.evaluated_count == 2
    assert alpha_stats.success_count == 1
    assert alpha_stats.failure_count == 1
    assert alpha_stats.pending_count == 1
    assert alpha_stats.empty_reasoning_count == 2

    beta_key = GroupKey(prompt_source="unknown", n_constraints=-1, model_short="BetaModel")
    assert beta_key in summary
    beta_stats = summary[beta_key]
    assert beta_stats.total_count == 1
    assert beta_stats.generated_count == 1
    assert beta_stats.evaluated_count == 1
    assert beta_stats.success_count == 1
    assert beta_stats.pending_count == 0
    assert beta_stats.empty_reasoning_count == 0


def test_format_summary_shows_n_constraints_label() -> None:
    summary = {
        GroupKey(prompt_source="ifbench", n_constraints=2, model_short="Alpha"): GroupStats(
            total_count=1,
            generated_count=1,
            evaluated_count=1,
            success_count=1,
        )
    }

    formatted = list(_format_summary(summary))

    assert formatted[0].strip().startswith("prompt_source")
    assert "n_constraints" in formatted[0]
    assert "empty_reasoning" in formatted[0]


def test_filter_records_by_constraint_matches_types_and_meta() -> None:
    records = [
        {
            "constraint_types": ["HeadingStructureConstraint"],
            "constraints": [{"name": "HeadingStructureConstraint"}],
        },
        {
            "constraint_types": ["OtherConstraint"],
            "constraints": [{"name": "HeadingStructureConstraint"}],
        },
        {
            "constraint_types": ["otherconstraint"],
            "constraints": [{"name": "Different"}],
        },
    ]

    filtered = filter_records_by_constraint(records, "headingstructureconstraint")

    assert len(filtered) == 2
    assert all(
        any(
            name.lower() == "headingstructureconstraint"
            for name in entry.get("constraint_types", ())
        )
        or any(
            constraint.get("name", "").lower() == "headingstructureconstraint"
            for constraint in entry.get("constraints", ())
            if isinstance(constraint, dict)
        )
        for entry in filtered
    )


def test_format_generated_responses_shows_preview() -> None:
    records: list[dict[str, Any]] = [
        {
            "prompt_source": "ifbench",
            "n_constraints": 2,
            "model_short": "Alpha",
            "data_id": "sample-1",
            "prompt": " first line of prompt ",
            "response": " First line.  Second line.",
            "reasoning_content": " Second line. ",
            "results": {"ConstraintA": True},
        },
        {
            "prompt_source": "ifbench",
            "n_constraints": 1,
            "model_short": "Beta",
            "data_id": "sample-2",
            "prompt": None,
            "response": "",
            "reasoning_content": None,
            "results": None,
        },
    ]

    formatted = list(format_generated_responses(records))

    assert formatted[0].startswith("[1] ifbench | n=2 | Alpha | sample-1 | passed")
    assert formatted[1] == "Prompt:"
    assert formatted[2].strip() == "first line of prompt"
    assert formatted[3] == "Response:"
    assert formatted[4].strip() == "First line.  Second line."
    assert formatted[5] == "Reasoning:"
    assert formatted[6].strip() == "Second line."
    assert formatted[8].startswith("[2] ifbench | n=1 | Beta | sample-2 | not evaluated")
    assert formatted[9] == "Prompt:"
    assert formatted[10].strip() == "<no prompt>"
    assert formatted[11] == "Response:"
    assert formatted[12].strip() == "<empty response>"
    assert formatted[13] == "Reasoning:"
    assert formatted[14].strip() == "<no reasoning>"


def test_format_generated_responses_marks_failed_when_any_constraint_false() -> None:
    records: list[dict[str, Any]] = [
        {
            "prompt_source": "ifbench",
            "n_constraints": 1,
            "model_short": "Alpha",
            "data_id": "sample-3",
            "prompt": "prompt",
            "response": "response",
            "results": {"ConstraintA": True, "ConstraintB": False},
        },
    ]

    formatted = list(format_generated_responses(records))

    assert formatted[0].endswith("failed")


def test_load_records_reads_all_jsonl_files_in_directory(tmp_path: Path) -> None:
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    first_path = tmp_path / "first.jsonl"
    second_path = nested_dir / "second.jsonl"

    first_records = [{"data_id": "a"}, {"data_id": "b"}]
    second_records = [{"data_id": "c"}]

    first_path.write_text(
        "\n".join(json.dumps(record) for record in first_records) + "\n",
        encoding="utf-8",
    )
    second_path.write_text(
        "\n".join(json.dumps(record) for record in second_records) + "\n",
        encoding="utf-8",
    )

    records = load_records(tmp_path)

    assert {record["data_id"] for record in records} == {"a", "b", "c"}
