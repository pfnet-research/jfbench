import json
from pathlib import Path

import pytest

from jfbench.visualization.data_loader import KEY_COLUMNS
from jfbench.visualization.data_loader import load_results


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_load_results_extracts_constraint_and_deduplicates(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    _write_jsonl(
        path,
        [
            {
                "model": "model-a",
                "model_short": "model-a-short",
                "constraint_groups": ["GroupZ", "GroupX"],
                "prompt_index": 1,
                "constraint_types": ["ConstraintZ", "ConstraintX"],
                "results": {"ConstraintZ": True, "ConstraintX": True},
                "response": "ok",
                "data_id": "id-1",
            },
            {
                "model": "model-a",
                "model_short": "model-a-short",
                "constraint_groups": ["GroupX", "GroupZ"],
                "prompt_index": 1,
                "constraint_types": ["ConstraintX", "ConstraintZ"],
                "results": {"ConstraintX": False, "ConstraintZ": False},
                "response": "ng",
                "data_id": "id-1",
            },
        ],
    )

    df = load_results([path])

    for column in KEY_COLUMNS:
        assert column in df.columns
    assert len(df) == 1
    row = df.iloc[0]
    assert row["constraint_name"] == ("ConstraintX", "ConstraintZ")
    assert row["constraint_groups"] == ["GroupX", "GroupZ"]
    assert row["constraint_items"] == (("ConstraintX", "GroupX"), ("ConstraintZ", "GroupZ"))
    assert bool(row["is_pass"]) is False
    assert row["n_constraints"] == 2
    assert row["model_label"] == "model-a-short"


def test_load_results_handles_multiple_files(tmp_path: Path) -> None:
    path_a = tmp_path / "a.jsonl"
    path_b = tmp_path / "b.jsonl"
    _write_jsonl(
        path_a,
        [
            {
                "model": "model-a",
                "model_short": "model-a-short",
                "constraint_groups": ["GroupY"],
                "prompt_index": 2,
                "constraint_types": ["ConstraintY"],
                "results": {"ConstraintY": True},
            },
        ],
    )
    _write_jsonl(
        path_b,
        [
            {
                "model": "model-b",
                "model_short": "model-b-short",
                "constraint_groups": ["GroupY"],
                "prompt_index": 2,
                "constraint_types": ["ConstraintY"],
                "results": {"ConstraintY": False},
            },
        ],
    )

    df = load_results([path_a, path_b])
    assert set(df["model_label"]) == {"model-a-short", "model-b-short"}
    assert set(df["is_pass"]) == {True, False}


def test_load_results_keeps_distinct_model_short(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    _write_jsonl(
        path,
        [
            {
                "model": "shared-model",
                "model_short": "variant-a",
                "constraint_groups": ["GroupV"],
                "prompt_index": 3,
                "constraint_types": ["ConstraintV"],
                "results": {"ConstraintV": True},
            },
            {
                "model": "shared-model",
                "model_short": "variant-b",
                "constraint_groups": ["GroupV"],
                "prompt_index": 3,
                "constraint_types": ["ConstraintV"],
                "results": {"ConstraintV": False},
            },
        ],
    )

    df = load_results([path])
    assert len(df) == 2
    assert set(df["model_label"]) == {"variant-a", "variant-b"}


def test_model_label_uses_model_short(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    _write_jsonl(
        path,
        [
            {
                "model": "some/long/path-model",
                "model_short": "compact-model",
                "constraint_groups": ["GroupZ"],
                "prompt_index": 0,
                "constraint_types": ["ConstraintZ"],
                "results": {"ConstraintZ": True},
            }
        ],
    )

    df = load_results([path])
    assert df.iloc[0]["model_label"] == "compact-model"


def test_model_label_map_overrides_model_short(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    _write_jsonl(
        path,
        [
            {
                "model": "model-a",
                "model_short": "short-a",
                "constraint_groups": ["GroupZ"],
                "prompt_index": 0,
                "constraint_types": ["ConstraintZ"],
                "results": {"ConstraintZ": True},
            },
            {
                "model": "model-b",
                "model_short": "short-b",
                "constraint_groups": ["GroupZ"],
                "prompt_index": 1,
                "constraint_types": ["ConstraintZ"],
                "results": {"ConstraintZ": False},
            },
        ],
    )

    overrides = {"short-a": "Custom-A"}
    df = load_results([path], overrides)

    short_a_rows = df[df["model_short"] == "short-a"]
    assert short_a_rows.iloc[0]["model_label"] == "Custom-A"

    other_rows = df[df["model_short"] == "short-b"]
    assert other_rows.iloc[0]["model_label"] == "short-b"


def test_load_results_splits_seed_from_model_short(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    _write_jsonl(
        path,
        [
            {
                "model": "model-a",
                "model_short": "short-a (seed 0)",
                "constraint_groups": ["GroupZ"],
                "prompt_index": 0,
                "constraint_types": ["ConstraintZ"],
                "results": {"ConstraintZ": True},
            },
            {
                "model": "model-a",
                "model_short": "short-a (seed 1)",
                "constraint_groups": ["GroupZ"],
                "prompt_index": 1,
                "constraint_types": ["ConstraintZ"],
                "results": {"ConstraintZ": False},
            },
        ],
    )

    df = load_results([path])

    assert set(df["model_label"]) == {"short-a"}
    assert set(df["seed"]) == {"0", "1"}


def test_load_results_raises_on_missing_prompt_index(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    _write_jsonl(
        path,
        [
            {
                "model": "model-a",
                "model_short": "model-a-short",
                "constraint_groups": ["GroupZ"],
                "prompt_index": None,
                "constraint_types": ["ConstraintZ"],
                "results": {"ConstraintZ": True},
            }
        ],
    )

    with pytest.raises(ValueError, match="prompt_index"):
        load_results([path])


def test_load_results_raises_on_missing_results(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    _write_jsonl(
        path,
        [
            {
                "model": "model-a",
                "model_short": "model-a-short",
                "constraint_groups": ["GroupZ"],
                "prompt_index": 0,
                "constraint_types": ["ConstraintZ"],
                "results": None,
            }
        ],
    )

    with pytest.raises(ValueError, match="Evaluation results"):
        load_results([path])


def test_load_results_raises_on_non_numeric_prompt_index(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    _write_jsonl(
        path,
        [
            {
                "model": "model-a",
                "model_short": "model-a-short",
                "constraint_groups": ["GroupZ"],
                "prompt_index": "not-a-number",
                "constraint_types": ["ConstraintZ"],
                "results": {"ConstraintZ": True},
            }
        ],
    )

    with pytest.raises(ValueError, match="numeric"):
        load_results([path])
