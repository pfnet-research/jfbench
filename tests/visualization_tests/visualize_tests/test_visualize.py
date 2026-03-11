from pathlib import Path
from types import SimpleNamespace
from typing import Callable
import warnings

import pandas as pd
import pytest

from jfbench.visualization import visualize


def _sample_dataframe() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    constraint_pairs = [
        (("Constraint-1", "Constraint-2"), ("Group-1", "Group-2")),
    ]
    for prompt_index in range(2):
        for constraints, groups in constraint_pairs:
            constraint_items = tuple(zip(constraints, groups))
            for model_label in ("A", "B"):
                model_name = f"model-{model_label.lower()}"
                base_pass = prompt_index % 2 == 0
                results = {
                    constraints[0]: True,
                    constraints[1]: base_pass if model_label == "A" else not base_pass,
                }
                prompt_source = "ifbench"
                rows.append(
                    {
                        "model": model_name,
                        "model_short": model_label,
                        "model_label": model_label,
                        "prompt_index": prompt_index,
                        "constraint_name": constraints,
                        "constraint_groups": list(groups),
                        "constraint_items": constraint_items,
                        "is_pass": all(results.values()),
                        "results": results,
                        "n_constraints": len(constraints),
                        "prompt_source": prompt_source,
                    }
                )
                for idx, (constraint, group) in enumerate(zip(constraints, groups)):
                    single_pass = (model_label == "A" and idx == 0) or base_pass
                    rows.append(
                        {
                            "model": model_name,
                            "model_short": model_label,
                            "model_label": model_label,
                            "prompt_index": prompt_index + 10 + idx,
                            "constraint_name": (constraint,),
                            "constraint_groups": [group],
                            "constraint_items": ((constraint, group),),
                            "is_pass": single_pass,
                            "results": {constraint: single_pass},
                            "n_constraints": 1,
                            "prompt_source": prompt_source,
                        }
                    )
    return pd.DataFrame(rows)


def test_main_does_not_write_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    df = _sample_dataframe()
    args = SimpleNamespace(
        input_dir=tmp_path,
        output_dir=tmp_path,
        drop_incomplete=False,
        n_constraints=[2],
        prompt_sources=None,
        models=None,
        constraint_sets=None,
        model_label_map=None,
    )

    monkeypatch.setattr(visualize, "parse_args", lambda: args)
    monkeypatch.setattr(visualize, "load_results", lambda *_args, **_kwargs: df)

    visualize.main()

    expected_outputs = [
        tmp_path / "overview" / "overview_model_pass_rates.png",
        tmp_path / "constraints" / "constraints_heatmap.png",
        tmp_path / "constraints" / "constraints_group_heatmap.png",
        tmp_path / "constraints" / "constraints_group_pass_rates.png",
        tmp_path / "constraints" / "constraints_group_bubble.png",
        tmp_path / "models" / "models_pairwise_win_rate.png",
        tmp_path / "models" / "models_cohen_kappa.png",
        tmp_path / "overview" / "overview_constraints_pass_rate.png",
    ]

    for path in expected_outputs:
        assert path.exists()

    csv_files = list(tmp_path.rglob("*.csv"))
    assert csv_files == []


def test_main_filters_prompt_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    df = _sample_dataframe()
    args = SimpleNamespace(
        input_dir=tmp_path,
        output_dir=tmp_path,
        drop_incomplete=False,
        n_constraints=[2],
        prompt_sources=["ifbench"],
        models=None,
        constraint_sets=None,
        model_label_map=None,
    )
    monkeypatch.setattr(visualize, "parse_args", lambda: args)
    monkeypatch.setattr(visualize, "load_results", lambda *_args, **_kwargs: df)

    captured_frames: dict[str, pd.DataFrame] = {}

    def _capture(name: str) -> Callable[[pd.DataFrame, object], None]:
        def _inner(data: pd.DataFrame, *_args: object, **_kwargs: object) -> None:
            captured_frames[name] = data.copy()

        return _inner

    def _capture_constraints(
        data: pd.DataFrame,
        *_args: object,
    ) -> None:
        captured_frames["constraints"] = data.copy()

    monkeypatch.setattr(visualize, "generate_overview_charts", _capture("overview"))
    monkeypatch.setattr(visualize, "generate_constraint_charts", _capture_constraints)
    monkeypatch.setattr(visualize, "generate_model_comparison_outputs", _capture("models"))

    visualize.main()

    assert captured_frames
    for frame in captured_frames.values():
        assert not frame.empty
        assert set(frame["prompt_source"]) == {"ifbench"}


def test_main_passes_model_label_map(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    df = _sample_dataframe()
    args = SimpleNamespace(
        input_dir=tmp_path,
        output_dir=tmp_path,
        drop_incomplete=False,
        n_constraints=[2],
        prompt_sources=["ifbench"],
        models=None,
        constraint_sets=None,
        model_label_map='{"model-a": "Custom A"}',
    )

    captured_map: dict[str, object] = {}
    monkeypatch.setattr(visualize, "parse_args", lambda: args)
    monkeypatch.setattr(
        visualize,
        "load_results",
        lambda _paths, model_label_map=None: (
            captured_map.update({"value": model_label_map}) or df
        ),
    )
    monkeypatch.setattr(visualize, "generate_overview_charts", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(visualize, "generate_constraint_charts", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        visualize, "generate_model_comparison_outputs", lambda *_args, **_kwargs: None
    )

    visualize.main()

    assert captured_map["value"] == {"model-a": "Custom A"}


def test_main_applies_model_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    df = _sample_dataframe()
    args = SimpleNamespace(
        input_dir=tmp_path,
        output_dir=tmp_path,
        drop_incomplete=False,
        n_constraints=[2],
        prompt_sources=None,
        models=["model-b", "model-a"],
        constraint_sets=None,
        model_label_map=None,
    )

    captured: dict[str, dict[str, object]] = {}

    def _capture(name: str) -> Callable[[pd.DataFrame, object], None]:
        def _inner(data: pd.DataFrame, *_args: object, **_kwargs: object) -> None:
            captured[name] = {
                "categories": list(data["model_label"].cat.categories),
                "dtype": data["model_label"].dtype,
            }

        return _inner

    monkeypatch.setattr(visualize, "parse_args", lambda: args)
    monkeypatch.setattr(visualize, "load_results", lambda *_args, **_kwargs: df)
    monkeypatch.setattr(visualize, "generate_overview_charts", _capture("overview"))
    monkeypatch.setattr(visualize, "generate_constraint_charts", _capture("constraints"))
    monkeypatch.setattr(visualize, "generate_model_comparison_outputs", _capture("models"))

    visualize.main()

    expected_dtype = pd.CategoricalDtype(categories=["B", "A"], ordered=True)
    assert captured
    for entry in captured.values():
        assert entry["categories"] == ["B", "A"]
        assert entry["dtype"] == expected_dtype


def test_filter_common_prompt_entries_drops_incomplete_combinations() -> None:
    df = pd.DataFrame(
        [
            {
                "prompt_source": "shared",
                "n_constraints": 2,
                "prompt_index": 0,
                "constraint_name": ("A", "B"),
                "model_short": "Model-A",
            },
            {
                "prompt_source": "shared",
                "n_constraints": 2,
                "prompt_index": 0,
                "constraint_name": ("A", "B"),
                "model_short": "Model-B",
            },
            {
                "prompt_source": "shared",
                "n_constraints": 2,
                "prompt_index": 1,
                "constraint_name": ("A", "C"),
                "model_short": "Model-A",
            },
            {
                "prompt_source": "exclusive",
                "n_constraints": 1,
                "prompt_index": 5,
                "constraint_name": ("Solo",),
                "model_short": "Model-C",
            },
        ]
    )

    with pytest.warns(UserWarning):
        filtered = visualize.filter_common_prompt_entries(df)

    assert set(filtered["prompt_index"]) == {0, 5}
    shared_rows = filtered[filtered["prompt_source"] == "shared"]
    assert set(shared_rows["model_short"]) == {"Model-A", "Model-B"}


def test_collect_input_files_filters_by_prompt_and_constraints(tmp_path: Path) -> None:
    matching_a = tmp_path / "ifbench-2-alpha.jsonl"
    matching_a.write_text("{}", encoding="utf-8")
    matching_b = tmp_path / "ifbench-2-beta.jsonl"
    matching_b.write_text("{}", encoding="utf-8")
    (tmp_path / "ifbench-4-gamma.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")

    paths = visualize._collect_input_files(  # noqa: SLF001
        tmp_path,
        ["ifbench"],
        [2],
        None,
        None,
    )

    assert [path.name for path in paths] == [matching_a.name, matching_b.name]


def test_collect_input_files_filters_by_models(tmp_path: Path) -> None:
    matching = tmp_path / "ifbench-2-alpha.jsonl"
    matching.write_text("{}", encoding="utf-8")
    (tmp_path / "ifbench-2-beta.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "ifbench-2-gamma.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "ifbench-2-alpha.tmp").write_text("{}", encoding="utf-8")

    paths = visualize._collect_input_files(  # noqa: SLF001
        tmp_path,
        ["ifbench"],
        [2],
        ["alpha"],
        None,
    )

    assert [path.name for path in paths] == [matching.name]


def test_collect_input_files_accepts_multiple_constraints(tmp_path: Path) -> None:
    match_one = tmp_path / "ifbench-2-alpha.jsonl"
    match_one.write_text("{}", encoding="utf-8")
    match_two = tmp_path / "ifbench-3-beta.jsonl"
    match_two.write_text("{}", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignore", encoding="utf-8")

    paths = visualize._collect_input_files(  # noqa: SLF001
        tmp_path,
        ["ifbench"],
        [2, 3],
        None,
        None,
    )

    assert [path.name for path in paths] == [
        match_one.name,
        match_two.name,
    ]


def test_main_aggregates_multiple_constraints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    df = _sample_dataframe()
    args = SimpleNamespace(
        input_dir=tmp_path,
        output_dir=tmp_path,
        drop_incomplete=False,
        n_constraints=["1", "2"],
        prompt_sources=["ifbench"],
        models=None,
        constraint_sets=None,
        model_label_map=None,
    )
    monkeypatch.setattr(visualize, "parse_args", lambda: args)
    monkeypatch.setattr(visualize, "load_results", lambda *_args, **_kwargs: df)

    captured_frames: dict[str, pd.DataFrame] = {}

    def _capture(name: str) -> Callable[[pd.DataFrame, object], None]:
        def _inner(data: pd.DataFrame, *_args: object, **_kwargs: object) -> None:
            captured_frames[name] = data.copy()

        return _inner

    monkeypatch.setattr(visualize, "generate_overview_charts", _capture("overview"))
    monkeypatch.setattr(visualize, "generate_constraint_charts", _capture("constraints"))
    monkeypatch.setattr(visualize, "generate_model_comparison_outputs", _capture("models"))

    visualize.main()

    assert captured_frames
    constraints_frame = captured_frames["constraints"]
    assert set(constraints_frame["n_constraints"]) == {1, 2}
    assert set(constraints_frame["prompt_source"]) == {"ifbench"}


def test_log_constraint_stats_suppresses_pandas_warning() -> None:
    df = pd.DataFrame(
        {
            "model_label": pd.Series(
                ["A", "B"],
                dtype=pd.CategoricalDtype(categories=["A", "B"], ordered=True),
            ),
            "n_constraints": pd.Series(
                [1, 2],
                dtype=pd.CategoricalDtype(categories=[1, 2], ordered=True),
            ),
        }
    )

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        visualize._log_constraint_stats(df)  # noqa: SLF001

    assert not any(issubclass(item.category, FutureWarning) for item in recorded)


def test_filter_common_prompt_entries_warns_on_incomplete_counts() -> None:
    df = pd.DataFrame(
        [
            {
                "prompt_source": "shared",
                "n_constraints": 1,
                "prompt_index": 0,
                "constraint_name": ("A",),
                "model_short": "Model-A",
            },
            {
                "prompt_source": "shared",
                "n_constraints": 1,
                "prompt_index": 1,
                "constraint_name": ("A",),
                "model_short": "Model-A",
            },
            {
                "prompt_source": "shared",
                "n_constraints": 1,
                "prompt_index": 0,
                "constraint_name": ("A",),
                "model_short": "Model-B",
            },
        ]
    )

    with pytest.warns(UserWarning) as record:
        filtered = visualize.filter_common_prompt_entries(df)

    assert not filtered.empty
    assert set(filtered["prompt_index"]) == {0}
    messages = [str(item.message) for item in record]
    assert any("expected 2 records" in message for message in messages)
