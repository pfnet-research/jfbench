from pathlib import Path
from typing import cast

from matplotlib.axes import Axes
import pandas as pd
import pytest

from jfbench.visualization import constraints


def test_with_primary_group_expands_all_groups() -> None:
    data = pd.DataFrame.from_records(
        [
            {
                "model": "model-a",
                "model_label": "A",
                "constraint_groups": ["GroupB", "GroupA"],
                "constraint_name": ("ConstraintY", "ConstraintX"),
                "constraint_items": (("ConstraintX", "GroupA"), ("ConstraintY", "GroupB")),
                "results": {"ConstraintX": True, "ConstraintY": False},
                "is_pass": True,
            }
        ]
    )

    prepared = constraints._with_primary_group(data)  # noqa: SLF001

    assert len(prepared) == 2
    extracted = {
        (row["constraint_group"], row["constraint_name"], row["is_pass"])
        for _, row in prepared.iterrows()
    }
    assert extracted == {
        ("GroupA", "ConstraintX", True),
        ("GroupB", "ConstraintY", False),
    }


def test_plot_constraint_counts_creates_output(tmp_path: Path) -> None:
    data = pd.DataFrame.from_records(
        [
            {
                "model": "model-a",
                "model_label": "A",
                "constraint_groups": ["GroupB", "GroupA"],
                "constraint_name": ("ConstraintY", "ConstraintX"),
                "constraint_items": (("ConstraintX", "GroupA"), ("ConstraintY", "GroupB")),
                "results": {"ConstraintX": True, "ConstraintY": False},
            },
            {
                "model": "model-b",
                "model_label": "B",
                "constraint_groups": ["GroupA"],
                "constraint_name": ("ConstraintX",),
                "constraint_items": (("ConstraintX", "GroupA"),),
                "results": {"ConstraintX": False},
            },
        ]
    )

    prepared = constraints._with_primary_group(data)  # noqa: SLF001
    output_path = tmp_path / "counts.png"

    constraints.plot_constraint_counts(prepared, output_path)

    assert output_path.exists()


def test_plot_constraint_group_bar_places_legend_outside(
    tmp_path: Path,
    monkeypatch: "pytest.MonkeyPatch",
) -> None:
    data = pd.DataFrame.from_records(
        [
            {"constraint_group": "GroupA", "model_label": "A", "is_pass": True},
            {"constraint_group": "GroupA", "model_label": "B", "is_pass": False},
        ]
    )
    output_path = tmp_path / "bar.png"
    legend_kwargs: dict[str, object] = {}

    original = Axes.legend

    def _capture_legend(self: Axes, *args: object, **kwargs: object) -> object:
        legend_kwargs.update(kwargs)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Axes, "legend", _capture_legend)

    constraints.plot_constraint_group_bar(data, output_path)

    anchor = legend_kwargs.get("bbox_to_anchor")
    assert isinstance(anchor, tuple)
    anchor_values = cast("tuple[float, float]", anchor)
    assert anchor_values[0] > 1


def test_with_primary_group_handles_missing_constraint_items() -> None:
    data = pd.DataFrame.from_records(
        [
            {
                "model": "model-a",
                "model_label": "A",
                "constraint_groups": ["GroupB", "GroupA"],
                "constraint_name": ("ConstraintY", "ConstraintX"),
                "results": {"ConstraintX": False, "ConstraintY": True},
                "is_pass": True,
            }
        ]
    )

    with pytest.raises(KeyError, match="constraint_items"):
        constraints._with_primary_group(data)  # noqa: SLF001
