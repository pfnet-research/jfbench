from pathlib import Path

import pandas as pd
import pytest

from jfbench.visualization.overview import aggregate_pass_rates_by_constraints
from jfbench.visualization.overview import generate_overview_charts


def _sample_dataframe() -> pd.DataFrame:
    records = []
    for model in ("model-a", "model-b"):
        for prompt_index in range(3):
            for constraint in ("ConstraintX", "ConstraintY"):
                records.append(
                    {
                        "model": model,
                        "prompt_index": prompt_index,
                        "constraint_name": constraint,
                        "is_pass": (prompt_index + (model == "model-a")) % 2 == 0,
                        "n_constraints": 2,
                    }
                )
    df = pd.DataFrame.from_records(records)
    df["model_short"] = df["model"].str.replace("model-", "", regex=False)
    df["model_label"] = df["model_short"]
    return df


def test_generate_overview_charts_creates_files(tmp_path: Path) -> None:
    df = _sample_dataframe()
    output_dir = tmp_path / "overview"

    generate_overview_charts(df, output_dir)

    expected_files = {"overview_model_pass_rates.png", "overview_constraints_pass_rate.png"}
    produced = {path.name for path in output_dir.glob("*.png")}
    assert expected_files <= produced


def test_aggregate_pass_rates_by_constraints_groups_seeds() -> None:
    df = pd.DataFrame(
        [
            {"n_constraints": 2, "model_label": "alpha", "seed": "0", "is_pass": True},
            {"n_constraints": 2, "model_label": "alpha", "seed": "0", "is_pass": False},
            {"n_constraints": 2, "model_label": "alpha", "seed": "1", "is_pass": True},
            {"n_constraints": 2, "model_label": "beta", "seed": None, "is_pass": False},
            {"n_constraints": 3, "model_label": "beta", "seed": None, "is_pass": True},
        ]
    )

    aggregated = aggregate_pass_rates_by_constraints(df)

    assert list(aggregated.columns) == [
        "n_constraints",
        "model_label",
        "pass_rate_mean",
        "pass_rate_std",
    ]
    assert len(aggregated) == 3
    alpha_row = aggregated[
        (aggregated["model_label"] == "alpha") & (aggregated["n_constraints"] == 2)
    ].iloc[0]
    assert alpha_row["pass_rate_mean"] == 0.75
    assert alpha_row["pass_rate_std"] == pytest.approx(0.353553, rel=1e-4)
    beta_two = aggregated[
        (aggregated["model_label"] == "beta") & (aggregated["n_constraints"] == 2)
    ].iloc[0]
    assert beta_two["pass_rate_mean"] == 0.0
    assert beta_two["pass_rate_std"] == 0.0


def test_plot_constraints_pass_rate_creates_file(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "n_constraints": 1,
                "model_label": "alpha",
                "seed": "0",
                "is_pass": True,
            },
            {
                "n_constraints": 2,
                "model_label": "alpha",
                "seed": "0",
                "is_pass": False,
            },
            {"n_constraints": 1, "model_label": "beta", "seed": None, "is_pass": True},
            {"n_constraints": 2, "model_label": "beta", "seed": None, "is_pass": True},
        ]
    )

    generate_overview_charts(df, tmp_path)

    expected_paths = [
        tmp_path / "overview_model_pass_rates.png",
        tmp_path / "overview_constraints_pass_rate.png",
    ]
    for path in expected_paths:
        assert path.exists()
