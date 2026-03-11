from pathlib import Path

import pandas as pd

from jfbench.visualization.model_comparison import generate_model_comparison_outputs


def _sample_dataframe() -> pd.DataFrame:
    data = []
    for prompt_index in range(3):
        for constraint in ("C1", "C2"):
            data.append(
                {
                    "model": "model-a",
                    "model_short": "A",
                    "model_label": "A",
                    "prompt_index": prompt_index,
                    "constraint_name": constraint,
                    "is_pass": True,
                }
            )
            data.append(
                {
                    "model": "model-b",
                    "model_short": "B",
                    "model_label": "B",
                    "prompt_index": prompt_index,
                    "constraint_name": constraint,
                    "is_pass": prompt_index % 2 == 0,
                }
            )
    return pd.DataFrame.from_records(data)


def test_generate_model_comparison_outputs(tmp_path: Path) -> None:
    df = _sample_dataframe()
    output_dir = tmp_path / "models"

    generate_model_comparison_outputs(df, output_dir)

    assert (output_dir / "models_pairwise_win_rate.png").exists()
    assert (output_dir / "models_cohen_kappa.png").exists()
    stats_path = output_dir / "models_pairwise_stats.csv"
    assert not stats_path.exists()
