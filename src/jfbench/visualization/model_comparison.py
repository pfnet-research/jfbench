import itertools
import math
from pathlib import Path

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _aligned_matrix(data: pd.DataFrame) -> pd.DataFrame:
    pivot = (
        data.pivot_table(
            index=["prompt_index", "constraint_name"],
            columns="model_label",
            values="is_pass",
            observed=False,
        )
        .dropna(axis=0, how="any")
        .astype(int)
    )
    return pivot


def _pairwise_counts(pivot: pd.DataFrame) -> dict[tuple[str, str], dict[str, int | float]]:
    models = list(pivot.columns)
    results: dict[tuple[str, str], dict[str, int | float]] = {}
    for model_a, model_b in itertools.permutations(models, 2):
        a_vals = pivot[model_a]
        b_vals = pivot[model_b]
        both = pd.crosstab(a_vals, b_vals)

        def _lookup(row_value: int, col_value: int) -> int:
            if row_value in both.index and col_value in both.columns:
                return int(both.loc[row_value, col_value])
            return 0

        tp = _lookup(1, 1)
        tn = _lookup(0, 0)
        fp = _lookup(1, 0)
        fn = _lookup(0, 1)
        results[(model_a, model_b)] = {
            "wins": int(((a_vals == 1) & (b_vals == 0)).sum()),
            "losses": int(((a_vals == 0) & (b_vals == 1)).sum()),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }
    return results


def _cohen_kappa(tp: int, tn: int, fp: int, fn: int) -> float:
    total = tp + tn + fp + fn
    if total == 0:
        return float("nan")
    po = (tp + tn) / total
    pe = (((tp + fp) * (tp + fn)) + ((fn + tn) * (fp + tn))) / (total**2)
    if math.isclose(1 - pe, 0.0):
        return float("nan")
    return (po - pe) / (1 - pe)


def _win_matrix(pivot: pd.DataFrame) -> pd.DataFrame:
    counts = _pairwise_counts(pivot)
    models = list(pivot.columns)
    matrix = pd.DataFrame(index=models, columns=models, dtype=float)
    for model_a in models:
        for model_b in models:
            if model_a == model_b:
                matrix.loc[model_a, model_b] = 0.0
                continue
            record = counts[(model_a, model_b)]
            total = record["wins"] + record["losses"]
            value = record["wins"] / total if total else 0.0
            matrix.loc[model_a, model_b] = value
    return matrix


def _kappa_matrix(pivot: pd.DataFrame) -> pd.DataFrame:
    counts = _pairwise_counts(pivot)
    models = list(pivot.columns)
    matrix = pd.DataFrame(index=models, columns=models, dtype=float)
    for model_a in models:
        for model_b in models:
            if model_a == model_b:
                matrix.loc[model_a, model_b] = 1.0
                continue
            record = counts[(model_a, model_b)]
            matrix.loc[model_a, model_b] = _cohen_kappa(
                int(record["tp"]), int(record["tn"]), int(record["fp"]), int(record["fn"])
            )
    return matrix


def plot_win_matrix(data: pd.DataFrame, output_path: Path) -> None:
    pivot = _aligned_matrix(data)
    matrix = _win_matrix(pivot)
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(matrix.values, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    for (i, j), value in np.ndenumerate(matrix.to_numpy()):
        if i == j:
            continue
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="black", fontsize=9)
    ax.set_title("Pairwise Win Rate (row defeats column)")
    fig.colorbar(cax, ax=ax, label="Win Rate")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_kappa_matrix(data: pd.DataFrame, output_path: Path) -> None:
    pivot = _aligned_matrix(data)
    matrix = _kappa_matrix(pivot)
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(matrix.values, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    for (i, j), value in np.ndenumerate(matrix.to_numpy()):
        ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="black", fontsize=9)
    ax.set_title("Cohen's Kappa Between Models")
    fig.colorbar(cax, ax=ax, label="κ")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_model_comparison_outputs(data: pd.DataFrame, output_dir: Path) -> None:
    pivot = _aligned_matrix(data)
    if pivot.empty:
        return
    plot_win_matrix(data, output_dir / "models_pairwise_win_rate.png")
    plot_kappa_matrix(data, output_dir / "models_cohen_kappa.png")
