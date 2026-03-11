from pathlib import Path

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_model_pass_rates(data: pd.DataFrame, output_path: Path) -> None:
    grouped = (
        data.groupby("model_label", as_index=False, observed=False)["is_pass"]
        .mean()
        .rename(columns={"is_pass": "pass_rate"})
        .sort_values("model_label")
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(grouped["model_label"], grouped["pass_rate"], color="#4C72B0", alpha=0.9)
    for bar, rate in zip(bars, grouped["pass_rate"], strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{float(rate):.2g}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylim(0, 1)
    ax.set_ylabel("Pass Rate")
    ax.set_title("Overall Pass Rate by Model")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def aggregate_pass_rates_by_constraints(data: pd.DataFrame) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(
            columns=[
                "n_constraints",
                "model_label",
                "pass_rate_mean",
                "pass_rate_std",
            ]
        )
    required = ("n_constraints", "model_label", "is_pass")
    missing = [column for column in required if column not in data.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Results are missing required columns: {missing_str}")
    prepared = data.dropna(subset=["is_pass"]).copy()
    if prepared.empty:
        return pd.DataFrame(
            columns=[
                "n_constraints",
                "model_label",
                "pass_rate_mean",
                "pass_rate_std",
            ]
        )
    if "seed" not in prepared.columns:
        prepared["seed"] = None
    seed_rates = (
        prepared.groupby(
            ["n_constraints", "model_label", "seed"],
            observed=False,
            dropna=False,
        )["is_pass"]
        .mean()
        .reset_index()
    )
    aggregated = (
        seed_rates.groupby(
            ["n_constraints", "model_label"],
            observed=False,
            dropna=False,
        )["is_pass"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "pass_rate_mean", "std": "pass_rate_std"})
    )
    aggregated["pass_rate_std"] = aggregated["pass_rate_std"].fillna(0.0)
    aggregated = aggregated.sort_values(["model_label", "n_constraints"])
    return aggregated


def plot_constraints_pass_rate(data: pd.DataFrame, output_path: Path) -> None:
    aggregated = aggregate_pass_rates_by_constraints(data)
    if aggregated.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for model_label, model_frame in aggregated.groupby("model_label", observed=False):
        model_frame = model_frame.sort_values("n_constraints")
        x_values = model_frame["n_constraints"]
        means = model_frame["pass_rate_mean"]
        stds = model_frame["pass_rate_std"]
        ax.plot(x_values, means, marker="o", label=model_label)
        lower = np.clip(means - stds, 0.0, 1.0)
        upper = np.clip(means + stds, 0.0, 1.0)
        ax.fill_between(x_values, lower, upper, alpha=0.2)
    ax.set_xlabel("n_constraints")
    ax.set_ylabel("Overall Pass Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Pass Rate by Constraint Count")
    ax.legend(loc="best")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_overview_charts(data: pd.DataFrame, output_dir: Path) -> None:
    plot_model_pass_rates(data, output_dir / "overview_model_pass_rates.png")
    plot_constraints_pass_rate(data, output_dir / "overview_constraints_pass_rate.png")
