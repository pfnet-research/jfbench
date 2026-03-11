from pathlib import Path
from typing import Sequence

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _normalize_constraint_items(value: object) -> list[tuple[str, str]]:
    if value is None:
        return []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result: list[tuple[str, str]] = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                name, group = item
                if name is None or group is None:
                    continue
                result.append((str(name), str(group)))
        return result
    return []


def _with_primary_group(data: pd.DataFrame) -> pd.DataFrame:
    if "constraint_items" not in data.columns:
        raise KeyError("constraint_items column is required for constraint visualizations")
    prepared = data.copy()
    prepared["constraint_items"] = prepared["constraint_items"].apply(_normalize_constraint_items)
    prepared = prepared.explode("constraint_items")
    prepared = prepared.dropna(subset=["constraint_items"])
    if prepared.empty:
        return prepared
    prepared["constraint_name"] = prepared["constraint_items"].apply(lambda item: str(item[0]))
    prepared["constraint_group"] = prepared["constraint_items"].apply(lambda item: str(item[1]))
    prepared["constraint_pass"] = prepared.apply(_constraint_pass_value, axis=1)
    prepared = prepared.dropna(subset=["constraint_pass"])
    prepared["is_pass"] = prepared["constraint_pass"].astype(bool)
    prepared = prepared.drop(columns=["constraint_items", "constraint_pass"])
    return prepared


def _constraint_pass_value(row: pd.Series) -> object:
    results = row.get("results")
    name = row.get("constraint_name")
    if isinstance(results, dict) and isinstance(name, str):
        value = results.get(name)
        if value is not None:
            return bool(value)
    return None


def _constraint_order(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data[["constraint_group", "constraint_name"]]
        .drop_duplicates()
        .sort_values(["constraint_group", "constraint_name"])
    )


def plot_constraint_heatmap(data: pd.DataFrame, output_path: Path) -> None:
    order = _constraint_order(data)
    stats = (
        data.groupby(
            ["constraint_group", "constraint_name", "model_label"],
            observed=False,
        )["is_pass"]
        .mean()
        .reset_index()
    )
    pivot = stats.pivot(
        index=["constraint_group", "constraint_name"],
        columns="model_label",
        values="is_pass",
    )
    order_index = pd.MultiIndex.from_frame(order)
    pivot = pivot.reindex(order_index)
    pivot = pivot.fillna(0.0)
    display_index = [f"{group} | {constraint}" for group, constraint in pivot.index]

    fig_height = max(6, len(pivot) * 0.25)
    fig_width = max(8, len(pivot.columns) * 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    cax = ax.imshow(pivot.values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(display_index)))
    ax.set_yticklabels(display_index)

    group_boundaries = []
    current = 0
    for group in order["constraint_group"].unique():
        count = (order["constraint_group"] == group).sum()
        current += count
        group_boundaries.append(current)
    for boundary in group_boundaries[:-1]:
        ax.axhline(boundary - 0.5, color="white", linewidth=1)

    ax.set_title("Constraint Pass Rate Heatmap (Grouped)")
    fig.colorbar(cax, ax=ax, label="Pass Rate")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_constraint_group_heatmap(data: pd.DataFrame, output_path: Path) -> None:
    group_pivot = (
        data.groupby(["constraint_group", "model_label"], observed=False)["is_pass"]
        .mean()
        .unstack(fill_value=0.0)
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(8, max(4, len(group_pivot) * 0.6)))
    cax = ax.imshow(group_pivot.values, aspect="auto", cmap="plasma", vmin=0, vmax=1)
    ax.set_xticks(range(len(group_pivot.columns)))
    ax.set_xticklabels(group_pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(group_pivot.index)))
    ax.set_yticklabels(group_pivot.index)
    ax.set_title("Constraint Group Pass Rate Heatmap")
    fig.colorbar(cax, ax=ax, label="Pass Rate")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_constraint_group_bar(data: pd.DataFrame, output_path: Path) -> None:
    group_rates = (
        data.groupby(["constraint_group", "model_label"], observed=False)["is_pass"]
        .mean()
        .reset_index()
    )
    groups = group_rates["constraint_group"].unique()
    models = group_rates["model_label"].unique()
    x = np.arange(len(groups))
    bar_width = 0.8 / max(len(models), 1)

    fig, ax = plt.subplots(figsize=(12, max(4, len(groups) * 0.4)))
    for idx, model in enumerate(models):
        subset = group_rates[group_rates["model_label"] == model].set_index("constraint_group")
        heights = [
            subset.loc[group, "is_pass"] if group in subset.index else 0 for group in groups
        ]
        ax.bar(x + idx * bar_width, heights, width=bar_width, label=model)

    ax.set_xticks(x + bar_width * (len(models) - 1) / 2)
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Pass Rate")
    ax.set_title("Pass Rate by Constraint Group")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_constraint_group_bubble(data: pd.DataFrame, output_path: Path) -> None:
    summary = data.groupby("constraint_group", observed=False).agg(
        pass_rate=("is_pass", "mean"),
        evaluations=("is_pass", "size"),
    )
    std_by_group = (
        data.groupby(["constraint_group", "model_label"], observed=False)["is_pass"]
        .mean()
        .groupby("constraint_group")
        .std()
    )
    summary["std"] = std_by_group.fillna(0.0)

    x_positions = np.arange(len(summary))
    sizes = np.clip(
        summary["evaluations"].to_numpy() / summary["evaluations"].max() * 1500, 100, 2000
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    scatter = ax.scatter(
        x_positions,
        summary["pass_rate"],
        s=sizes,
        c=summary["std"],
        cmap="coolwarm",
        alpha=0.7,
        edgecolor="black",
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(summary.index, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Average Pass Rate")
    ax.set_title("Constraint Group Difficulty Bubble Chart")
    fig.colorbar(scatter, ax=ax, label="Std. Dev. Across Models")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_constraint_counts(data: pd.DataFrame, output_path: Path) -> None:
    counts = (
        data.groupby(["constraint_group", "constraint_name"])["model"]
        .count()
        .reset_index(name="count")
        .sort_values(["constraint_group", "constraint_name"])
    )
    fig_height = max(6, len(counts) * 0.3)
    fig, ax = plt.subplots(figsize=(10, fig_height))
    bars = ax.barh(
        range(len(counts)),
        counts["count"],
        color="steelblue",
        alpha=0.8,
    )
    labels = [f"{row.constraint_group} | {row.constraint_name}" for row in counts.itertuples()]
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Number of Evaluations")
    ax.set_title("Constraint Evaluation Counts")
    for bar, value in zip(bars, counts["count"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, str(value), va="center")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_constraint_charts(
    data: pd.DataFrame,
    output_dir: Path,
) -> None:
    prepared = _with_primary_group(data)
    plot_constraint_heatmap(prepared, output_dir / "constraints_heatmap.png")
    plot_constraint_group_heatmap(prepared, output_dir / "constraints_group_heatmap.png")
    plot_constraint_group_bar(prepared, output_dir / "constraints_group_pass_rates.png")
    plot_constraint_group_bubble(prepared, output_dir / "constraints_group_bubble.png")
    plot_constraint_counts(prepared, output_dir / "constraints_counts.png")
