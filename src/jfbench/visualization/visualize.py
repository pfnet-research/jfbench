import argparse
import json
from pathlib import Path
from typing import cast
from typing import Literal
from typing import Sequence
import warnings

import pandas as pd

from jfbench.visualization.constraints import generate_constraint_charts
from jfbench.visualization.data_loader import load_results
from jfbench.visualization.data_loader import parse_result_filename
from jfbench.visualization.model_comparison import generate_model_comparison_outputs
from jfbench.visualization.model_order import order_labels
from jfbench.visualization.model_order import preferred_model_labels
from jfbench.visualization.overview import generate_overview_charts


DEFAULT_INPUT_DIR = Path("data/benchmark_results")
PromptSource = Literal["ifbench"]
PROMPT_SOURCE_CHOICES: tuple[PromptSource, ...] = ("ifbench",)
ModelLabelMap = dict[str, dict[str, str]]
ModelShortLabelMap = dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate benchmark visualizations.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing benchmark result JSONL files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("visualization_output"),
        help="Directory where charts will be written.",
    )
    parser.add_argument(
        "--drop-incomplete",
        action="store_true",
        help="Drop evaluations without completed results.",
    )
    parser.add_argument(
        "--n-constraints",
        dest="n_constraints",
        action="append",
        required=True,
        default=None,
        help=(
            "Constraint counts to include for multi-constraint visualizations. "
            "Can be provided multiple times or as a comma-separated string."
        ),
    )
    parser.add_argument(
        "--prompt-source",
        dest="prompt_sources",
        action="append",
        default=None,
        help=(
            "Filter visualizations to specific prompt sources "
            "(ifbench). "
            "Can be provided multiple times or as a comma-separated string."
        ),
    )
    parser.add_argument(
        "--models",
        dest="models",
        action="append",
        default=None,
        help=(
            "Filter visualizations to specific model names. "
            "Can be provided multiple times or as a comma-separated string."
        ),
    )
    parser.add_argument(
        "--constraint-set",
        dest="constraint_sets",
        action="append",
        default=None,
        help=(
            "Constraint set names to include (train, test). "
            "Can be provided multiple times or as a comma-separated string."
        ),
    )
    parser.add_argument(
        "--model-label-map",
        dest="model_label_map",
        default=None,
        help=(
            "JSON string mapping model_label to labels. "
            'Example: \'{"model-a": "Model A", "gpt-4": "GPT-4"}\'.'
        ),
    )
    return parser.parse_args()


def _normalize_prompt_sources(values: Sequence[str] | None) -> list[PromptSource] | None:
    if not values:
        return None
    normalized: list[PromptSource] = []
    for value in values:
        if not value:
            continue
        parts = [item.strip() for item in str(value).split(",")]
        for part in parts:
            if not part:
                continue
            if part not in PROMPT_SOURCE_CHOICES:
                allowed = ", ".join(PROMPT_SOURCE_CHOICES)
                raise ValueError(f"Unknown prompt source {part!r}. Allowed values: {allowed}")
            normalized.append(cast("PromptSource", part))
    return normalized or None


def _normalize_constraint_counts(values: Sequence[str | int] | None) -> list[int] | None:
    if not values:
        return None
    normalized: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value is None:
            continue
        parts = [item.strip() for item in str(value).split(",")]
        for part in parts:
            if not part:
                continue
            try:
                count = int(part)
            except ValueError as exc:  # pragma: no cover - defensive guard
                raise ValueError(
                    f"Invalid constraint count {part!r}. Constraint counts must be integers."
                ) from exc
            if count not in seen:
                normalized.append(count)
                seen.add(count)
    return normalized or None


def _normalize_models(values: Sequence[str] | None) -> list[str] | None:
    if not values:
        return None
    normalized: list[str] = []
    for value in values:
        if not value:
            continue
        parts = [item.strip() for item in str(value).split(",")]
        normalized.extend(part for part in parts if part)
    return normalized or None


def _normalize_constraint_sets(values: Sequence[str] | None) -> list[str] | None:
    if not values:
        return None
    normalized: list[str] = []
    for value in values:
        if not value:
            continue
        parts = [item.strip() for item in str(value).split(",")]
        for part in parts:
            if not part:
                continue
            if part not in {"train", "test"}:
                raise ValueError("Constraint set must be 'train' or 'test'.")
            normalized.append(part)
    return normalized or None


def _apply_model_label_order(df: pd.DataFrame, models: list[str] | None) -> pd.DataFrame:
    if df.empty or "model_label" not in df.columns:
        return df
    preferred_labels = preferred_model_labels(df, models)
    ordered_labels = order_labels(pd.unique(df["model_label"].astype(str)), preferred_labels)
    ordered = df.copy()
    ordered["model_label"] = pd.Categorical(
        ordered["model_label"].astype(str),
        categories=ordered_labels,
        ordered=True,
    )
    return ordered


def _normalize_model_label_map(raw: str | None) -> ModelShortLabelMap | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Model label map must be a valid JSON string.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Model label map must be a JSON object keyed by model_label.")
    normalized: ModelShortLabelMap = {}
    for model_name, label in parsed.items():
        if model_name is None or label is None:
            continue
        normalized[str(model_name)] = str(label)
    return normalized or None


def _collect_input_files(
    input_dir: Path,
    prompt_sources: list[PromptSource] | None,
    constraint_counts: list[int] | None,
    models: list[str] | None,
    constraint_sets: list[str] | None,
) -> list[Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        return []
    files = sorted(path for path in input_dir.glob("*.jsonl") if path.is_file())
    if not files:
        return []
    if not constraint_counts:
        raise ValueError("--n-constraints must be provided to select benchmark files.")
    constraint_set_filter = set(constraint_sets) if constraint_sets else None
    model_set = set(models) if models else None
    sources = set(prompt_sources or list(PROMPT_SOURCE_CHOICES))
    constraint_count_set = set(constraint_counts)

    def _matches(path: Path) -> bool:
        parsed = parse_result_filename(path)
        if not parsed:
            return False
        if parsed["benchmark"] not in sources:
            return False
        if parsed["n_constraints"] not in constraint_count_set:
            return False
        if constraint_set_filter and parsed["constraint_set"] not in constraint_set_filter:
            return False
        if model_set and parsed["model_short"] not in model_set:
            return False
        return True

    return [path for path in files if _matches(path)]


def _filter_dataframe(
    df: pd.DataFrame,
    drop_incomplete: bool,
    prompt_sources: list[PromptSource] | None,
    constraint_sets: list[str] | None,
) -> pd.DataFrame:
    df = df.copy()
    if drop_incomplete:
        df = df[df["results"].notna()]
    if df.empty:
        return df
    if prompt_sources:
        df = df[df["prompt_source"].isin(prompt_sources)]
    if constraint_sets and "constraint_set" in df.columns:
        df = df[df["constraint_set"].isin(constraint_sets)]
    return df


def filter_common_prompt_entries(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    working = df.copy()
    required_columns = [
        "prompt_source",
        "n_constraints",
        "prompt_index",
        "constraint_name",
        "model_label",
    ]
    missing = [column for column in required_columns if column not in working.columns]
    if "model_label" in missing and "model_short" in working.columns:
        warnings.warn(
            "model_label column missing; falling back to model_short values.",
            UserWarning,
        )
        working["model_label"] = working["model_short"].astype(str)
        missing = [column for column in required_columns if column not in working.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Results are missing required columns: {missing_str}")
    group_keys = ["prompt_source", "n_constraints", "model_label"]
    coverage_counts = (
        working.groupby(group_keys)["prompt_index"].nunique().rename("observed").reset_index()
    )
    expected_counts = (
        coverage_counts.groupby(["prompt_source", "n_constraints"])["observed"]
        .max()
        .rename("expected")
        .reset_index()
    )
    coverage = coverage_counts.merge(expected_counts, on=["prompt_source", "n_constraints"])
    incomplete = coverage[coverage["observed"] < coverage["expected"]]
    for row in incomplete.itertuples(index=False):
        warnings.warn(
            (
                "Incomplete data before grouping: expected "
                f"{row.expected} records for prompt_source={row.prompt_source}, "
                f"n_constraints={row.n_constraints}, model_label={row.model_label} "
                f"but found {row.observed}."
            ),
            UserWarning,
        )
    group_columns = ["prompt_source", "n_constraints"]
    total_models = (
        working.groupby(group_columns)["model_label"]
        .nunique()
        .rename("total_models")
        .reset_index()
    )
    combination_columns = [*group_columns, "prompt_index", "constraint_name"]
    shared_combinations = (
        working.groupby(combination_columns)["model_label"]
        .nunique()
        .rename("model_count")
        .reset_index()
    )
    merged = working.merge(total_models, on=group_columns, how="left")
    merged = merged.merge(shared_combinations, on=combination_columns, how="left")
    filtered = merged[merged["model_count"] == merged["total_models"]].copy()
    filtered = filtered.drop(columns=["total_models", "model_count"])
    return filtered.reset_index(drop=True)


def _select_constraint_data(
    df: pd.DataFrame,
    constraint_counts: list[int] | None,
) -> pd.DataFrame:
    if df.empty or "n_constraints" not in df.columns:
        return df
    if not constraint_counts:
        return df[df["n_constraints"] > 1].reset_index(drop=True).copy()
    return df[df["n_constraints"].isin(constraint_counts)].reset_index(drop=True).copy()


def _log_constraint_stats(df: pd.DataFrame) -> None:
    if df.empty:
        print("Filtered data: no records.")
        return
    if "model_label" in df.columns and "n_constraints" in df.columns:
        grouped = (
            df.groupby(["model_label", "n_constraints"], observed=False)
            .size()
            .reset_index(name="count")
            .sort_values(["model_label", "n_constraints"])
        )
        print("Filtered data:")
        for row in grouped.itertuples(index=False):
            message = (
                f"  model_label={row.model_label}, "
                f"n_constraints={row.n_constraints}: {row.count} records"
            )
            print(message)
    else:
        print(f"Filtered data: {len(df)} records (missing model_label or n_constraints).")


def main() -> None:
    args = parse_args()
    prompt_sources = _normalize_prompt_sources(args.prompt_sources)
    constraint_counts = _normalize_constraint_counts(args.n_constraints)
    models = _normalize_models(args.models)
    constraint_sets = _normalize_constraint_sets(args.constraint_sets)
    model_label_map = _normalize_model_label_map(args.model_label_map)
    input_files = _collect_input_files(
        args.input_dir,
        prompt_sources,
        constraint_counts,
        models,
        constraint_sets,
    )
    df = load_results(input_files, model_label_map)
    df = _filter_dataframe(df, args.drop_incomplete, prompt_sources, constraint_sets)
    df = filter_common_prompt_entries(df)

    if df.empty:
        print("No evaluation records found for the supplied inputs.")
        return

    df = _apply_model_label_order(df, models)

    primary_df = _select_constraint_data(df, constraint_counts)
    if primary_df.empty:
        print("No evaluation records found after constraint filtering.")
        return
    _log_constraint_stats(primary_df)

    output_dir = args.output_dir
    overview_dir = output_dir / "overview"
    constraints_dir = output_dir / "constraints"
    models_dir = output_dir / "models"

    overview_dir.mkdir(parents=True, exist_ok=True)
    generate_overview_charts(primary_df, overview_dir)

    constraints_dir.mkdir(parents=True, exist_ok=True)
    generate_constraint_charts(primary_df, constraints_dir)

    models_dir.mkdir(parents=True, exist_ok=True)
    generate_model_comparison_outputs(primary_df, models_dir)

    print(f"Visualizations saved under: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
