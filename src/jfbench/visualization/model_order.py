from __future__ import annotations

from typing import Iterable
from typing import Sequence

import pandas as pd


def _sanitize_model_name(value: str) -> str:
    sanitized = value.strip()
    sanitized = sanitized.replace("/", "-").replace("\\", "-")
    return sanitized.replace(" ", "_")


def order_labels(existing: Iterable[str], preferred: Sequence[str] | None) -> list[str]:
    values = [str(label) for label in existing]
    available = set(values)
    ordered: list[str] = []
    if preferred:
        for label in preferred:
            label_str = str(label)
            if label_str in available and label_str not in ordered:
                ordered.append(label_str)
    for label in values:
        if label not in ordered:
            ordered.append(label)
    return ordered


def preferred_model_labels(
    data: pd.DataFrame,
    models: Sequence[str] | None,
) -> list[str] | None:
    if data.empty or "model_label" not in data.columns:
        return None

    if models:
        sanitized_short = (
            data["model_short"].astype(str).map(_sanitize_model_name)
            if "model_short" in data.columns
            else None
        )
        sanitized_model = (
            data["model"].astype(str).map(_sanitize_model_name)
            if "model" in data.columns
            else None
        )

        ordered: list[str] = []
        for name in models:
            label_values = pd.Series(dtype=object)
            name_str = str(name)
            sanitized_name = _sanitize_model_name(name_str)
            if "model_short" in data.columns:
                label_values = data.loc[data["model_short"] == name_str, "model_label"]
                if label_values.empty and sanitized_short is not None:
                    label_values = data.loc[sanitized_short == sanitized_name, "model_label"]
            if label_values.empty and "model" in data.columns:
                label_values = data.loc[data["model"] == name_str, "model_label"]
                if label_values.empty and sanitized_model is not None:
                    label_values = data.loc[sanitized_model == sanitized_name, "model_label"]
            if not label_values.empty:
                ordered.append(str(label_values.iloc[0]))
            else:
                ordered.append(name_str)
        return ordered

    unique_labels = pd.unique(data["model_label"].astype(str))
    return list(unique_labels) if len(unique_labels) else None
