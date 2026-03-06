import json
from pathlib import Path
import re
from typing import Iterable
from typing import Sequence

import pandas as pd


KEY_COLUMNS = (
    "model",
    "model_label",
    "prompt_index",
    "constraint_name",
    "prompt_source",
    "constraint_set",
    "seed",
)
_SEED_IN_MODEL_SHORT_PATTERN = re.compile(
    r"^(?P<base>.+?)\s*\(\s*seed\s+(?P<seed>.+?)\s*\)\s*$", flags=re.IGNORECASE
)


def parse_result_filename(path: Path) -> dict[str, str | int] | None:
    stem = path.stem
    parts = stem.split("-")
    if len(parts) < 3:
        return None
    constraint_set = "train"
    if len(parts) >= 4 and parts[1] in {"train", "test"}:
        benchmark = parts[0]
        constraint_set = parts[1]
        n_part = parts[2]
        model_short = "-".join(parts[3:])
    else:
        benchmark = parts[0]
        n_part = parts[1]
        model_short = "-".join(parts[2:])
    try:
        n_constraints = int(n_part)
    except ValueError:
        return None
    if not benchmark or not model_short:
        return None
    return {
        "benchmark": benchmark,
        "constraint_set": constraint_set,
        "n_constraints": n_constraints,
        "model_short": model_short,
    }


def _read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_json(path, lines=True)
    except ValueError:
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                rows.append(json.loads(stripped))
        return pd.DataFrame(rows)


def _normalize_sequence(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _extract_constraint(row: pd.Series) -> pd.Series:
    names = row.get("constraint_types")
    groups = row.get("constraint_groups")
    normalized_names = [str(name) for name in _normalize_sequence(names)]
    normalized_groups = [str(value) for value in _normalize_sequence(groups)]
    if not normalized_names:
        return pd.Series(
            {
                "constraint_name": None,
                "constraint_groups": [],
                "constraint_items": tuple(),
                "n_constraints": 0,
            }
        )
    if not normalized_groups:
        normalized_groups = ["Unknown"] * len(normalized_names)
    if len(normalized_groups) < len(normalized_names):
        fill_value = normalized_groups[-1] if normalized_groups else "Unknown"
        normalized_groups.extend([fill_value] * (len(normalized_names) - len(normalized_groups)))
    if len(normalized_groups) > len(normalized_names):
        normalized_groups = normalized_groups[: len(normalized_names)]
    pairs = [(name, group) for name, group in zip(normalized_names, normalized_groups)]
    pairs.sort(key=lambda item: item[0])
    sorted_names = tuple(name for name, _ in pairs)
    sorted_groups = [group for _, group in pairs]
    return pd.Series(
        {
            "constraint_name": sorted_names,
            "constraint_groups": sorted_groups,
            "constraint_items": tuple(pairs),
            "n_constraints": len(sorted_names),
        }
    )


def _results_to_pass(results: object) -> bool | None:
    if results is None:
        return None
    if isinstance(results, dict):
        if not results:
            return None
        values = [bool(value) for value in results.values()]
        return all(values)
    return None


def _validate_required_columns(df: pd.DataFrame) -> None:
    required = ("model", "prompt_index", "constraint_types", "results")
    missing = [column for column in required if column not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Results are missing required columns: {missing_str}")


def _raise_on_missing_values(df: pd.DataFrame, column: str, label: str) -> None:
    if df[column].isna().any():
        count = int(df[column].isna().sum())
        raise ValueError(f"{label} are missing for {count} rows.")


def _validate_prompt_index(df: pd.DataFrame) -> None:
    _raise_on_missing_values(df, "prompt_index", "prompt_index values")
    try:
        df["prompt_index"] = pd.to_numeric(df["prompt_index"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("prompt_index must be numeric for all rows.") from exc
    df["prompt_index"] = df["prompt_index"].astype(int)


def _apply_model_label_overrides(
    df: pd.DataFrame,
    overrides: dict[str, str] | None,
) -> pd.Series:
    if not overrides:
        return df["model_label"]

    def _resolve(row: pd.Series) -> str:
        for key in (row.get("model_short_base"), row.get("model_short"), row.get("model")):
            if isinstance(key, str) and key in overrides:
                return str(overrides[key])
        return str(row["model_label"])

    return df.apply(_resolve, axis=1).astype(str)


def _split_model_short_seed(value: object) -> tuple[str, str | None]:
    raw = "" if value is None else str(value)
    match = _SEED_IN_MODEL_SHORT_PATTERN.match(raw)
    if match:
        base = match.group("base").strip()
        seed = match.group("seed").strip()
        return base or raw, seed or None
    return raw, None


def load_results(
    paths: Iterable[Path],
    model_label_overrides: dict[str, str] | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = _read_jsonl(path)
        if not frame.empty:
            parsed = parse_result_filename(path)
            constraint_set = parsed["constraint_set"] if parsed else "unknown"
            if "constraint_set" not in frame.columns:
                frame["constraint_set"] = constraint_set
            else:
                frame["constraint_set"] = frame["constraint_set"].fillna(constraint_set)
            frames.append(frame)
    if not frames:
        columns = [
            "model",
            "model_short",
            "model_label",
            "prompt_index",
            "constraint_name",
            "is_pass",
            "results",
            "response",
            "data_id",
            "n_constraints",
            "constraint_set",
        ]
        return pd.DataFrame(columns=columns)

    combined = pd.concat(frames, ignore_index=True)
    _validate_required_columns(combined)
    constraint_info = combined.apply(_extract_constraint, axis=1)
    combined["constraint_name"] = constraint_info["constraint_name"]
    combined["constraint_groups"] = constraint_info["constraint_groups"]
    combined["constraint_items"] = constraint_info["constraint_items"]
    combined["n_constraints"] = constraint_info["n_constraints"].astype(int)
    combined["is_pass"] = combined["results"].apply(_results_to_pass)
    _raise_on_missing_values(combined, "model", "Model values")
    _raise_on_missing_values(combined, "constraint_name", "Constraint definitions")
    _raise_on_missing_values(combined, "is_pass", "Evaluation results")
    if (combined["n_constraints"] <= 0).any():
        raise ValueError("Constraint information is missing for one or more rows.")
    _validate_prompt_index(combined)
    if "model_short" in combined.columns:
        short_values = combined["model_short"]
    else:
        short_values = pd.Series([None] * len(combined), index=combined.index)

    parsed_short = short_values.apply(_split_model_short_seed)
    combined["model_short_base"] = parsed_short.apply(lambda pair: pair[0])
    combined["seed"] = parsed_short.apply(lambda pair: pair[1])
    combined["model_label"] = combined["model_short_base"].where(
        combined["model_short_base"].notna(), combined["model"]
    )
    combined["model_label"] = combined["model_label"].astype(str)
    combined["model_label"] = _apply_model_label_overrides(combined, model_label_overrides)
    if "prompt_source" not in combined.columns:
        combined["prompt_source"] = None

    combined = combined.drop_duplicates(subset=list(KEY_COLUMNS), keep="last")
    combined = combined.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)

    return combined
