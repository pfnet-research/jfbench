from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any
from typing import Sequence

import datasets


DEFAULT_DATASET_PATH = "data/dpo_dataset"
DEFAULT_OUTPUT_PATH = "data/dpo_dataset_postprocessed"
TOKEN_LENGTH_KEYS = (
    "chosen_token_length",
    "rejected_token_length",
    "chosen_reasoning_token_length",
    "rejected_reasoning_token_length",
)


@dataclass(frozen=True)
class _Thresholds:
    token_length_p95: dict[str, float]
    chosen_delta_logp_p50: float


@dataclass(frozen=True)
class PostprocessStats:
    total_records: int
    removed_by_token_length_p95: int
    removed_by_individual_sum_reward_difference: int
    removed_by_chat_template_reward_difference: int
    removed_by_chosen_delta_logp_p50: int
    removed_by_rejected_delta_logp_non_negative: int
    removed_by_delta_margin_non_positive: int
    removed_by_pass_count_difference_threshold: int
    saved_records: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Postprocess DPO datasets by filtering rows using metrics computed by analyze.py."
        )
    )
    parser.add_argument(
        "dataset_paths",
        nargs="*",
        default=(DEFAULT_DATASET_PATH,),
        help="Paths to HuggingFace Dataset directories.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(DEFAULT_OUTPUT_PATH),
        help="Output path for the filtered HuggingFace Dataset.",
    )
    parser.add_argument(
        "--apply-token-length-p95-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Remove rows when any token length in chosen/rejected/chosen reasoning/"
            "rejected reasoning is >= p95."
        ),
    )
    parser.add_argument(
        "--apply-individual-sum-reward-difference-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove rows when individual sum reward difference <= 0.",
    )
    parser.add_argument(
        "--apply-chat-template-reward-difference-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove rows when chat template reward difference <= 0.",
    )
    parser.add_argument(
        "--apply-chosen-delta-logp-p50-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove rows when chosen_delta_logp <= p50 of chosen_delta_logp.",
    )
    parser.add_argument(
        "--apply-rejected-delta-logp-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove rows when rejected_delta_logp >= 0.",
    )
    parser.add_argument(
        "--apply-delta-margin-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove rows when delta_margin <= 0.",
    )
    parser.add_argument(
        "--apply-pass-count-difference-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Remove rows when (chosen pass count - rejected pass count) <= 7.",
    )
    return parser.parse_args(argv)


def postprocess_datasets(
    dataset_paths: Sequence[str],
    *,
    output_path: Path,
    apply_token_length_p95_filter: bool = True,
    apply_individual_sum_reward_difference_filter: bool = True,
    apply_chat_template_reward_difference_filter: bool = True,
    apply_chosen_delta_logp_p50_filter: bool = True,
    apply_rejected_delta_logp_filter: bool = True,
    apply_delta_margin_filter: bool = True,
    apply_pass_count_difference_filter: bool = True,
) -> PostprocessStats:
    merged = _load_merged_dataset(dataset_paths)
    total_records = len(merged)
    thresholds = _compute_thresholds(merged)
    keep_mask = [True] * total_records

    removed_by_token_length_p95 = _apply_filter(
        merged,
        keep_mask,
        enabled=apply_token_length_p95_filter,
        predicate=lambda row: _fails_token_length_p95(row, thresholds.token_length_p95),
    )
    removed_by_individual_sum_reward_difference = _apply_filter(
        merged,
        keep_mask,
        enabled=apply_individual_sum_reward_difference_filter,
        predicate=_fails_individual_sum_reward_difference,
    )
    removed_by_chat_template_reward_difference = _apply_filter(
        merged,
        keep_mask,
        enabled=apply_chat_template_reward_difference_filter,
        predicate=_fails_chat_template_reward_difference,
    )
    removed_by_chosen_delta_logp_p50 = _apply_filter(
        merged,
        keep_mask,
        enabled=apply_chosen_delta_logp_p50_filter,
        predicate=lambda row: float(row["chosen_delta_logp"]) <= thresholds.chosen_delta_logp_p50,
    )
    removed_by_rejected_delta_logp_non_negative = _apply_filter(
        merged,
        keep_mask,
        enabled=apply_rejected_delta_logp_filter,
        predicate=lambda row: float(row["rejected_delta_logp"]) >= 0.0,
    )
    removed_by_delta_margin_non_positive = _apply_filter(
        merged,
        keep_mask,
        enabled=apply_delta_margin_filter,
        predicate=lambda row: float(row["delta_margin"]) <= 0.0,
    )
    removed_by_pass_count_difference_threshold = _apply_filter(
        merged,
        keep_mask,
        enabled=apply_pass_count_difference_filter,
        predicate=lambda row: _pass_count_difference(row) <= 7.0,
    )

    kept_indices = [idx for idx, keep in enumerate(keep_mask) if keep]
    filtered = merged.select(kept_indices)
    _save_dataset(filtered, output_path)

    stats = PostprocessStats(
        total_records=total_records,
        removed_by_token_length_p95=removed_by_token_length_p95,
        removed_by_individual_sum_reward_difference=removed_by_individual_sum_reward_difference,
        removed_by_chat_template_reward_difference=removed_by_chat_template_reward_difference,
        removed_by_chosen_delta_logp_p50=removed_by_chosen_delta_logp_p50,
        removed_by_rejected_delta_logp_non_negative=removed_by_rejected_delta_logp_non_negative,
        removed_by_delta_margin_non_positive=removed_by_delta_margin_non_positive,
        removed_by_pass_count_difference_threshold=removed_by_pass_count_difference_threshold,
        saved_records=len(filtered),
    )
    _print_stats(
        stats,
        thresholds=thresholds,
        output_path=output_path,
        enabled_filters={
            "token_length_p95": apply_token_length_p95_filter,
            "individual_sum_reward_difference": apply_individual_sum_reward_difference_filter,
            "chat_template_reward_difference": apply_chat_template_reward_difference_filter,
            "chosen_delta_logp_p50": apply_chosen_delta_logp_p50_filter,
            "rejected_delta_logp_non_negative": apply_rejected_delta_logp_filter,
            "delta_margin_non_positive": apply_delta_margin_filter,
            "pass_count_difference_threshold": apply_pass_count_difference_filter,
        },
    )
    return stats


def _load_merged_dataset(dataset_paths: Sequence[str]) -> datasets.Dataset:
    loaded: list[datasets.Dataset] = []
    for raw_path in dataset_paths:
        path = Path(raw_path)
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {path}")
        dataset = datasets.load_from_disk(str(path))
        if isinstance(dataset, datasets.DatasetDict):
            for split in dataset.values():
                loaded.append(split)
        else:
            loaded.append(dataset)
    if not loaded:
        raise ValueError("No datasets were loaded.")
    if len(loaded) == 1:
        return loaded[0]
    return datasets.concatenate_datasets(loaded)


def _save_dataset(dataset: datasets.Dataset, output_path: Path) -> None:
    tmp_path = output_path.with_name(f"{output_path.name}_postprocess_tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    dataset.save_to_disk(str(tmp_path))
    if output_path.exists():
        shutil.rmtree(output_path)
    shutil.move(str(tmp_path), str(output_path))


def _apply_filter(
    dataset: datasets.Dataset,
    keep_mask: list[bool],
    *,
    enabled: bool,
    predicate: Any,
) -> int:
    if not enabled:
        return 0
    removed = 0
    for idx, row in enumerate(dataset):
        if not keep_mask[idx]:
            continue
        if predicate(row):
            keep_mask[idx] = False
            removed += 1
    return removed


def _compute_thresholds(dataset: datasets.Dataset) -> _Thresholds:
    token_length_p95 = {
        key: _percentile([float(row[key]) for row in dataset], 0.95) for key in TOKEN_LENGTH_KEYS
    }
    chosen_delta_logp_p50 = _percentile([float(row["chosen_delta_logp"]) for row in dataset], 0.50)
    return _Thresholds(
        token_length_p95=token_length_p95,
        chosen_delta_logp_p50=chosen_delta_logp_p50,
    )


def _percentile(values: Sequence[float], ratio: float) -> float:
    sorted_values = sorted(values)
    if not sorted_values:
        return 0.0
    idx = round((len(sorted_values) - 1) * ratio)
    idx = max(0, min(idx, len(sorted_values) - 1))
    return float(sorted_values[idx])


def _fails_token_length_p95(row: dict[str, Any], thresholds: dict[str, float]) -> bool:
    for key in TOKEN_LENGTH_KEYS:
        if float(row[key]) >= thresholds[key]:
            return True
    return False


def _fails_individual_sum_reward_difference(row: dict[str, Any]) -> bool:
    difference = float(row["sum_of_chosen_and_chosen_reasoning_reward"]) - float(
        row["sum_of_rejected_and_rejected_reasoning_reward"]
    )
    return difference <= 0.0


def _fails_chat_template_reward_difference(row: dict[str, Any]) -> bool:
    difference = float(row["concatenated_chosen_and_chosen_reasoning_reward"]) - float(
        row["concatenated_rejected_and_rejected_reasoning_reward"]
    )
    return difference <= 0.0


def _pass_count_difference(row: dict[str, Any]) -> float:
    return float(row["chosen_all_constraints_pass_count"]) - float(
        row["rejected_al_constraints_pass_count"]
    )


def _print_stats(
    stats: PostprocessStats,
    *,
    thresholds: _Thresholds,
    output_path: Path,
    enabled_filters: dict[str, bool],
) -> None:
    print("Postprocess summary:")
    print(f"  Total records: {stats.total_records}")
    print("  Thresholds:")
    for key in TOKEN_LENGTH_KEYS:
        print(f"    {key} p95: {thresholds.token_length_p95[key]:.4f}")
    print(f"    chosen_delta_logp p50: {thresholds.chosen_delta_logp_p50:.4f}")
    print("  Removed records by filter:")
    print(
        "    token length >= p95: "
        f"{stats.removed_by_token_length_p95} (enabled={enabled_filters['token_length_p95']})"
    )
    print(
        "    individual sum reward difference <= 0: "
        f"{stats.removed_by_individual_sum_reward_difference} "
        f"(enabled={enabled_filters['individual_sum_reward_difference']})"
    )
    print(
        "    chat template reward difference <= 0: "
        f"{stats.removed_by_chat_template_reward_difference} "
        f"(enabled={enabled_filters['chat_template_reward_difference']})"
    )
    print(
        "    chosen delta_logp <= p50: "
        f"{stats.removed_by_chosen_delta_logp_p50} "
        f"(enabled={enabled_filters['chosen_delta_logp_p50']})"
    )
    print(
        "    rejected delta_logp >= 0: "
        f"{stats.removed_by_rejected_delta_logp_non_negative} "
        f"(enabled={enabled_filters['rejected_delta_logp_non_negative']})"
    )
    print(
        "    delta_margin <= 0: "
        f"{stats.removed_by_delta_margin_non_positive} "
        f"(enabled={enabled_filters['delta_margin_non_positive']})"
    )
    print(
        "    all-constraints-pass count difference <= 7: "
        f"{stats.removed_by_pass_count_difference_threshold} "
        f"(enabled={enabled_filters['pass_count_difference_threshold']})"
    )
    print(f"  Saved records: {stats.saved_records}")
    print(f"  Output path: {output_path}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    postprocess_datasets(
        args.dataset_paths,
        output_path=args.output_path,
        apply_token_length_p95_filter=bool(args.apply_token_length_p95_filter),
        apply_individual_sum_reward_difference_filter=bool(
            args.apply_individual_sum_reward_difference_filter
        ),
        apply_chat_template_reward_difference_filter=bool(
            args.apply_chat_template_reward_difference_filter
        ),
        apply_chosen_delta_logp_p50_filter=bool(args.apply_chosen_delta_logp_p50_filter),
        apply_rejected_delta_logp_filter=bool(args.apply_rejected_delta_logp_filter),
        apply_delta_margin_filter=bool(args.apply_delta_margin_filter),
        apply_pass_count_difference_filter=bool(args.apply_pass_count_difference_filter),
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
