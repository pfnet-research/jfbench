from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Sequence

from datasets import concatenate_datasets
from datasets import Dataset
from datasets import DatasetDict
from datasets import load_from_disk


DEFAULT_MAX_REPORT = 20
INSTRUCTION_SUMMARY_LIMIT = 80


@dataclass(frozen=True)
class DuplicateEntry:
    instruction: str
    occurrences: int
    indices: list[int]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        description="Merge HuggingFace Dataset directories and screen duplicate instructions.",
    )
    parser.add_argument(
        "dataset_paths",
        nargs="+",
        type=Path,
        help="Paths to HuggingFace Dataset directories created via datasets.save_to_disk.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination directory to store the deduplicated dataset.",
    )
    parser.add_argument(
        "--instruction-column",
        type=str,
        default="instruction",
        help="Column name used to detect duplicate rows.",
    )
    parser.add_argument(
        "--max-report",
        type=int,
        default=DEFAULT_MAX_REPORT,
        help="Maximum number of duplicate entries to show per split (0 to disable).",
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Show duplicate summary without writing any changes.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.max_report < 0:
        parser.error("--max-report must be non-negative.")
    if not args.dry and args.output is None:
        parser.error("--output is required unless --dry is provided.")
    return args


def load_dataset_directory(path: Path) -> DatasetDict:
    dataset = load_from_disk(str(path))
    if isinstance(dataset, Dataset):
        return DatasetDict({"train": dataset})
    if isinstance(dataset, DatasetDict):
        return dataset
    raise TypeError(f"Unsupported dataset type loaded from '{path}'.")


def merge_dataset_directories(paths: Sequence[Path]) -> DatasetDict:
    merged: dict[str, Dataset] = {}
    for path in paths:
        dataset = load_dataset_directory(path)
        for split_name, split in dataset.items():
            if split_name in merged:
                merged[split_name] = concatenate_datasets([merged[split_name], split])
            else:
                merged[split_name] = split
    return DatasetDict(merged)


def collect_duplicate_rows(
    dataset: DatasetDict, instruction_column: str
) -> dict[str, list[DuplicateEntry]]:
    duplicates_by_split: dict[str, list[DuplicateEntry]] = {}
    for split_name, split in dataset.items():
        if instruction_column not in split.column_names:
            raise ValueError(
                f"Column '{instruction_column}' does not exist in split '{split_name}'."
            )
        value_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx, value in enumerate(split[instruction_column]):
            value_to_indices[_ensure_string(value, instruction_column, split_name)].append(idx)
        duplicates = [
            DuplicateEntry(instruction=value, occurrences=len(indices), indices=indices)
            for value, indices in value_to_indices.items()
            if len(indices) > 1
        ]
        duplicates.sort(key=lambda entry: (-entry.occurrences, entry.instruction))
        duplicates_by_split[split_name] = duplicates
    return duplicates_by_split


def deduplicate_dataset(
    dataset: DatasetDict, instruction_column: str
) -> tuple[DatasetDict, dict[str, int]]:
    output_splits: dict[str, Dataset] = {}
    removed_counts: dict[str, int] = {}
    for split_name, split in dataset.items():
        if instruction_column not in split.column_names:
            raise ValueError(
                f"Column '{instruction_column}' does not exist in split '{split_name}'."
            )
        seen: set[str] = set()
        keep_mask: list[bool] = []
        removed = 0
        for value in split[instruction_column]:
            normalized = _ensure_string(value, instruction_column, split_name)
            if normalized in seen:
                keep_mask.append(False)
                removed += 1
                continue
            seen.add(normalized)
            keep_mask.append(True)
        if removed:
            output_splits[split_name] = split.filter(
                lambda _, idx: keep_mask[idx], with_indices=True
            )
        else:
            output_splits[split_name] = split
        removed_counts[split_name] = removed
    return DatasetDict(output_splits), removed_counts


def format_duplicate_report(
    duplicates_by_split: dict[str, list[DuplicateEntry]], max_report: int = DEFAULT_MAX_REPORT
) -> str:
    total_duplicates = sum(
        entry.occurrences - 1 for entries in duplicates_by_split.values() for entry in entries
    )
    lines: list[str] = []
    if total_duplicates:
        lines.append(f"Detected {total_duplicates} duplicate rows.")
    else:
        lines.append("No duplicate instructions detected.")
    for split_name in sorted(duplicates_by_split):
        entries = duplicates_by_split[split_name]
        duplicate_rows = sum(entry.occurrences - 1 for entry in entries)
        lines.append(f"[{split_name}] duplicate rows: {duplicate_rows}")
        if duplicate_rows and max_report:
            for entry in entries[:max_report]:
                summary = _summarize_instruction(entry.instruction)
                lines.append(
                    f"  - '{summary}' ({entry.occurrences} occurrences; indices: {entry.indices})"
                )
    return "\n".join(lines)


def ensure_output_path_available(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Output path '{path}' already exists.")
    path.parent.mkdir(parents=True, exist_ok=True)


def _ensure_string(value: Any, column: str, split_name: str) -> str:
    if isinstance(value, str):
        return value
    raise TypeError(
        f"Column '{column}' in split '{split_name}' contains non-string values, which is unsupported."
    )


def _summarize_instruction(value: str) -> str:
    single_line = value.replace("\n", "\\n")
    if len(single_line) <= INSTRUCTION_SUMMARY_LIMIT:
        return single_line
    return f"{single_line[: INSTRUCTION_SUMMARY_LIMIT - 3]}..."


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = merge_dataset_directories(args.dataset_paths)
    print("Merged dataset splits:")
    for split_name, split in dataset.items():
        print(f"  - {split_name}: {len(split)} rows")
    duplicates = collect_duplicate_rows(dataset, args.instruction_column)
    print(format_duplicate_report(duplicates, args.max_report))
    if args.dry:
        return 0
    deduplicated_dataset, removed_counts = deduplicate_dataset(dataset, args.instruction_column)
    removed_total = sum(removed_counts.values())
    output_path = args.output
    assert output_path is not None
    ensure_output_path_available(output_path)
    deduplicated_dataset.save_to_disk(str(output_path))
    print(f"Removed {removed_total} duplicate rows and saved dataset to '{output_path}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
