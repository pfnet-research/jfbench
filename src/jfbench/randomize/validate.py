from __future__ import annotations

import argparse
import asyncio
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from datasets import Dataset
from datasets import DatasetDict
from datasets import load_from_disk


def validate_randomization(
    before_dirs: Sequence[Path] | Path | None = None,
    after_dirs: Sequence[Path] | Path | None = None,
    *,
    before_dir: Path | None = None,
    after_dir: Path | None = None,
) -> None:
    before_arg: Sequence[Path] | Path | None = (
        before_dirs if before_dirs is not None else before_dir
    )
    after_arg: Sequence[Path] | Path | None = after_dirs if after_dirs is not None else after_dir
    if before_arg is None or after_arg is None:
        raise ValueError("Both before_dirs and after_dirs are required.")
    before_paths = _normalize_path_list(before_arg)
    after_paths = _normalize_path_list(after_arg)
    before_records = _load_records(before_paths)
    after_records = _load_records(after_paths)
    before_index = _index_by_id(before_records)
    after_index = _index_by_id(after_records)

    errors: list[str] = []
    before_ids = set(before_index)
    after_ids = set(after_index)
    missing_in_after = sorted(before_ids - after_ids)
    missing_in_before = sorted(after_ids - before_ids)
    if missing_in_after:
        errors.append(f"Missing records after randomization: {', '.join(missing_in_after)}")
    if missing_in_before:
        errors.append(f"Unexpected records after randomization: {', '.join(missing_in_before)}")

    shared_ids = sorted(before_ids & after_ids)
    for data_id in shared_ids:
        before = before_index[data_id]
        after = after_index[data_id]
        if _prompt_signature(before) != _prompt_signature(after):
            errors.append(f"Prompt mismatch for {data_id}")
        if _constraint_signature(before) != _constraint_signature(after):
            errors.append(f"Constraint types mismatch for {data_id}")
        if _evaluation_signature(before) != _evaluation_signature(after):
            errors.append(f"Constraint evaluations mismatch for {data_id}")

    if errors:
        joined = "\n".join(errors)
        raise ValueError(f"Validation failed:\n{joined}")

    print(f"Validated {len(shared_ids)} records successfully.")


def _normalize_path_list(value: Sequence[Path] | Path) -> list[Path]:
    if isinstance(value, Path):
        return [value]
    return [Path(item) for item in value]


def _load_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        dataset = load_from_disk(str(path))
        if isinstance(dataset, DatasetDict):
            for split_name in sorted(dataset.keys()):
                records.extend(dataset[split_name].to_list())
        elif isinstance(dataset, Dataset):
            records.extend(dataset.to_list())
        else:
            raise TypeError(f"Unsupported dataset type: {type(dataset)}")
    return records


def _index_by_id(records: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for idx, record in enumerate(records):
        data_id = str(record.get("data_id") or "").strip()
        if not data_id:
            raise ValueError(f"Record at position {idx} is missing data_id.")
        indexed[data_id] = record
    return indexed


def _prompt_signature(record: Mapping[str, Any]) -> str:
    document = str(record.get("prompt_document") or "").strip()
    if document:
        return document
    prompt = str(record.get("prompt") or "").strip()
    return prompt


def _constraint_signature(record: Mapping[str, Any]) -> tuple[str, ...]:
    constraint_types = record.get("constraint_types") or []
    return tuple(str(name).strip() for name in constraint_types)


def _evaluation_signature(record: Mapping[str, Any]) -> dict[str, dict[str, bool | None]]:
    signature: dict[str, dict[str, bool | None]] = {}
    base_evaluation = record.get("evaluation")
    base_signature = _normalize_evaluation_map(base_evaluation)
    if base_signature:
        signature["response"] = base_signature
    for section in ("chosen", "rejected"):
        section_data = record.get(section)
        if not isinstance(section_data, Mapping):
            continue
        section_eval = _normalize_evaluation_map(section_data.get("evaluation"))
        if section_eval:
            signature[section] = section_eval
    return signature


def _normalize_evaluation_map(value: Any) -> dict[str, bool | None]:
    if not isinstance(value, Mapping):
        return {}
    normalized: dict[str, bool | None] = {}
    for key, result in value.items():
        if isinstance(result, bool):
            normalized[str(key)] = result
        elif result is None:
            normalized[str(key)] = None
        else:
            normalized[str(key)] = bool(result)
    return normalized


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate randomized SFT/DPO datasets.")
    parser.add_argument(
        "--before",
        type=Path,
        required=True,
        nargs="+",
        help="Dataset directories before randomization.",
    )
    parser.add_argument(
        "--after",
        type=Path,
        required=True,
        nargs="+",
        help="Dataset directories after randomization.",
    )
    return parser.parse_args(argv)


async def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_randomization(
        before_dirs=args.before,
        after_dirs=args.after,
    )


if __name__ == "__main__":  # pragma: no cover - manual invocation
    asyncio.run(main())
