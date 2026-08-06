from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from typing import Literal

import datasets
import yaml


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input", type=Path, required=True, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--instruction-column", type=str, default="instruction")
    args = parser.parse_args()
    input_paths: list[Path] = args.input
    output_path: Path = args.output or input_paths[0]
    write_mode: Literal["w", "x"] = "w" if args.allow_overwrite else "x"
    output_path.mkdir(exist_ok=True)

    dataset_dicts: list[datasets.DatasetDict] = []
    for input_path in input_paths:
        dataset = datasets.load_from_disk(f"{input_path}/")
        if isinstance(dataset, datasets.Dataset):
            dataset = datasets.DatasetDict({"train": dataset})
        assert isinstance(dataset, datasets.DatasetDict)
        dataset_dicts.append(dataset)

    if len(dataset_dicts) == 1:
        dataset_dict = dataset_dicts[0]
    else:
        merged: dict[str, datasets.Dataset] = {}
        for dataset in dataset_dicts:
            for split_name, split in dataset.items():
                if split_name in merged:
                    merged[split_name] = datasets.concatenate_datasets([merged[split_name], split])
                else:
                    merged[split_name] = split
        dataset_dict = datasets.DatasetDict(merged)

    instruction_column = args.instruction_column
    total_loaded = 0
    total_removed = 0
    total_written = 0

    for split_name, split in dataset_dict.items():
        total_loaded += len(split)
        with (output_path / f"{split_name}.jsonl").open(write_mode) as handle:
            for row in split:
                instruction = row[instruction_column]
                response = row.get("response")
                reasoning_content = row.get("reasoning_content")
                if _is_empty(instruction) or _is_empty(response) or _is_empty(reasoning_content):
                    total_removed += 1
                    continue
                payload = {
                    "prompt": [{"role": "user", "content": instruction}],
                    "metadata": json.dumps(
                        dict(row), ensure_ascii=False, allow_nan=False, separators=(",", ":")
                    ),
                }
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                handle.write("\n")
                total_written += 1
    selected_features = [
        {
            "name": "prompt",
            "list": [{"name": "content", "dtype": "string"}, {"name": "role", "dtype": "string"}],
        },
        {"name": "metadata", "dtype": "string"},
    ]
    with (output_path / "README.md").open(write_mode) as handle:
        handle.write("---\n")
        yaml.safe_dump({"dataset_info": {"features": selected_features}}, handle, sort_keys=False)
        handle.write("---\n")
    print(f"Loaded records: {total_loaded}")
    print(f"Removed records: {total_removed}")
    print(f"Written records: {total_written}")


if __name__ == "__main__":
    main()
