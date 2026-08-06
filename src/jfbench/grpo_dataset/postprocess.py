from __future__ import annotations

import argparse
from collections.abc import Mapping
import inspect
import json
from pathlib import Path
import shutil
from typing import Any
from typing import Sequence

import datasets

import jfbench.constraints as all_constraints


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Postprocess GRPO datasets by reconstructing constraints.")
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="Input HuggingFace dataset directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory for the postprocessed dataset. Defaults to '<first_input>_postprocessed'.",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Overwrite existing output directory.",
    )
    return parser.parse_args(argv)


def postprocess_datasets(
    input_paths: Sequence[Path],
    *,
    output_path: Path,
    allow_overwrite: bool = False,
) -> tuple[int, int]:
    dataset_dict = _load_as_dataset_dict(input_paths)
    processed: dict[str, datasets.Dataset] = {}
    total_loaded = 0
    total_written = 0
    for split_name, split in dataset_dict.items():
        rows = [dict(row) for row in split]
        total_loaded += len(rows)
        converted = [_postprocess_row(row) for row in rows]
        total_written += len(converted)
        processed[split_name] = datasets.Dataset.from_list(converted)

    output_dataset = datasets.DatasetDict(processed)
    _save_dataset(output_dataset, output_path, allow_overwrite=allow_overwrite)
    return total_loaded, total_written


def _load_as_dataset_dict(input_paths: Sequence[Path]) -> datasets.DatasetDict:
    merged: dict[str, datasets.Dataset] = {}
    for input_path in input_paths:
        loaded = datasets.load_from_disk(str(input_path))
        if isinstance(loaded, datasets.Dataset):
            current = datasets.DatasetDict({"train": loaded})
        else:
            current = loaded
        for split_name, split in current.items():
            if split_name in merged:
                merged[split_name] = datasets.concatenate_datasets([merged[split_name], split])
            else:
                merged[split_name] = split
    return datasets.DatasetDict(merged)


def _save_dataset(
    dataset: datasets.Dataset | datasets.DatasetDict,
    output_path: Path,
    *,
    allow_overwrite: bool,
) -> None:
    tmp_path = output_path.with_name(f"{output_path.name}_postprocess_tmp")
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    dataset.save_to_disk(str(tmp_path))
    if output_path.exists():
        if not allow_overwrite:
            raise FileExistsError(f"Output path already exists: {output_path}")
        shutil.rmtree(output_path)
    shutil.move(str(tmp_path), str(output_path))


def _postprocess_row(row: dict[str, Any]) -> dict[str, Any]:
    constraint_names = _as_string_list(row.get("constraint_types"))
    old_constraint_kwargs = _as_list(row.get("constraint_kwargs"))
    prompt_document = str(row.get("prompt_document") or "")
    seed = _parse_optional_int(row.get("seed"))

    constraints = [
        _instantiate_constraint_for_row(
            constraint_name,
            old_constraint_kwargs[idx] if idx < len(old_constraint_kwargs) else None,
            prompt_document=prompt_document,
            seed=seed,
        )
        for idx, constraint_name in enumerate(constraint_names)
    ]

    row["constraint_types"] = [constraint.__class__.__name__ for constraint in constraints]
    row["constraint_groups"] = [constraint.group for constraint in constraints]
    row["constraint_count"] = len(constraints)

    existing_instructions = _as_string_list(row.get("constraint_instructions"))
    if len(existing_instructions) != len(constraints):
        existing_instructions = [constraint.instructions() for constraint in constraints]
    row["constraint_instructions"] = existing_instructions

    row["constraints"] = [constraint.to_json() for constraint in constraints]
    row.pop("constraint_kwargs", None)
    return row


def _instantiate_constraint_for_row(
    constraint_name: str,
    legacy_kwargs_entry: Any,
    *,
    prompt_document: str,
    seed: int | None,
) -> Any:
    kwargs = _legacy_kwargs_for_constraint(constraint_name, legacy_kwargs_entry)
    constraint_cls = getattr(all_constraints, constraint_name, None)
    if constraint_cls is None:
        raise ValueError(f"Unknown constraint class: {constraint_name}")
    return _instantiate_constraint_class(
        constraint_cls,
        kwargs,
        prompt_document=prompt_document,
        seed=seed,
    )


def _instantiate_constraint_class(
    constraint_cls: type[Any],
    kwargs: dict[str, Any],
    *,
    prompt_document: str,
    seed: int | None,
) -> Any:
    signature = inspect.signature(constraint_cls.__init__)
    params = dict(kwargs)
    if "document" in signature.parameters and "document" not in params:
        params["document"] = prompt_document
    if "seed" in signature.parameters and "seed" not in params:
        params["seed"] = seed
    if "client" in signature.parameters and "client" not in params:
        raise ValueError(
            f"Constraint '{constraint_cls.__name__}' requires client, but no client is available."
        )
    return constraint_cls(**params)


def _legacy_kwargs_for_constraint(constraint_name: str, legacy_entry: Any) -> dict[str, Any]:
    if isinstance(legacy_entry, str):
        loaded = _try_json_load(legacy_entry)
        if isinstance(loaded, dict):
            return {key: _deserialize_legacy_value(value) for key, value in loaded.items()}
        return {}
    if not isinstance(legacy_entry, Mapping):
        return {}

    prefix = f"{_to_snake_case(constraint_name)}_"
    kwargs: dict[str, Any] = {}
    for key, value in legacy_entry.items():
        if not isinstance(key, str) or not key.startswith(prefix):
            continue
        if value is None:
            continue
        bare_key = key[len(prefix) :]
        kwargs[bare_key] = _deserialize_legacy_value(value)
    return kwargs


def _deserialize_legacy_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_deserialize_legacy_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _deserialize_legacy_value(item) for key, item in value.items()}
    if isinstance(value, str):
        loaded = _try_json_load(value)
        if (
            isinstance(loaded, Mapping)
            and isinstance(loaded.get("name"), str)
            and isinstance(loaded.get("kwargs"), Mapping)
        ):
            nested_name = str(loaded["name"])
            nested_kwargs_raw = dict(loaded["kwargs"])
            nested_kwargs = {
                str(key): _deserialize_legacy_value(item)
                for key, item in nested_kwargs_raw.items()
            }
            nested_cls = getattr(all_constraints, nested_name, None)
            if nested_cls is None:
                raise ValueError(f"Unknown nested constraint class: {nested_name}")
            return _instantiate_constraint_class(
                nested_cls,
                nested_kwargs,
                prompt_document="",
                seed=None,
            )
        return value
    return value


def _to_snake_case(name: str) -> str:
    out: list[str] = []
    for idx, char in enumerate(name):
        if char.isupper() and idx > 0 and (not name[idx - 1].isupper()):
            out.append("_")
        out.append(char.lower())
    return "".join(out)


def _try_json_load(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _parse_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_path = args.output
    if output_path is None:
        output_path = args.input[0].with_name(f"{args.input[0].name}_postprocessed")

    total_loaded, total_written = postprocess_datasets(
        args.input,
        output_path=output_path,
        allow_overwrite=args.allow_overwrite,
    )
    print(f"Loaded records: {total_loaded}")
    print(f"Written records: {total_written}")


if __name__ == "__main__":
    main()
