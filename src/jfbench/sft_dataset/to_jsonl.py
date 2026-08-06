import argparse
import json
from pathlib import Path
from typing import Any
from typing import Literal

import datasets
from transformers import AutoTokenizer
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input", type=Path, required=True, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--instruction-column", type=str, default="instruction")
    parser.add_argument("--response-column", type=str, default="response")
    parser.add_argument("--reasoning-content-column", type=str, default="reasoning_content")
    parser.add_argument(
        "--tokenizer-model",
        type=Path,
        default=Path("pfnet/plamo-2-1b"),
    )
    args = parser.parse_args()
    input_paths: list[Path] = args.input
    output_path: Path = args.output or input_paths[0]
    write_mode: Literal["w", "x"] = "w" if args.allow_overwrite else "x"
    output_path.mkdir(exist_ok=True)
    # assert not list(output_path.glob("*.jsonl*"))
    # print(input_path, output_path)

    dataset_dicts: list[datasets.DatasetDict] = []
    for input_path in input_paths:
        dataset = datasets.load_from_disk(f"{input_path}/")
        if isinstance(dataset, datasets.Dataset):
            dataset = datasets.DatasetDict({"train": dataset})
        assert isinstance(dataset, datasets.DatasetDict)
        dataset_dicts.append(dataset)

    if len(dataset_dicts) == 1:
        d = dataset_dicts[0]
    else:
        merged: dict[str, datasets.Dataset] = {}
        for dataset in dataset_dicts:
            for split_name, split in dataset.items():
                if split_name in merged:
                    merged[split_name] = datasets.concatenate_datasets([merged[split_name], split])
                else:
                    merged[split_name] = split
        d = datasets.DatasetDict(merged)

    instruction_column = args.instruction_column
    response_column = args.response_column
    reasoning_content_column = args.reasoning_content_column

    def _build_chat(row: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        return {
            "chat": [
                {"role": "user", "content": row[instruction_column]},
                {
                    "role": "assistant",
                    "content": row[response_column],
                    "reasoning_content": row.get(reasoning_content_column),
                },
            ]
        }

    d = d.map(_build_chat)

    def _is_empty(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str):
            return value.strip() == ""
        return False

    def _count_tokens(text: str) -> int:
        if tokenizer is None:
            return 0
        return len(tokenizer.encode(text, add_special_tokens=False))

    tokenizer = None
    total_loaded = 0
    empty_instruction = 0
    empty_response = 0
    empty_reasoning_content = 0
    total_tokens = 0
    tokenizer_model = args.tokenizer_model

    for split_name, split in d.items():
        reasoning_present = reasoning_content_column in split.column_names
        total_loaded += len(split)
        with (output_path / f"{split_name}.jsonl").open(write_mode) as f:
            for row in split:
                instruction = row[instruction_column]
                response = row[response_column]
                reasoning_content = row.get(reasoning_content_column)
                instruction_empty = _is_empty(instruction)
                response_empty = _is_empty(response)
                reasoning_empty = reasoning_present and _is_empty(reasoning_content)
                if instruction_empty:
                    empty_instruction += 1
                if response_empty:
                    empty_response += 1
                if reasoning_empty:
                    empty_reasoning_content += 1
                if instruction_empty or response_empty or reasoning_empty:
                    continue
                if tokenizer is None:
                    tokenizer = AutoTokenizer.from_pretrained(
                        tokenizer_model, use_fast=True, trust_remote_code=True
                    )
                chat = [
                    {key: value for key, value in message.items() if value is not None}
                    for message in row["chat"]
                ]
                if len(chat) != 2:
                    raise
                payload: dict[str, Any] = {"chat": chat}
                prompt_source = row.get("prompt_source")
                if prompt_source is not None:
                    payload["prompt_source"] = prompt_source
                json.dump(
                    payload,
                    f,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                f.write("\n")
                total_tokens += _count_tokens(instruction)
                total_tokens += _count_tokens(response)
                if reasoning_present and reasoning_content is not None:
                    total_tokens += _count_tokens(reasoning_content)
        # assert len(list(output_path.glob(f"*{split_name}*"))) == 1
    features = d["train"].features._to_yaml_list()  # noqa: SLF001
    _features = []
    for feature in features:
        if feature["name"] == "prompt_source" or feature["name"] == "chat":
            _features.append(feature)
    with (output_path / "README.md").open(write_mode) as f:
        f.write("---\n")
        yaml.safe_dump({"dataset_info": {"features": _features}}, f, sort_keys=False)
        f.write("---\n")
    print(f"Loaded records: {total_loaded}")
    print(f"Empty instruction: {empty_instruction}")
    print(f"Empty response: {empty_response}")
    print(f"Empty reasoning_content: {empty_reasoning_content}")
    print(f"Total tokens: {total_tokens}")


if __name__ == "__main__":
    main()
