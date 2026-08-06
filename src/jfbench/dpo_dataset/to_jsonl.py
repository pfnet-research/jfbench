from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import random
import re
from typing import Any
from typing import Literal
from typing import Sequence

import datasets
from tqdm import tqdm
from transformers import AutoTokenizer
import yaml


def convert_to_jsonl(
    input_paths: Sequence[Path],
    output_path: Path,
    *,
    allow_overwrite: bool,
    instruction_column: str = "instruction",
    chosen_column: str = "chosen",
    rejected_column: str = "rejected",
    prompt_source_column: str = "prompt_source",
    chosen_reasoning_column: str = "chosen_reasoning",
    rejected_reasoning_column: str = "rejected_reasoning",
    sampling_ratio: float = 1.0,
    token_distribution_plot: Path | None = None,
    skip_tokenize: bool = False,
    tokenizer_model: Path = Path("pfnet/plamo-2-1b"),
) -> None:
    dataset_dicts: list[datasets.DatasetDict] = []
    for input_path in input_paths:
        dataset = datasets.load_from_disk(str(input_path))
        if isinstance(dataset, datasets.Dataset):
            dataset = datasets.DatasetDict({"train": dataset})
        dataset_dicts.append(dataset)

    if len(dataset_dicts) == 1:
        merged = dataset_dicts[0]
    else:
        combined: dict[str, datasets.Dataset] = {}
        for dataset in dataset_dicts:
            for split_name, split in dataset.items():
                if split_name in combined:
                    combined[split_name] = datasets.concatenate_datasets(
                        [combined[split_name], split]
                    )
                else:
                    combined[split_name] = split
        merged = datasets.DatasetDict(combined)

    plamo_tag_pattern = re.compile(r"<\|plamo:|:plamo\|>")

    def tag_filter(example: dict) -> bool:
        """Filters out examples whose `chosen`'s content contains plamo tags,
        which are not supposed to be present.
        """
        chosen_value = example.get(chosen_column)
        response_value = (
            chosen_value.get("response") if isinstance(chosen_value, Mapping) else chosen_value
        )
        content = _normalize_text(response_value)
        reasoning_content = _normalize_text(example.get(chosen_reasoning_column))
        if example.get(chosen_reasoning_column) is None:
            return False
        if plamo_tag_pattern.search(content):
            return False
        if plamo_tag_pattern.search(reasoning_content):
            return False
        return True

    merged = merged.filter(tag_filter)

    output_path.mkdir(parents=True, exist_ok=True)
    write_mode: Literal["w", "x"] = "w" if allow_overwrite else "x"

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
    empty_chosen_response = 0
    empty_rejected_response = 0
    empty_chosen_reasoning = 0
    empty_rejected_reasoning = 0
    total_tokens = 0
    chosen_token_counts: list[int] = []
    rejected_token_counts: list[int] = []

    for split_name, split in merged.items():
        chosen_reasoning_present = chosen_reasoning_column in split.column_names
        rejected_reasoning_present = rejected_reasoning_column in split.column_names
        total_loaded += len(split)
        with (output_path / f"{split_name}.jsonl").open(write_mode) as handle:
            progress = tqdm(
                split,
                total=len(split),
                desc=f"Converting {split_name}",
                unit="record",
            )
            for row in progress:
                if sampling_ratio < 1.0 and random.random() > sampling_ratio:
                    continue
                instruction = row.get(instruction_column)
                prompt = [
                    {
                        "role": "user",
                        "content": _normalize_text(instruction),
                    }
                ]
                chosen_response, chosen_reasoning = _extract_response_and_reasoning(
                    row.get(chosen_column),
                    row.get(chosen_reasoning_column),
                )
                rejected_response, rejected_reasoning = _extract_response_and_reasoning(
                    row.get(rejected_column),
                    row.get(rejected_reasoning_column),
                )
                instruction_empty = _is_empty(instruction)
                chosen_response_empty = _is_empty(chosen_response)
                rejected_response_empty = _is_empty(rejected_response)
                chosen_reasoning_empty = chosen_reasoning_present and _is_empty(chosen_reasoning)
                rejected_reasoning_empty = rejected_reasoning_present and _is_empty(
                    rejected_reasoning
                )
                if instruction_empty:
                    empty_instruction += 1
                if chosen_response_empty:
                    empty_chosen_response += 1
                if rejected_response_empty:
                    empty_rejected_response += 1
                if chosen_reasoning_empty:
                    empty_chosen_reasoning += 1
                if rejected_reasoning_empty:
                    empty_rejected_reasoning += 1
                if (
                    instruction_empty
                    or chosen_response_empty
                    or rejected_response_empty
                    or chosen_reasoning_empty
                    or rejected_reasoning_empty
                ):
                    continue
                if not skip_tokenize:
                    if tokenizer is None:
                        tokenizer = AutoTokenizer.from_pretrained(
                            tokenizer_model, use_fast=True, trust_remote_code=True
                        )
                    instruction_tokens = _count_tokens(_normalize_text(instruction))
                    chosen_response_tokens = _count_tokens(chosen_response)
                    rejected_response_tokens = _count_tokens(rejected_response)
                    chosen_reasoning_tokens = (
                        _count_tokens(chosen_reasoning)
                        if chosen_reasoning_present and chosen_reasoning is not None
                        else 0
                    )
                    rejected_reasoning_tokens = (
                        _count_tokens(rejected_reasoning)
                        if rejected_reasoning_present and rejected_reasoning is not None
                        else 0
                    )
                payload = {
                    "prompt_source": "",
                    "prompt": prompt,
                    "chosen": [
                        {
                            "role": "assistant",
                            "content": chosen_response,
                            "reasoning_content": chosen_reasoning,
                        }
                    ],
                    "rejected": [
                        {
                            "role": "assistant",
                            "content": rejected_response,
                            "reasoning_content": rejected_reasoning,
                        }
                    ],
                }
                prompt_source = row.get(prompt_source_column)
                if prompt_source is not None:
                    payload["prompt_source"] = prompt_source
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                handle.write("\n")
                if not skip_tokenize:
                    total_tokens += instruction_tokens
                    total_tokens += chosen_response_tokens
                    total_tokens += rejected_response_tokens
                    total_tokens += chosen_reasoning_tokens
                    total_tokens += rejected_reasoning_tokens
                    chosen_token_counts.append(
                        instruction_tokens + chosen_response_tokens + chosen_reasoning_tokens
                    )
                    rejected_token_counts.append(
                        instruction_tokens + rejected_response_tokens + rejected_reasoning_tokens
                    )
            progress.close()

    features = [
        {"name": "prompt_source", "dtype": "string"},
        {
            "name": "prompt",
            "list": [{"name": "content", "dtype": "string"}, {"name": "role", "dtype": "string"}],
        },
        {
            "name": "chosen",
            "list": [
                {"name": "content", "dtype": "string"},
                {"name": "reasoning_content", "dtype": "string"},
                {"name": "role", "dtype": "string"},
            ],
        },
        {
            "name": "rejected",
            "list": [
                {"name": "content", "dtype": "string"},
                {"name": "reasoning_content", "dtype": "string"},
                {"name": "role", "dtype": "string"},
            ],
        },
    ]
    with (output_path / "README.md").open(write_mode) as readme:
        readme.write("---\n")
        yaml.safe_dump({"dataset_info": {"features": features}}, readme, sort_keys=False)
        readme.write("---\n")
    if not skip_tokenize:
        token_distribution_plot_path = token_distribution_plot or (
            output_path / "token_count_distribution.html"
        )
        if chosen_token_counts and rejected_token_counts:
            _write_token_distribution_plot(
                token_distribution_plot_path,
                chosen_token_counts,
                rejected_token_counts,
                allow_overwrite=allow_overwrite,
            )
            _print_token_distribution_table(chosen_token_counts, rejected_token_counts)
    print(f"Loaded records: {total_loaded}")
    print(f"Empty instruction: {empty_instruction}")
    print(f"Empty chosen response: {empty_chosen_response}")
    print(f"Empty rejected response: {empty_rejected_response}")
    print(f"Empty chosen reasoning_content: {empty_chosen_reasoning}")
    print(f"Empty rejected reasoning_content: {empty_rejected_reasoning}")
    if not skip_tokenize:
        print(f"Total tokens: {total_tokens}")
        if chosen_token_counts and rejected_token_counts:
            print(f"Token distribution plot: {token_distribution_plot_path}")


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _extract_response_and_reasoning(value: Any, explicit_reasoning: Any) -> tuple[str, str]:
    response_value: Any = value
    reasoning_value: Any = explicit_reasoning
    if isinstance(value, Mapping):
        if reasoning_value is None:
            reasoning_value = value.get("reasoning_content")
        response_value = value.get("response")
    return _normalize_text(response_value), _normalize_text(reasoning_value)


def _write_token_distribution_plot(
    output_path: Path,
    chosen_token_counts: Sequence[int],
    rejected_token_counts: Sequence[int],
    *,
    allow_overwrite: bool,
) -> None:
    if output_path.exists() and not allow_overwrite:
        raise FileExistsError(f"Token distribution plot already exists: {output_path}")
    import plotly.graph_objects as go

    fig = go.Figure(
        data=[
            go.Histogram(x=chosen_token_counts, name="chosen", opacity=0.7),
            go.Histogram(x=rejected_token_counts, name="rejected", opacity=0.7),
        ]
    )
    fig.update_layout(
        barmode="overlay",
        title="Token Count Distribution",
        xaxis_title="Tokens per Record",
        yaxis_title="Count",
        legend_title="Response Type",
    )
    fig.write_html(str(output_path))


def _print_token_distribution_table(
    chosen_token_counts: Sequence[int],
    rejected_token_counts: Sequence[int],
) -> None:
    bins = [
        (1, 1000, "1-1k"),
        (1000, 2000, "1k-2k"),
        (2000, 4000, "2k-4k"),
        (4000, 8000, "4k-8k"),
        (8000, 16000, "8k-16k"),
        (16000, 32000, "16k-32k"),
        (32000, None, "32k+"),
    ]
    chosen_total = len(chosen_token_counts)
    rejected_total = len(rejected_token_counts)
    header = [
        "range",
        "chosen_count",
        "chosen_ratio",
        "rejected_count",
        "rejected_ratio",
    ]
    rows = [header]
    for lower, upper, label in bins:
        chosen_count = _count_in_bin(chosen_token_counts, lower, upper)
        rejected_count = _count_in_bin(rejected_token_counts, lower, upper)
        rows.append(
            [
                label,
                str(chosen_count),
                _format_ratio(chosen_count, chosen_total),
                str(rejected_count),
                _format_ratio(rejected_count, rejected_total),
            ]
        )
    print("Token count distribution table:")
    print(_render_table(rows))


def _count_in_bin(values: Sequence[int], lower: int, upper: int | None) -> int:
    if upper is None:
        return sum(1 for value in values if value >= lower)
    return sum(1 for value in values if lower <= value < upper)


def _format_ratio(count: int, total: int) -> str:
    if total == 0:
        return "0.00%"
    ratio = count / total * 100
    return f"{ratio:.2f}%"


def _render_table(rows: Sequence[Sequence[str]]) -> str:
    widths = [0] * len(rows[0])
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    lines: list[str] = []
    for idx, row in enumerate(rows):
        padded = [cell.ljust(widths[i]) for i, cell in enumerate(row)]
        lines.append(" | ".join(padded))
        if idx == 0:
            lines.append("-+-".join("-" * width for width in widths))
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input", type=Path, required=True, nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--instruction-column", type=str, default="instruction")
    parser.add_argument("--chosen-column", type=str, default="chosen")
    parser.add_argument("--rejected-column", type=str, default="rejected")
    parser.add_argument("--prompt-source-column", type=str, default="prompt_source")
    parser.add_argument("--chosen-reasoning-column", type=str, default="chosen_reasoning")
    parser.add_argument("--rejected-reasoning-column", type=str, default="rejected_reasoning")
    parser.add_argument("--sampling-ratio", type=float, default=1.0)
    parser.add_argument("--token-distribution-plot", type=Path)
    parser.add_argument("--skip-tokenize", action="store_true")
    parser.add_argument(
        "--tokenizer-model",
        type=Path,
        default=Path("pfnet/plamo-2-1b"),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    input_paths = args.input
    output_path = args.output or input_paths[0]
    convert_to_jsonl(
        input_paths=input_paths,
        output_path=output_path,
        allow_overwrite=args.allow_overwrite,
        instruction_column=args.instruction_column,
        chosen_column=args.chosen_column,
        rejected_column=args.rejected_column,
        prompt_source_column=args.prompt_source_column,
        chosen_reasoning_column=args.chosen_reasoning_column,
        rejected_reasoning_column=args.rejected_reasoning_column,
        sampling_ratio=args.sampling_ratio,
        token_distribution_plot=args.token_distribution_plot,
        skip_tokenize=args.skip_tokenize,
        tokenizer_model=args.tokenizer_model,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
