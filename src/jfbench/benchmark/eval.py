import argparse
import asyncio
from dataclasses import dataclass
import inspect
import io
import json
from pathlib import Path
import random
from typing import Any
from typing import Awaitable
from typing import cast
from typing import Iterable
from typing import Literal
from typing import Sequence
from typing import TYPE_CHECKING
from typing import TypeVar


if TYPE_CHECKING:
    import torch
else:
    from jfbench.imports import LazyImport

    torch = LazyImport("torch")

import datasets
import pandas as pd
from tqdm import tqdm
import zstandard as zstd

from jfbench.benchmark.build import BenchmarkData
from jfbench.benchmark.build import ConstraintSetName
from jfbench.benchmark.build import get_ifbench_benchmark_data
from jfbench.benchmark.build import get_ifbench_benchmark_data_with_multiple_constraints
from jfbench.benchmark.build import MetaData
from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.llm import extract_reasoning_content
from jfbench.llm import LLMClient
from jfbench.protocol import Constraint


MAX_CONCURRENT_EVALUATIONS = 200
DEFAULT_OUTPUT_DIR = Path("data/benchmark_results")
T = TypeVar("T")


@dataclass
class ModelConfig:
    provider: Literal["openrouter", "vllm"]
    model: str
    model_short: str
    extra_body: dict[str, Any] | None = None


DEFAULT_JUDGE_MODEL_SPEC = ModelConfig(
    provider="openrouter",
    model="openai/gpt-oss-120b",
    model_short="gpt-oss-120b",
    extra_body={"reasoning": {"effort": "medium"}},
)


class _LoadedPrompt:
    def __init__(self, prompt_text: str, prompt_document: str) -> None:
        self._prompt_text = prompt_text
        self._prompt_document = prompt_document

    def text(self, constraints: list[Any], *, train_or_test: str = "train") -> str:
        _ = (constraints, train_or_test)
        return self._prompt_text

    @property
    def document(self) -> str:
        return self._prompt_document


def _parse_model_specs_json(value: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("model_specs_json must be valid JSON.") from exc
    if not isinstance(parsed, list):
        raise ValueError("model_specs_json must be a JSON array of objects.")
    specs: list[dict[str, Any]] = []
    for index, spec in enumerate(parsed):
        if not isinstance(spec, dict):
            raise ValueError(f"Each model spec must be an object. Invalid entry at index {index}.")
        specs.append(spec)
    if not specs:
        raise ValueError("model_specs_json must not be empty.")
    return specs


def _instantiate_model_configs(specs: Sequence[dict[str, Any]]) -> list[ModelConfig]:
    return [ModelConfig(**spec) for spec in specs]


def _select_judge_config(judge_model_spec_json: str | None) -> ModelConfig | None:
    if judge_model_spec_json is None:
        return None
    try:
        parsed = json.loads(judge_model_spec_json)
    except json.JSONDecodeError as exc:
        raise ValueError("judge_model_spec_json must be valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("judge_model_spec_json must be a JSON object.")
    return ModelConfig(**parsed)


def _sanitize_filename_component(value: str) -> str:
    text = value.strip()
    if not text:
        return "unknown"
    sanitized = text.replace("/", "-").replace("\\", "-")
    sanitized = sanitized.replace(" ", "_")
    return sanitized


def _result_file_path(
    output_dir: Path,
    benchmark: str,
    n_constraints: int,
    model_short: str,
    constraint_set: ConstraintSetName,
) -> Path:
    safe_benchmark = _sanitize_filename_component(benchmark)
    safe_model_short = _sanitize_filename_component(model_short)
    safe_constraint_set = _sanitize_filename_component(constraint_set)
    return (
        output_dir
        / f"{safe_benchmark}-{safe_constraint_set}-{n_constraints}-{safe_model_short}.jsonl"
    )


def _build_dataset(
    benchmark: str,
    n_constraints: int,
    n_benchmark_data: int | None,
    seed: int,
    constraint_set: ConstraintSetName,
    judge_config: ModelConfig | None = None,
    ifbench_dataset_path: str | None = None,
) -> list[BenchmarkData]:
    active_judge_config = judge_config or DEFAULT_JUDGE_MODEL_SPEC
    judge_client = LLMClient(
        provider=active_judge_config.provider,
        model=active_judge_config.model,
        extra_body=active_judge_config.extra_body,
    )
    if benchmark == "ifbench":
        if n_constraints == 1:
            data = list(
                get_ifbench_benchmark_data(
                    judge_client,
                    seed=seed,
                    constraint_set=constraint_set,
                    dataset_path=ifbench_dataset_path,
                )
            )
            if n_benchmark_data is not None:
                data = data[:n_benchmark_data]
        else:
            if n_benchmark_data is None:
                raise ValueError("When n_constraints > 1, n_benchmark_data must be specified.")
            data = list(
                get_ifbench_benchmark_data_with_multiple_constraints(
                    judge_client,
                    n_constraints=n_constraints,
                    n_benchmark_data=n_benchmark_data,
                    seed=seed,
                    constraint_set=constraint_set,
                    dataset_path=ifbench_dataset_path,
                )
            )
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    return data


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _constraint_from_saved_entry(saved_entry: Any) -> Constraint:
    if isinstance(saved_entry, str):
        return cast("Constraint", ConstraintGroupMixin.from_json(saved_entry))
    if isinstance(saved_entry, dict):
        return cast("Constraint", ConstraintGroupMixin.from_serializable_dict(saved_entry))
    raise ValueError(f"Constraint entry must be dict or JSON string, got: {type(saved_entry)}")


def _replace_llm_client_references(
    value: Any,
    replacement: LLMClient,
    seen: set[int],
) -> Any:
    if isinstance(value, LLMClient):
        return replacement
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    if isinstance(value, bytes):
        return value
    value_id = id(value)
    if value_id in seen:
        return value
    seen.add(value_id)
    if isinstance(value, list):
        for idx, item in enumerate(value):
            value[idx] = _replace_llm_client_references(item, replacement, seen)
        return value
    if isinstance(value, tuple):
        return tuple(_replace_llm_client_references(item, replacement, seen) for item in value)
    if isinstance(value, set):
        return {_replace_llm_client_references(item, replacement, seen) for item in value}
    if isinstance(value, dict):
        for key in list(value.keys()):
            value[key] = _replace_llm_client_references(value[key], replacement, seen)
        return value
    if hasattr(value, "__dict__"):
        for attr_name, attr_value in vars(value).items():
            replaced = _replace_llm_client_references(attr_value, replacement, seen)
            if replaced is not attr_value:
                setattr(value, attr_name, replaced)
        return value
    return value


def _apply_judge_config_override(
    dataset: list[BenchmarkData],
    judge_config: ModelConfig | None,
) -> list[BenchmarkData]:
    if judge_config is None:
        return dataset
    judge_client = LLMClient(
        provider=judge_config.provider,
        model=judge_config.model,
        extra_body=judge_config.extra_body,
    )
    for benchmark_data in dataset:
        updated_constraints: list[Constraint] = []
        for constraint in benchmark_data.constraints:
            replaced = _replace_llm_client_references(constraint, judge_client, seen=set())
            updated_constraints.append(cast("Constraint", replaced))
        benchmark_data.constraints = updated_constraints
    return dataset


def _load_dataset_from_path(
    dataset_path: str,
    constraint_set: ConstraintSetName,
    judge_config: ModelConfig | None = None,
) -> list[BenchmarkData]:
    _ = judge_config
    path = Path(dataset_path)
    rows: list[dict[str, Any]]
    if path.is_dir():
        test_path = path / "test.jsonl.zst"
        if test_path.is_file():
            rows = _load_rows_from_jsonl_zst(test_path)
        else:
            zst_files = sorted(file_path for file_path in path.glob("*.jsonl.zst"))
            if zst_files:
                rows = []
                for zst_file in zst_files:
                    rows.extend(_load_rows_from_jsonl_zst(zst_file))
            else:
                rows = _load_rows_from_hf_dataset(dataset_path)
    else:
        rows = _load_rows_from_hf_dataset(dataset_path)

    dataset: list[BenchmarkData] = []
    for row in rows:
        prompt_document = str(row.get("prompt_document") or "")
        constraint_names = _as_string_list(row.get("constraint_types"))
        saved_constraints = _as_list(row.get("constraints"))
        if constraint_names and len(saved_constraints) != len(constraint_names):
            raise ValueError(
                "constraints and constraint_types length mismatch. "
                f"constraints={len(saved_constraints)}, constraint_types={len(constraint_names)}"
            )
        constraints: list[Any] = []
        instructions = _as_string_list(row.get("constraint_instructions"))
        for index, saved_entry in enumerate(saved_constraints):
            try:
                constraint = _constraint_from_saved_entry(saved_entry)
            except Exception as exc:
                raise ValueError(
                    "Failed to deserialize constraint entry. "
                    "Please ensure constraints are serialized via to_json()."
                ) from exc
            if constraint_names and constraint.__class__.__name__ != constraint_names[index]:
                raise ValueError(
                    f"Constraint name mismatch: constraint_types has {constraint_names[index]}, "
                    f"but constraints entry has {constraint.__class__.__name__}."
                )
            constraints.append(constraint)
        row_constraint_set_raw = row.get("constraint_set")
        if row_constraint_set_raw in {"train", "test"}:
            row_constraint_set = cast("ConstraintSetName", row_constraint_set_raw)
        else:
            row_constraint_set = constraint_set
        if len(instructions) != len(constraints):
            instructions = [
                constraint.instructions(train_or_test=row_constraint_set)
                for constraint in constraints
            ]
        prompt_text = str(row.get("prompt") or "")
        prompt_source = str(row.get("prompt_source") or "")
        data_id = str(row.get("data_id") or row.get("prompt_id") or "")
        meta_data = MetaData(
            prompt_source=prompt_source,
            data_id=data_id,
            n_constraints=len(constraints),
            constraint_types=[constraint.__class__.__name__ for constraint in constraints],
            constraint_groups=[constraint.group for constraint in constraints],
            constraint_instructions=instructions,
            prompt=prompt_text,
            constraint_set=row_constraint_set,
        )
        dataset.append(
            BenchmarkData(
                prompt=_LoadedPrompt(prompt_text=prompt_text, prompt_document=prompt_document),
                constraints=constraints,
                meta_data=meta_data,
            )
        )
    return _apply_judge_config_override(dataset, judge_config)


def _load_rows_from_jsonl_zst(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    decompressor = zstd.ZstdDecompressor()
    with path.open("rb") as raw:
        with decompressor.stream_reader(raw) as reader:
            with io.TextIOWrapper(reader, encoding="utf-8") as text_stream:
                for line in text_stream:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    record = json.loads(stripped)
                    if isinstance(record, dict):
                        rows.append(record)
    return rows


def _load_rows_from_hf_dataset(dataset_path: str) -> list[dict[str, Any]]:
    loaded = datasets.load_from_disk(dataset_path)
    rows: list[dict[str, Any]] = []
    if isinstance(loaded, datasets.DatasetDict):
        for split in loaded.values():
            rows.extend(dict(row) for row in split)
        return rows
    if isinstance(loaded, datasets.Dataset):
        return [dict(row) for row in loaded]
    return [dict(row) for row in loaded]


def _filter_loaded_dataset(
    dataset_list: Sequence[BenchmarkData],
    benchmark: str,
    n_constraints: int,
    n_benchmark_data: int | None,
    seed: int,
) -> list[BenchmarkData]:
    filtered = [
        item
        for item in dataset_list
        if item.meta_data.prompt_source == benchmark
        and item.meta_data.n_constraints == n_constraints
    ]
    return _shuffle_and_slice_dataset(filtered, n_benchmark_data, seed)


def _load_results(results_path: Path) -> pd.DataFrame:
    if not results_path.exists() or results_path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_json(results_path, lines=True)
    except ValueError:
        return pd.DataFrame()


def _parse_comma_separated(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _normalize_dataset_paths(
    dataset_path: str | Sequence[str] | None,
) -> list[str] | None:
    dataset_paths: list[str] = []
    if isinstance(dataset_path, str):
        dataset_paths.extend(_parse_comma_separated(dataset_path))
    elif dataset_path is not None:
        for item in dataset_path:
            dataset_paths.extend(_parse_comma_separated(str(item)))
    if not dataset_paths:
        return None
    return dataset_paths


def _normalize_benchmark_list(benchmark: str | Sequence[str]) -> list[str]:
    if isinstance(benchmark, str):
        candidates = _parse_comma_separated(benchmark)
    else:
        candidates = [item.strip() for item in benchmark if str(item).strip()]
    if not candidates:
        raise ValueError("No benchmarks specified.")
    allowed = {"ifbench"}
    for candidate in candidates:
        if candidate not in allowed:
            raise ValueError(
                f"Unsupported benchmark: {candidate}. Allowed values are {sorted(allowed)}."
            )
    return candidates


def _normalize_n_constraints_list(
    n_constraints: int | str | Sequence[int],
) -> list[int]:
    if isinstance(n_constraints, int):
        values = [n_constraints]
    elif isinstance(n_constraints, str):
        str_items = _parse_comma_separated(n_constraints)
        if not str_items:
            raise ValueError("No n-constraints values specified.")
        values = []
        for item in str_items:
            try:
                values.append(int(item))
            except ValueError as exc:
                raise ValueError(f"Invalid n-constraints value: {item}") from exc
    else:
        values = [int(item) for item in n_constraints]
    if not values:
        raise ValueError("No n-constraints values specified.")
    return values


def _dataset_filter_values(
    dataset_list: Sequence[BenchmarkData],
) -> tuple[set[str], set[int]]:
    prompt_sources = {
        str(getattr(item.meta_data, "prompt_source", "")).strip()
        for item in dataset_list
        if getattr(item.meta_data, "prompt_source", None)
    }
    n_constraints_values = {
        int(getattr(item.meta_data, "n_constraints", 0))
        for item in dataset_list
        if getattr(item.meta_data, "n_constraints", None) is not None
    }
    return prompt_sources, n_constraints_values


def _filter_results_by_dataset(
    frame: pd.DataFrame,
    prompt_sources: set[str],
    n_constraints_values: set[int],
) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.loc[
        frame["prompt_source"].isin(prompt_sources)
        & frame["n_constraints"].isin(n_constraints_values)
    ]


def _shuffle_and_slice_dataset(
    dataset: Iterable[T],
    limit: int | None,
    seed: int,
) -> list[T]:
    dataset_list = list(dataset)
    if limit is None:
        return dataset_list
    shuffled = list(dataset_list)
    random.Random(seed).shuffle(shuffled)
    return shuffled[:limit]


def _collect_pending_generation(
    config: ModelConfig,
    dataset_list: list[BenchmarkData],
    override: bool,
    results_path: Path,
) -> list[tuple[int, BenchmarkData]]:
    prompt_sources, n_constraints_values = _dataset_filter_values(dataset_list)
    existing_results = _load_results(results_path)
    completed_indices: set[int] = set()
    if not existing_results.empty:
        mask = existing_results["model_short"] == config.model_short
        model_results = existing_results.loc[mask]
        model_results = _filter_results_by_dataset(
            model_results,
            prompt_sources,
            n_constraints_values,
        )
        completed_indices = set(model_results["prompt_index"].astype(int))
    if override:
        completed_indices = set()
    pending_items = [
        (index, benchmark_data)
        for index, benchmark_data in enumerate(dataset_list)
        if index not in completed_indices
    ]
    skip_count = len(dataset_list) - len(pending_items)
    if skip_count:
        print(
            f"Skipping {skip_count} existing entries for model {config.model} ({config.model_short})."
        )
    if not pending_items:
        print(f"No new prompts to generate for model: {config.model} ({config.model_short})")
    return pending_items


def _collect_pending_evaluations(
    config: ModelConfig,
    dataset_list: list[BenchmarkData],
    override: bool,
    results_path: Path,
) -> list[tuple[int, BenchmarkData, str, Any | None]]:
    if not dataset_list:
        return []
    prompt_sources, n_constraints_values = _dataset_filter_values(dataset_list)
    existing_results = _load_results(results_path)
    if existing_results.empty:
        return []
    mask = existing_results["model_short"] == config.model_short
    model_results = existing_results.loc[mask]
    model_results = _filter_results_by_dataset(
        model_results,
        prompt_sources,
        n_constraints_values,
    )
    if model_results.empty:
        return []
    if override:
        pending_eval = model_results
    else:
        pending_eval = model_results.loc[model_results["results"].isna()]
    evaluation_items: list[tuple[int, BenchmarkData, str, Any | None]] = []
    for row in pending_eval.itertuples():
        if pd.isna(row.response):
            continue
        index = int(row.prompt_index)
        if 0 <= index < len(dataset_list):
            response_details = getattr(row, "response_details", None)
            try:
                if pd.isna(response_details):
                    response_details = None
            except Exception:
                pass
            evaluation_items.append((index, dataset_list[index], row.response, response_details))
    return evaluation_items


async def _evaluate_entries(
    config: ModelConfig,
    evaluation_items: list[tuple[int, BenchmarkData, str, Any | None]],
) -> list[dict[str, Any]]:
    async def _evaluate_entry(
        index: int,
        benchmark_data: BenchmarkData,
        response: str,
        response_details: Any | None,
    ) -> dict[str, Any]:
        evaluation = benchmark_data.evaluate(response)
        if inspect.isawaitable(evaluation):
            awaitable = cast("Awaitable[dict[str, bool]]", evaluation)
            evaluation_dict = await awaitable
        else:
            evaluation_dict = cast("dict[str, bool]", evaluation)
        reasoning_content = extract_reasoning_content(config.provider, response_details)
        return {
            "model": config.model,
            "model_short": config.model_short,
            "prompt_index": index,
            "response": response,
            "response_details": response_details,
            "reasoning_content": reasoning_content,
            "results": evaluation_dict,
            **benchmark_data.meta_data.__dict__,
        }

    tasks = [
        _evaluate_entry(index, benchmark_data, response, response_details)
        for index, benchmark_data, response, response_details in evaluation_items
    ]
    entries: list[dict[str, Any]] = []
    if tasks:
        progress = tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"Evaluating {config.model}",
            leave=False,
        )
        try:
            for coro in progress:
                entries.append(await coro)
        finally:
            progress.close()
        entries.sort(key=lambda entry: entry["prompt_index"])
    return entries


async def _generate_responses(
    client: LLMClient,
    config: ModelConfig,
    pending_items: list[tuple[int, BenchmarkData]],
    n_concurrent_generations: int = -1,
) -> list[dict[str, Any]]:
    prompts = [benchmark_data.text() for _, benchmark_data in pending_items]
    if not prompts:
        return []

    batch_size = len(prompts)
    if n_concurrent_generations > 0:
        batch_size = min(n_concurrent_generations, len(prompts))

    responses: list[str] = []
    response_details: list[Any] = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        if config.provider != "openrouter" and config.provider != "vllm":
            raise ValueError(f"Unsupported provider: {config.provider}")
        batch_responses, batch_details = await client.async_ask(batch_prompts, use_tqdm=True)
        responses.extend(batch_responses)
        response_details.extend(batch_details)

    if len(responses) != len(pending_items):
        raise RuntimeError("Number of responses does not match number of pending prompts.")
    if len(response_details) != len(pending_items):
        raise RuntimeError("Number of response details does not match number of pending prompts.")

    return [
        {
            "model": config.model,
            "model_short": config.model_short,
            "prompt_index": index,
            "response": response,
            "response_details": detail,
            "reasoning_content": extract_reasoning_content(config.provider, detail),
            "results": None,
            **benchmark_data.meta_data.__dict__,
        }
        for (index, benchmark_data), response, detail in zip(
            pending_items, responses, response_details, strict=True
        )
    ]


def _upsert_entries(results_path: Path, entries: list[dict[str, Any]]) -> None:
    if not entries:
        return
    results_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_results(results_path)
    new_df = pd.DataFrame(entries)
    if existing.empty:
        merged = new_df
    else:
        merged = pd.concat([existing, new_df], ignore_index=True)
    required_columns = {"model_short", "prompt_index", "prompt_source", "n_constraints"}
    if required_columns <= set(merged.columns):
        merged["prompt_index"] = merged["prompt_index"].astype(int)
        subset = ["model_short", "prompt_index", "prompt_source", "n_constraints"]
        sort_by = list(subset)
        merged = merged.drop_duplicates(subset=subset, keep="last")
        merged = merged.sort_values(by=sort_by).reset_index(drop=True)
    json_lines = merged.to_json(orient="records", lines=True, index=False)
    if json_lines:
        results_path.write_text(json_lines, encoding="utf-8")


async def evaluate_model(
    dataset: Iterable[BenchmarkData],
    config: ModelConfig,
    with_generate: bool,
    with_eval: bool,
    override: bool,
    results_path: Path,
    client: LLMClient | None,
    n_concurrent_generations: int = -1,
) -> LLMClient | None:
    dataset_list = list(dataset)

    pending_items: list[tuple[int, BenchmarkData]] = []
    if with_generate:
        pending_items = _collect_pending_generation(config, dataset_list, override, results_path)
        if pending_items:
            if client is None:
                client = LLMClient(
                    provider=config.provider,
                    model=config.model,
                    extra_body=config.extra_body,
                )
            generated_entries = await _generate_responses(
                client,
                config,
                pending_items,
                n_concurrent_generations=n_concurrent_generations,
            )
            _upsert_entries(results_path, generated_entries)

    if with_eval:
        evaluation_items = _collect_pending_evaluations(
            config,
            dataset_list,
            override,
            results_path,
        )
        if evaluation_items:
            evaluated_entries = await _evaluate_entries(config, evaluation_items)
            _upsert_entries(results_path, evaluated_entries)
    return client


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark evaluation.")
    parser.add_argument(
        "--benchmark",
        type=str,
        default="ifbench",
        help="Benchmark dataset to use. Only ifbench is supported.",
    )
    parser.add_argument(
        "--ifbench-dataset-path",
        type=str,
        default=None,
        help="Optional path to the IFBench dataset JSONL. Defaults to the bundled dataset when omitted.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        action="append",
        default=None,
        help=(
            "Optional path to a prebuilt benchmark dataset directory or .jsonl.zst file. "
            "Specify this flag multiple times to load and merge multiple paths."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory where benchmark results will be written. Each benchmark/model combination "
            "will be stored as its own JSONL file."
        ),
    )
    parser.add_argument(
        "--with-generate",
        dest="with_generate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable running the generation step.",
    )
    parser.add_argument(
        "--with-eval",
        dest="with_eval",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable running the evaluation step.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Re-run generation and evaluation even if entries already exist.",
    )
    parser.add_argument(
        "--n-constraints",
        type=str,
        default="1",
        help="Number of constraints to apply. Provide comma-separated values to run multiple counts.",
    )
    parser.add_argument(
        "--constraint-set",
        type=str,
        choices=["train", "test"],
        default="test",
        help="Choose whether to build with the train or test constraint set.",
    )
    parser.add_argument(
        "--n-benchmark-data",
        type=int,
        default=None,
        help=(
            "Number of benchmark data entries to use. If not set, use all available entries when "
            "n_constraints is 1. If n_constraints > 1, you must set this value."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--model-specs-json",
        type=str,
        required=True,
        help="JSON string describing the models to evaluate.",
    )
    parser.add_argument(
        "--judge-model-spec-json",
        type=str,
        default=None,
        help="JSON string describing the model to use for judge_client.",
    )
    parser.add_argument(
        "--n-concurrent-generations",
        type=int,
        default=-1,
        help=(
            "Number of prompts to send concurrently to ask/async_ask. "
            "Use -1 to send all prompts at once."
        ),
    )
    return parser.parse_args(argv)


async def main(
    benchmark: str | Sequence[str] = "ifbench",
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    with_generate: bool = True,
    with_eval: bool = True,
    override: bool = False,
    n_constraints: int | str | Sequence[int] = 1,
    n_benchmark_data: int | None = None,
    seed: int = 42,
    model_specs_json: str | None = None,
    judge_model_spec_json: str | None = None,
    constraint_set: ConstraintSetName = "train",
    ifbench_dataset_path: str | None = None,
    dataset_path: str | Sequence[str] | None = None,
    n_concurrent_generations: int = -1,
) -> None:
    if model_specs_json is not None:
        custom_specs = _parse_model_specs_json(model_specs_json)
        configs = _instantiate_model_configs(custom_specs)
    else:
        raise ValueError("model_specs_json is required.")
    judge_config = _select_judge_config(judge_model_spec_json)
    benchmark_list = _normalize_benchmark_list(benchmark)
    n_constraints_list = _normalize_n_constraints_list(n_constraints)
    dataset_paths = _normalize_dataset_paths(dataset_path)
    output_dir_path = Path(output_dir)
    benchmark_combinations = [
        (benchmark_name, n_constraints_value)
        for benchmark_name in benchmark_list
        for n_constraints_value in n_constraints_list
    ]
    # Build datasets
    datasets: dict[tuple[str, int], list[BenchmarkData]] = {}
    if dataset_paths is None:

        async def _build_single_dataset(
            benchmark_name: str,
            n_constraints_value: int,
        ) -> tuple[tuple[str, int], list[BenchmarkData]]:
            print(
                f"Building benchmark: {benchmark_name} with {n_constraints_value} constraints "
                f"using {constraint_set} constraint set."
            )
            dataset = await asyncio.to_thread(
                _build_dataset,
                benchmark=benchmark_name,
                n_constraints=n_constraints_value,
                n_benchmark_data=n_benchmark_data,
                seed=seed,
                constraint_set=constraint_set,
                judge_config=judge_config,
                ifbench_dataset_path=ifbench_dataset_path,
            )
            return (benchmark_name, n_constraints_value), dataset

        dataset_results = await asyncio.gather(
            *[
                _build_single_dataset(benchmark_name, n_constraints_value)
                for benchmark_name, n_constraints_value in benchmark_combinations
            ]
        )
        datasets = {combo: dataset for combo, dataset in dataset_results}
    else:
        loaded_dataset: list[BenchmarkData] = []
        for path in dataset_paths:
            loaded_dataset.extend(
                _load_dataset_from_path(
                    dataset_path=path,
                    constraint_set=constraint_set,
                    judge_config=judge_config,
                )
            )
        datasets = {
            (benchmark_name, n_constraints_value): _filter_loaded_dataset(
                dataset_list=loaded_dataset,
                benchmark=benchmark_name,
                n_constraints=n_constraints_value,
                n_benchmark_data=n_benchmark_data,
                seed=seed,
            )
            for benchmark_name, n_constraints_value in benchmark_combinations
        }
    for config in configs:
        print(f"Selected model: {config.model} ({config.model_short})")

        async def _run_single_combination(
            benchmark_name: str,
            n_constraints_value: int,
        ) -> LLMClient | None:
            print(
                f"Running benchmark: {benchmark_name} with {n_constraints_value} constraints "
                f"using {constraint_set} constraint set."
            )
            dataset = datasets[(benchmark_name, n_constraints_value)]
            print(f"Loaded {len(dataset)} benchmark data entries.")
            results_path = _result_file_path(
                output_dir_path,
                benchmark_name,
                n_constraints_value,
                config.model_short,
                constraint_set,
            )
            client = await evaluate_model(
                dataset=dataset,
                config=config,
                with_generate=with_generate,
                with_eval=with_eval,
                override=override,
                results_path=results_path,
                client=None,
                n_concurrent_generations=n_concurrent_generations,
            )
            print(
                f"Evaluation finished for {benchmark_name} with {n_constraints_value} constraints."
            )
            return client

        await asyncio.gather(
            *[
                _run_single_combination(benchmark_name, n_constraints_value)
                for benchmark_name, n_constraints_value in benchmark_combinations
            ]
        )


def _empty_torch_cuda_cache() -> None:
    if torch is None:
        return
    cuda = getattr(torch, "cuda", None)
    if cuda is None:
        return
    is_available = getattr(cuda, "is_available", None)
    if callable(is_available) and not is_available():
        return
    empty_cache = getattr(cuda, "empty_cache", None)
    if callable(empty_cache):
        empty_cache()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(
        main(
            benchmark=args.benchmark,
            output_dir=args.output_dir,
            with_generate=args.with_generate,
            with_eval=args.with_eval,
            override=args.override,
            n_constraints=args.n_constraints,
            n_benchmark_data=args.n_benchmark_data,
            seed=args.seed,
            model_specs_json=args.model_specs_json,
            judge_model_spec_json=args.judge_model_spec_json,
            constraint_set=args.constraint_set,
            ifbench_dataset_path=args.ifbench_dataset_path,
            dataset_path=args.dataset_path,
            n_concurrent_generations=args.n_concurrent_generations,
        )
    )
