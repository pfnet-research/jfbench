from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
import re
from typing import Any
from typing import cast
from typing import Iterable
from typing import Literal
from typing import Mapping
from typing import Sequence
from typing import TYPE_CHECKING

from datasets import Dataset
from openai import APITimeoutError
from openai import AsyncOpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

from jfbench.llm import LLMClient
from jfbench.sft_dataset.generate import _score_one_with_echo
from jfbench.sft_dataset.generate import ConstraintInventory


if TYPE_CHECKING:
    from jfbench.benchmark.build import ConstraintSetName
    from jfbench.protocol import Constraint


DEFAULT_DATASET_PATH = "data/dpo_dataset"
DEFAULT_ANALYSIS_PLOT_DIR = "work/fix-jftrain-binary-reasoning/analysis-plots"
DEFAULT_REWARD_BASE_URL = "http://localhost:8000/v1"
DEFAULT_REWARD_API_KEY = "unused"
DEFAULT_REWARD_TOKEN_LIMIT = 512
DEFAULT_REWARD_CONCURRENCY = 40
DEFAULT_LOGPROB_API_KEY = "unused"
DEFAULT_ANALYSIS_CONCURRENCY = 512
TIMEOUT = 120
TOKEN_LENGTH_KEYS = (
    "chosen_token_length",
    "rejected_token_length",
    "chosen_reasoning_token_length",
    "rejected_reasoning_token_length",
)
PASS_COUNT_KEYS = (
    "chosen_all_constraints_pass_count",
    "rejected_al_constraints_pass_count",
)
REWARD_KEYS = (
    "chosen_reward",
    "rejected_reward",
    "chosen_reasoning_reward",
    "rejected_reasoning_reward",
    "sum_of_chosen_and_chosen_reasoning_reward",
    "sum_of_rejected_and_rejected_reasoning_reward",
    "concatenated_chosen_and_chosen_reasoning_reward",
    "concatenated_rejected_and_rejected_reasoning_reward",
)
LOGPROB_KEYS = (
    "chosen_delta_logp",
    "rejected_delta_logp",
    "delta_margin",
)
ALL_METRIC_KEYS = TOKEN_LENGTH_KEYS + PASS_COUNT_KEYS + REWARD_KEYS + LOGPROB_KEYS

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConstraintStats:
    total_records: int
    counts_by_constraint_total: dict[int, int]
    counts_by_prompt_and_constraint_count: dict[str, dict[int, int]]
    per_constraint_counts: dict[int, dict[str, dict[str, int]]]


@dataclass(frozen=True)
class NumericDistribution:
    count: int
    minimum: float
    maximum: float
    mean: float
    p50: float
    p90: float
    p95: float
    p99: float
    bucket_counts: dict[str, int]


class _AnalyzeLLMClient:
    async def async_ask(
        self, prompts: list[str], *, use_tqdm: bool = False
    ) -> tuple[list[str], list[None]]:
        raise RuntimeError("This placeholder judge client cannot perform LLM calls.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a DPO dataset stored on disk and show statistics."
    )
    parser.add_argument(
        "dataset_paths",
        nargs="*",
        default=(DEFAULT_DATASET_PATH,),
        help="Paths to HuggingFace Dataset directories.",
    )
    parser.add_argument(
        "--show-generated",
        action="store_true",
        help="Show prompt, response, and reasoning content for each data entry.",
    )
    parser.add_argument(
        "--tokenizer-model-name",
        type=str,
        default=None,
        help="Tokenizer model name used for token-length and chat-template analyses.",
    )
    parser.add_argument(
        "--judge-model-spec",
        type=str,
        default=None,
        help=(
            "JSON object describing the judge model. "
            "Expected keys: provider, model, optional name/extra_body."
        ),
    )
    parser.add_argument(
        "--n-evaluation",
        type=int,
        default=0,
        help="Number of repeated evaluations per chosen/rejected response.",
    )
    parser.add_argument(
        "--evaluation-temperatures",
        type=str,
        default="0.0,0.2,0.4,0.6,0.8,1.0",
        help="Comma-separated temperatures used for repeated evaluation.",
    )
    parser.add_argument(
        "--constraint-set",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Constraint set used when reconstructing constraints from names.",
    )
    parser.add_argument(
        "--reward-model-spec",
        type=str,
        default=None,
        help=(
            "JSON object describing reward scoring model. "
            "Expected keys: model and optional base_url/api_key/token_limit/concurrency."
        ),
    )
    parser.add_argument(
        "--before-training-model-name",
        type=str,
        default=None,
        help="Model name used to compute logp_before for chosen/rejected.",
    )
    parser.add_argument(
        "--after-training-model-name",
        type=str,
        default=None,
        help="Model name used to compute logp_after for chosen/rejected.",
    )
    parser.add_argument(
        "--before-training-model-base-url",
        type=str,
        default=None,
        help=(
            "Base URL for OpenAI-compatible endpoint used with "
            "--before-training-model-name when computing logp_before."
        ),
    )
    parser.add_argument(
        "--after-training-model-base-url",
        type=str,
        default=None,
        help=(
            "Base URL for OpenAI-compatible endpoint used with "
            "--after-training-model-name when computing logp_after."
        ),
    )
    parser.add_argument(
        "--logprob-api-key",
        type=str,
        default=DEFAULT_LOGPROB_API_KEY,
        help="API key for logprob endpoint.",
    )
    parser.add_argument(
        "--analysis-concurrency",
        type=int,
        default=DEFAULT_ANALYSIS_CONCURRENCY,
        help="Maximum number of concurrent analysis tasks.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=100,
        help="Flush enriched dataset to disk every N updated rows.",
    )
    parser.add_argument(
        "--analysis-plot-dir",
        type=str,
        default=DEFAULT_ANALYSIS_PLOT_DIR,
        help="Directory where Plotly histogram HTML files are written.",
    )
    return parser.parse_args(argv)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {path}")
    dataset = Dataset.load_from_disk(str(path))
    return dataset.to_list()


def load_datasets(paths: Sequence[str]) -> tuple[list[dict[str, Any]], list[tuple[Path, int]]]:
    merged: list[dict[str, Any]] = []
    details: list[tuple[Path, int]] = []
    for raw_path in paths:
        path = Path(raw_path)
        records = load_jsonl(path)
        merged.extend(records)
        details.append((path, len(records)))
    return merged, details


def compute_constraint_stats(
    rows: Iterable[Mapping[str, Any]],
) -> ConstraintStats:
    total_records = 0
    count_totals: Counter[int] = Counter()
    counts_by_prompt_and_constraint: dict[str, Counter[int]] = defaultdict(Counter)
    per_constraint: dict[int, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    for row in rows:
        count = _resolve_constraint_count(row)
        total_records += 1
        count_totals[count] += 1
        prompt_source = _normalize_prompt_source(row.get("prompt_source"))
        if prompt_source:
            counts_by_prompt_and_constraint[prompt_source][count] += 1
        for group, name in _iter_constraint_pairs(row):
            per_constraint[count][group][name] += 1
    sorted_counts = dict(sorted(count_totals.items()))
    sorted_constraint_counts = {
        count: {
            group: dict(sorted(counter.items()))
            for group, counter in sorted(group_counters.items())
        }
        for count, group_counters in sorted(per_constraint.items())
    }
    sorted_prompt_counts = {
        prompt_source: dict(sorted(counter.items()))
        for prompt_source, counter in sorted(counts_by_prompt_and_constraint.items())
    }
    return ConstraintStats(
        total_records=total_records,
        counts_by_constraint_total=sorted_counts,
        counts_by_prompt_and_constraint_count=sorted_prompt_counts,
        per_constraint_counts=sorted_constraint_counts,
    )


def format_stats(stats: ConstraintStats) -> str:
    lines = [f"Total data: {stats.total_records}"]
    lines.append("Total data per constraint count:")
    if stats.counts_by_constraint_total:
        for count, total in stats.counts_by_constraint_total.items():
            lines.append(f"  {count}: {total}")
    else:
        lines.append("  (no data)")
    lines.append("Total data per prompt source and constraint count:")
    if stats.counts_by_prompt_and_constraint_count:
        for prompt_source, counts in stats.counts_by_prompt_and_constraint_count.items():
            lines.append(f"  {prompt_source}:")
            for count, total in counts.items():
                lines.append(f"    {count}: {total}")
    else:
        lines.append("  (no data)")
    lines.append("Constraint breakdown by constraint count:")
    if stats.per_constraint_counts:
        for count, group_counters in stats.per_constraint_counts.items():
            lines.append(f"  {count}:")
            for group, counters in group_counters.items():
                lines.append(f"    {group}:")
                for name, total in counters.items():
                    lines.append(f"      {name}: {total}")
    else:
        lines.append("  (no data)")
    return "\n".join(lines)


def format_numeric_distribution(
    title: str,
    values: Sequence[float],
    *,
    bucket_size: float,
) -> str:
    print(f"Computing numeric distribution for {title}...")
    print(f"  count of values: {len(values)}")
    print(f"  bucket size: {bucket_size}")
    stats = compute_numeric_distribution(values, bucket_size=bucket_size)
    lines = [title]
    if stats.count == 0:
        lines.append("  (no data)")
        return "\n".join(lines)
    lines.append(f"  count: {stats.count}")
    lines.append(f"  min: {stats.minimum:.4f}")
    lines.append(f"  max: {stats.maximum:.4f}")
    lines.append(f"  mean: {stats.mean:.4f}")
    lines.append(f"  p50: {stats.p50:.4f}")
    lines.append(f"  p90: {stats.p90:.4f}")
    lines.append(f"  p95: {stats.p95:.4f}")
    lines.append(f"  p99: {stats.p99:.4f}")
    lines.append("  buckets:")
    for label, count in stats.bucket_counts.items():
        lines.append(f"    {label}: {count}")
    return "\n".join(lines)


def _slugify_for_filename(value: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()
    if slug:
        return slug
    return "plot"


def _write_numeric_distribution_histogram(
    *,
    title: str,
    values: Sequence[float],
    bucket_size: float,
    output_path: Path,
) -> Path | None:
    return _write_numeric_distribution_histogram_series(
        title=title,
        series=({"name": "values", "values": values, "color": "#636EFA"},),
        bucket_size=bucket_size,
        output_path=output_path,
    )


def _write_numeric_distribution_histogram_series(
    *,
    title: str,
    series: Sequence[Mapping[str, Any]],
    bucket_size: float,
    output_path: Path,
) -> Path | None:
    import plotly.graph_objects as go

    traces: list[Any] = []
    stats_per_series: list[tuple[str, NumericDistribution, str]] = []
    all_numeric_values: list[float] = []

    for item in series:
        name = str(item["name"])
        color = str(item.get("color", "#636EFA"))
        numeric_values: list[float] = []
        for value in cast("Sequence[float]", item["values"]):
            numeric_value = float(value)
            if math.isfinite(numeric_value):
                numeric_values.append(numeric_value)
        if not numeric_values:
            continue
        stats = compute_numeric_distribution(numeric_values, bucket_size=bucket_size)
        stats_per_series.append((name, stats, color))
        all_numeric_values.extend(numeric_values)
        histogram_kwargs: dict[str, Any] = {
            "x": numeric_values,
            "name": name,
            "opacity": 0.6,
            "marker_color": color,
        }
        traces.append(go.Histogram(**histogram_kwargs))

    if not traces:
        return None

    fig = go.Figure(data=traces)

    if bucket_size > 0:
        min_value = min(all_numeric_values)
        max_value = max(all_numeric_values)
        bin_start = math.floor(min_value / bucket_size) * bucket_size
        bin_end = math.ceil(max_value / bucket_size) * bucket_size + bucket_size
        xbins = {
            "start": bin_start,
            "end": bin_end,
            "size": bucket_size,
        }
        for trace in fig.data:
            trace.xbins = xbins

    fig.update_layout(barmode="overlay")

    stat_line_specs = [
        ("min", "minimum", "#636EFA"),
        ("max", "maximum", "#EF553B"),
        ("mean", "mean", "#00CC96"),
        ("p50", "p50", "#AB63FA"),
        ("p90", "p90", "#FFA15A"),
        ("p95", "p95", "#19D3F3"),
        ("p99", "p99", "#FF6692"),
    ]
    y_base = 1.02
    y_step = 0.05
    annotation_index = 0
    for series_name, stats, _ in stats_per_series:
        for label, field_name, color in stat_line_specs:
            x_value = float(getattr(stats, field_name))
            fig.add_vline(
                x=x_value,
                line_width=2,
                line_dash="dot",
                line_color=color,
            )
            fig.add_annotation(
                x=x_value,
                y=y_base + (annotation_index % 8) * y_step,
                xref="x",
                yref="paper",
                text=f"{series_name} {label}: {x_value:.4f}",
                showarrow=False,
                xanchor="left",
                yanchor="bottom",
                font={"color": color},
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=color,
                borderwidth=1,
            )
            annotation_index += 1

    count_text = " | ".join(
        f"{series_name} count: {stats.count}" for series_name, stats, _ in stats_per_series
    )
    fig.add_annotation(
        x=1.02,
        y=1.08,
        xref="paper",
        yref="paper",
        text=count_text,
        showarrow=False,
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="black",
        borderwidth=1,
    )
    fig.update_layout(
        title=title,
        xaxis_title="Value",
        yaxis_title="Count",
        bargap=0.05,
        margin={"r": 220, "t": 220},
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output_path))
    return output_path


def write_numeric_distribution_histogram(
    *,
    title: str,
    values: Sequence[float],
    bucket_size: float,
    output_path: Path,
) -> Path | None:
    return _write_numeric_distribution_histogram(
        title=title,
        values=values,
        bucket_size=bucket_size,
        output_path=output_path,
    )


def compute_numeric_distribution(
    values: Sequence[float],
    *,
    bucket_size: float,
) -> NumericDistribution:
    numeric_values: list[float] = []
    for value in values:
        numeric_value = float(value)
        if math.isfinite(numeric_value):
            numeric_values.append(numeric_value)
    if not numeric_values:
        return NumericDistribution(
            count=0,
            minimum=0.0,
            maximum=0.0,
            mean=0.0,
            p50=0.0,
            p90=0.0,
            p95=0.0,
            p99=0.0,
            bucket_counts={},
        )
    sorted_values = sorted(numeric_values)
    count = len(sorted_values)

    def percentile(ratio: float) -> float:
        idx = round((count - 1) * ratio)
        idx = min(max(idx, 0), count - 1)
        return sorted_values[idx]

    min_value = sorted_values[0]
    max_value = sorted_values[-1]
    mean_value = sum(sorted_values) / count
    bucket_counter: Counter[str] = Counter()
    if bucket_size > 0:
        for value in sorted_values:
            lower = int(value // bucket_size) * bucket_size
            upper = lower + bucket_size
            label = f"[{lower:.0f}, {upper:.0f})"
            bucket_counter[label] += 1
    return NumericDistribution(
        count=count,
        minimum=min_value,
        maximum=max_value,
        mean=mean_value,
        p50=percentile(0.50),
        p90=percentile(0.90),
        p95=percentile(0.95),
        p99=percentile(0.99),
        bucket_counts=dict(sorted(bucket_counter.items())),
    )


def _resolve_constraint_count(row: Mapping[str, Any]) -> int:
    value = row.get("constraint_count")
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    constraints = row.get("constraints") or []
    if isinstance(constraints, Sequence) and not isinstance(constraints, (str, bytes)):
        return len(constraints)
    return 0


def _iter_constraint_pairs(row: Mapping[str, Any]) -> Iterable[tuple[str, str]]:
    groups = _normalize_string_sequence(row.get("constraint_groups"))
    constraints = row.get("constraints") or []
    yielded = False
    if isinstance(constraints, Sequence) and not isinstance(constraints, (str, bytes)):
        for idx, constraint in enumerate(constraints):
            if not isinstance(constraint, Mapping):
                continue
            name = constraint.get("name")
            if not isinstance(name, str) or not name:
                continue
            group = constraint.get("group")
            if (not isinstance(group, str) or not group) and idx < len(groups):
                group = groups[idx]
            yield (_normalize_group_name(group), name)
            yielded = True
        if yielded:
            return
    names = _normalize_string_sequence(row.get("constraint_types"))
    length = min(len(names), len(groups))
    if length:
        for idx in range(length):
            yield (_normalize_group_name(groups[idx]), names[idx])
        return
    for name in names:
        yield (_normalize_group_name(None), name)


def _normalize_string_sequence(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        result: list[str] = []
        for item in value:
            if item is None:
                continue
            result.append(str(item))
        return result
    return [str(value)]


def _normalize_group_name(value: object) -> str:
    if isinstance(value, str) and value:
        return value
    return "unknown"


def _normalize_prompt_source(value: object | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    return normalized


def _format_generated_value(value: Any, missing_label: str, empty_label: str) -> str:
    if value is None:
        return missing_label
    text = str(value)
    if text == "":
        return empty_label
    return text


def _extract_section_mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for entry in value:
            if isinstance(entry, Mapping):
                return entry
    return None


def _extract_chosen_response(row: Mapping[str, Any]) -> Any:
    section = _extract_section_mapping(row.get("chosen"))
    if section is not None:
        response = section.get("response")
        if response is None:
            response = section.get("content")
        if response is not None:
            return response
    return row.get("response")


def _extract_chosen_reasoning(row: Mapping[str, Any]) -> Any:
    explicit_reasoning = row.get("chosen_reasoning")
    if explicit_reasoning is not None:
        return explicit_reasoning
    section = _extract_section_mapping(row.get("chosen"))
    if section is not None:
        return section.get("reasoning_content")
    return row.get("reasoning_content")


def _extract_rejected_response(row: Mapping[str, Any]) -> Any:
    section = _extract_section_mapping(row.get("rejected"))
    if section is not None:
        response = section.get("response")
        if response is None:
            response = section.get("content")
        if response is not None:
            return response
    return row.get("response")


def _extract_rejected_reasoning(row: Mapping[str, Any]) -> Any:
    explicit_reasoning = row.get("rejected_reasoning")
    if explicit_reasoning is not None:
        return explicit_reasoning
    section = _extract_section_mapping(row.get("rejected"))
    if section is not None:
        return section.get("reasoning_content")
    return row.get("reasoning_content")


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _token_length(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _extract_token_lengths(
    rows: Sequence[Mapping[str, Any]],
    *,
    tokenizer: Any,
) -> dict[str, list[int]]:
    lengths: dict[str, list[int]] = {
        "chosen": [],
        "rejected": [],
        "chosen_reasoning": [],
        "rejected_reasoning": [],
    }
    for row in tqdm(rows, desc="Tokenizing chosen/rejected texts", unit="row"):
        lengths["chosen"].append(_token_length(tokenizer, _as_text(_extract_chosen_response(row))))
        lengths["rejected"].append(
            _token_length(tokenizer, _as_text(_extract_rejected_response(row)))
        )
        lengths["chosen_reasoning"].append(
            _token_length(tokenizer, _as_text(_extract_chosen_reasoning(row)))
        )
        lengths["rejected_reasoning"].append(
            _token_length(tokenizer, _as_text(_extract_rejected_reasoning(row)))
        )
    return lengths


def _row_has_keys(row: Mapping[str, Any], keys: Sequence[str]) -> bool:
    for key in keys:
        if key not in row:
            return False
    return True


def _collect_missing_indices(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> list[int]:
    return [idx for idx, row in enumerate(rows) if not _row_has_keys(row, keys)]


def _save_records(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    Dataset.from_list([dict(row) for row in rows]).save_to_disk(str(path))


def _parse_model_spec(value: str, *, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} must be valid JSON.") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return parsed


def _build_wrapper_map(inventory: ConstraintInventory) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    dummy_client = cast("LLMClient", _AnalyzeLLMClient())
    for wrapper in inventory.all_wrappers:
        constraint = wrapper.build(dummy_client, "", seed=0)
        mapping[constraint.__class__.__name__] = wrapper
    return mapping


def _has_llm_constraints(
    rows: Sequence[Mapping[str, Any]],
    wrapper_map: Mapping[str, Any],
) -> bool:
    for row in rows:
        for name in _normalize_string_sequence(row.get("constraint_types")):
            wrapper = wrapper_map.get(name)
            if wrapper is None:
                continue
            if getattr(wrapper, "mode", "rule") == "llm":
                return True
    return False


def _stable_seed(*parts: str) -> int:
    payload = "||".join(parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


async def _evaluate_constraint(constraint: Constraint, value: str) -> bool:
    result = constraint.evaluate(value)
    if asyncio.iscoroutine(result):
        passed, _ = await result
    else:
        passed, _ = cast("tuple[bool, str | None]", result)
    return bool(passed)


async def _evaluate_with_temperatures(
    rows: Sequence[Mapping[str, Any]],
    *,
    wrapper_map: Mapping[str, Any],
    judge_spec: dict[str, Any] | None,
    n_evaluation: int,
    temperatures: Sequence[float],
    max_concurrency: int,
) -> tuple[list[int], list[int]]:
    if n_evaluation <= 0:
        return [0] * len(rows), [0] * len(rows)

    judge_clients: dict[float, LLMClient] = {}
    if judge_spec is not None:
        for temp in temperatures:
            extra_body = dict(judge_spec.get("extra_body") or {})
            extra_body["temperature"] = temp
            provider = cast(
                "Literal['openrouter', 'vllm']",
                str(judge_spec.get("provider", "vllm")),
            )
            judge_clients[temp] = LLMClient(
                provider=provider,
                model=str(judge_spec["model"]),
                extra_body=extra_body,
            )

    fallback_client = cast("LLMClient", _AnalyzeLLMClient())
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _evaluate_constraint_limited(constraint: Constraint, value: str) -> bool:
        async with semaphore:
            return await _evaluate_constraint(constraint, value)

    async def _evaluate_trial(
        *,
        constraint_names: Sequence[str],
        document: str,
        data_id: str,
        chosen_text: str,
        rejected_text: str,
        trial_index: int,
    ) -> tuple[bool, bool]:
        temperature = temperatures[trial_index % len(temperatures)]
        judge_client = judge_clients.get(temperature, fallback_client)
        constraints: list[Constraint] = []
        for name in constraint_names:
            wrapper = wrapper_map.get(name)
            if wrapper is None:
                continue
            seed = _stable_seed(data_id, name, str(trial_index), str(temperature))
            constraints.append(wrapper.build(judge_client, document, seed=seed))

        if not constraints:
            return True, True

        paired_tasks = [
            _evaluate_constraint_limited(constraint, chosen_text) for constraint in constraints
        ] + [_evaluate_constraint_limited(constraint, rejected_text) for constraint in constraints]
        paired_results = await asyncio.gather(*paired_tasks)
        split_at = len(constraints)
        chosen_results = paired_results[:split_at]
        rejected_results = paired_results[split_at:]
        return all(chosen_results), all(rejected_results)

    row_inputs: list[tuple[list[str], str, str, str, str]] = []
    for row in rows:
        constraint_names = _normalize_string_sequence(row.get("constraint_types"))
        document = _as_text(row.get("prompt_document")) or _as_text(row.get("prompt"))
        data_id = _as_text(row.get("data_id"))
        chosen_text = _as_text(_extract_chosen_response(row))
        rejected_text = _as_text(_extract_rejected_response(row))
        row_inputs.append((constraint_names, document, data_id, chosen_text, rejected_text))

    async def _evaluate_trial_with_row(
        *,
        row_index: int,
        trial_index: int,
        progress: Any,
    ) -> tuple[int, bool, bool]:
        constraint_names, document, data_id, chosen_text, rejected_text = row_inputs[row_index]
        try:
            chosen_ok, rejected_ok = await _evaluate_trial(
                constraint_names=constraint_names,
                document=document,
                data_id=data_id,
                chosen_text=chosen_text,
                rejected_text=rejected_text,
                trial_index=trial_index,
            )
            return row_index, chosen_ok, rejected_ok
        finally:
            progress.update(1)

    total_steps = len(rows) * n_evaluation
    chosen_passes_per_row = [0] * len(rows)
    rejected_passes_per_row = [0] * len(rows)
    with tqdm(total=total_steps, desc="Evaluating constraints", unit="eval") as progress:
        all_trials = [
            _evaluate_trial_with_row(
                row_index=row_index,
                trial_index=trial_index,
                progress=progress,
            )
            for row_index in range(len(row_inputs))
            for trial_index in range(n_evaluation)
        ]
        all_results = await asyncio.gather(*all_trials)
    for row_index, chosen_ok, rejected_ok in all_results:
        if chosen_ok:
            chosen_passes_per_row[row_index] += 1
        if rejected_ok:
            rejected_passes_per_row[row_index] += 1
    return chosen_passes_per_row, rejected_passes_per_row


def _format_count_distribution(title: str, counts: Mapping[int, int]) -> str:
    lines = [title]
    if not counts:
        lines.append("  (no data)")
        return "\n".join(lines)
    for key, value in sorted(counts.items()):
        lines.append(f"  {key}: {value}")
    return "\n".join(lines)


def _build_combined_answer_via_chat_template(
    tokenizer: Any,
    *,
    prompt: str,
    response: str,
    reasoning: str,
) -> str:
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
        {"role": "assistant", "content": reasoning},
    ]
    return str(
        tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    )


async def _compute_reward_distributions(
    rows: Sequence[Mapping[str, Any]],
    *,
    reward_spec: dict[str, Any],
    tokenizer: Any,
    max_concurrency: int,
) -> dict[str, list[float]]:
    model = str(reward_spec["model"])
    base_url = str(reward_spec.get("base_url", DEFAULT_REWARD_BASE_URL))
    api_key = str(reward_spec.get("api_key", DEFAULT_REWARD_API_KEY))
    token_limit = int(reward_spec.get("token_limit", DEFAULT_REWARD_TOKEN_LIMIT))
    reward_concurrency = int(reward_spec.get("concurrency", DEFAULT_REWARD_CONCURRENCY))
    effective_concurrency = max(1, min(max_concurrency, reward_concurrency))
    client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=TIMEOUT)
    reward_tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)
    result: dict[str, list[float]] = {
        "chosen": [],
        "rejected": [],
        "chosen_reasoning": [],
        "rejected_reasoning": [],
        "chosen_plus_reasoning_individual": [],
        "rejected_plus_reasoning_individual": [],
        "chosen_plus_reasoning_chat_template": [],
        "rejected_plus_reasoning_chat_template": [],
    }
    semaphore = asyncio.Semaphore(effective_concurrency)
    reward_inputs: list[tuple[int, str, list[int], int]] = []
    per_row_scores: list[dict[str, float]] = [dict() for _ in rows]

    for row_index, row in enumerate(rows):
        prompt = _as_text(row.get("prompt") or row.get("instruction"))
        chosen = _as_text(_extract_chosen_response(row))
        rejected = _as_text(_extract_rejected_response(row))
        chosen_reasoning = _as_text(_extract_chosen_reasoning(row))
        rejected_reasoning = _as_text(_extract_rejected_reasoning(row))
        chosen_chat = _build_combined_answer_via_chat_template(
            tokenizer,
            prompt=prompt,
            response=chosen,
            reasoning=chosen_reasoning,
        )
        rejected_chat = _build_combined_answer_via_chat_template(
            tokenizer,
            prompt=prompt,
            response=rejected,
            reasoning=rejected_reasoning,
        )
        prefix = f"User:\n{prompt}\nAssistant:\n"
        prefix_ids = reward_tokenizer.encode(prefix, add_special_tokens=False)
        chat_prefix = "User:\n\nAssistant:\n"
        chat_prefix_ids = reward_tokenizer.encode(chat_prefix, add_special_tokens=False)

        chosen_ids = reward_tokenizer.encode(
            f"{prefix}{chosen}",
            add_special_tokens=False,
        )
        rejected_ids = reward_tokenizer.encode(
            f"{prefix}{rejected}",
            add_special_tokens=False,
        )
        chosen_reasoning_ids = reward_tokenizer.encode(
            f"{prefix}{chosen_reasoning}",
            add_special_tokens=False,
        )
        rejected_reasoning_ids = reward_tokenizer.encode(
            f"{prefix}{rejected_reasoning}",
            add_special_tokens=False,
        )
        chosen_chat_ids = reward_tokenizer.encode(
            f"{chat_prefix}{chosen_chat}",
            add_special_tokens=False,
        )
        rejected_chat_ids = reward_tokenizer.encode(
            f"{chat_prefix}{rejected_chat}",
            add_special_tokens=False,
        )
        reward_inputs.extend(
            [
                (row_index, "chosen", chosen_ids, len(prefix_ids)),
                (row_index, "rejected", rejected_ids, len(prefix_ids)),
                (row_index, "chosen_reasoning", chosen_reasoning_ids, len(prefix_ids)),
                (row_index, "rejected_reasoning", rejected_reasoning_ids, len(prefix_ids)),
                (
                    row_index,
                    "chosen_plus_reasoning_chat_template",
                    chosen_chat_ids,
                    len(chat_prefix_ids),
                ),
                (
                    row_index,
                    "rejected_plus_reasoning_chat_template",
                    rejected_chat_ids,
                    len(chat_prefix_ids),
                ),
            ]
        )

    async def _score_task(
        input_item: tuple[int, str, list[int], int],
        progress: Any,
    ) -> tuple[int, str, float]:
        row_index, key, token_ids, candidate_start_idx = input_item
        async with semaphore:
            while True:
                try:
                    score, _ = await _score_one_with_echo(
                        client,
                        model,
                        token_ids=token_ids,
                        candidate_start_idx=candidate_start_idx,
                        tokenizer=reward_tokenizer,
                        token_limit=token_limit,
                        chunk_size=2048,
                        overlap=32,
                    )
                    return row_index, key, score
                except (APITimeoutError, TimeoutError) as exc:
                    logger.warning(f"Timeout error during reward scoring, retrying: {exc}")
                    continue
                finally:
                    progress.update(1)

    with tqdm(
        total=len(reward_inputs), desc="Computing reward distributions", unit="reward"
    ) as progress:
        all_scores = await asyncio.gather(
            *[_score_task(input_item, progress) for input_item in reward_inputs]
        )
    for row_index, key, score in all_scores:
        per_row_scores[row_index][key] = score

    for row_score in per_row_scores:
        chosen_reward = row_score["chosen"]
        rejected_reward = row_score["rejected"]
        chosen_reasoning_reward = row_score["chosen_reasoning"]
        rejected_reasoning_reward = row_score["rejected_reasoning"]
        result["chosen"].append(chosen_reward)
        result["rejected"].append(rejected_reward)
        result["chosen_reasoning"].append(chosen_reasoning_reward)
        result["rejected_reasoning"].append(rejected_reasoning_reward)
        result["chosen_plus_reasoning_individual"].append(chosen_reward + chosen_reasoning_reward)
        result["rejected_plus_reasoning_individual"].append(
            rejected_reward + rejected_reasoning_reward
        )
        result["chosen_plus_reasoning_chat_template"].append(
            row_score["chosen_plus_reasoning_chat_template"]
        )
        result["rejected_plus_reasoning_chat_template"].append(
            row_score["rejected_plus_reasoning_chat_template"]
        )
    return result


def _extract_token_logprob(entry: Mapping[str, Any]) -> tuple[str, float]:
    token = entry.get("token") or entry.get("text") or ""
    if "logprob" in entry and isinstance(entry["logprob"], (int, float)):
        return token, float(entry["logprob"])
    top_candidates = entry.get("top_logprobs") or []
    for candidate in top_candidates:
        if (candidate.get("token") == token) or (candidate.get("text") == token):
            return token, float(candidate.get("logprob"))
    if top_candidates:
        fallback = top_candidates[0]
        return (
            fallback.get("token") or fallback.get("text") or "",
            float(fallback.get("logprob")),
        )
    return token, 0.0


def _chunk_by_tokens(
    tokenizer: Any,
    text: str,
    *,
    chunk_size: int,
    overlap: int,
) -> list[list[int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks: list[list[int]] = []
    i = 0
    n = len(ids)
    while i < n:
        chunks.append(ids[i : min(i + chunk_size, n)])
        if i + chunk_size >= n:
            break
        i += chunk_size - overlap
    return chunks


async def _completions_prompt_logprobs_chunked(
    client: AsyncOpenAI,
    *,
    model: str,
    text: str,
    tokenizer: Any,
    chunk_size: int,
    overlap: int,
) -> dict[str, Any]:
    chunks = _chunk_by_tokens(tokenizer, text, chunk_size=chunk_size, overlap=overlap)
    all_tokens: list[str] = []
    all_logprobs: list[float] = []

    for idx, ids in enumerate(chunks):
        prompt_chunk = tokenizer.decode(ids, clean_up_tokenization_spaces=False)
        while True:
            try:
                resp = await client.completions.create(
                    model=model,
                    prompt=prompt_chunk,
                    max_tokens=0,
                    temperature=0,
                    logprobs=1,
                    echo=True,
                )
                break
            except (APITimeoutError, TimeoutError) as exc:
                logger.warning(f"Timeout error during logprob request, retrying: {exc}")
                continue
        logprob_block: Iterable[Mapping[str, Any]] | None = None
        choice = resp.choices[0]
        raw_logprobs = getattr(choice, "logprobs", None)
        if raw_logprobs and isinstance(raw_logprobs, Mapping):
            content = raw_logprobs.get("content")
            if content:
                logprob_block = content
        if logprob_block is None and raw_logprobs:
            tokens_attr = getattr(raw_logprobs, "tokens", None)
            token_lps_attr = getattr(raw_logprobs, "token_logprobs", None)
            if tokens_attr is not None and token_lps_attr is not None:
                logprob_block = [
                    {"token": tok, "logprob": lp}
                    for tok, lp in zip(tokens_attr, token_lps_attr, strict=False)
                ]
        if not logprob_block:
            raise RuntimeError("logprobs not returned; ensure server supports logprobs.")

        tokens: list[str] = []
        token_lps: list[float] = []
        for entry in logprob_block:
            tok, lp = _extract_token_logprob(entry)
            tokens.append(tok)
            token_lps.append(lp)

        if idx > 0 and overlap > 0:
            tokens = tokens[overlap:]
            token_lps = token_lps[overlap:]

        all_tokens.extend(tokens)
        all_logprobs.extend(token_lps)

    return {
        "tokens": all_tokens,
        "token_logprobs": all_logprobs,
    }


async def _score_sequence_logprob(
    client: AsyncOpenAI,
    *,
    model: str,
    tokenizer: Any,
    prompt: str,
    answer: str,
) -> float:
    prefix = f"User:\n{prompt}\nAssistant:\n"
    text = prefix + answer
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    candidate_start_idx = len(prefix_tokens)
    logprob_result = await _completions_prompt_logprobs_chunked(
        client,
        model=model,
        text=text,
        tokenizer=tokenizer,
        chunk_size=1024,
        overlap=32,
    )
    token_lps: list[float] = []
    for idx, (tok, lp) in enumerate(
        zip(logprob_result["tokens"], logprob_result["token_logprobs"], strict=False)
    ):
        if idx < candidate_start_idx:
            continue
        if str(tok).strip() == "":
            continue
        token_lps.append(float(lp))
    return float(sum(token_lps))


async def _compute_logprob_deltas(
    rows: Sequence[Mapping[str, Any]],
    *,
    before_model_name: str,
    after_model_name: str,
    before_base_url: str,
    after_base_url: str,
    api_key: str,
    tokenizer: Any,
    max_concurrency: int,
) -> dict[str, list[float]]:
    before_client = AsyncOpenAI(base_url=before_base_url, api_key=api_key, timeout=TIMEOUT)
    after_client = AsyncOpenAI(base_url=after_base_url, api_key=api_key, timeout=TIMEOUT)
    deltas_chosen: list[float] = []
    deltas_rejected: list[float] = []
    deltas_margin: list[float] = []

    semaphore = asyncio.Semaphore(max_concurrency)

    async def _compute_one(row: Mapping[str, Any]) -> tuple[float, float, float]:
        async with semaphore:
            prompt = _as_text(row.get("prompt") or row.get("instruction"))
            chosen = _as_text(_extract_chosen_response(row))
            rejected = _as_text(_extract_rejected_response(row))

            before_chosen_fut = _score_sequence_logprob(
                before_client,
                model=before_model_name,
                tokenizer=tokenizer,
                prompt=prompt,
                answer=chosen,
            )
            after_chosen_fut = _score_sequence_logprob(
                after_client,
                model=after_model_name,
                tokenizer=tokenizer,
                prompt=prompt,
                answer=chosen,
            )
            before_rejected_fut = _score_sequence_logprob(
                before_client,
                model=before_model_name,
                tokenizer=tokenizer,
                prompt=prompt,
                answer=rejected,
            )
            after_rejected_fut = _score_sequence_logprob(
                after_client,
                model=after_model_name,
                tokenizer=tokenizer,
                prompt=prompt,
                answer=rejected,
            )
            before_chosen, after_chosen, before_rejected, after_rejected = await asyncio.gather(
                before_chosen_fut,
                after_chosen_fut,
                before_rejected_fut,
                after_rejected_fut,
            )
            delta_chosen = after_chosen - before_chosen
            delta_rejected = after_rejected - before_rejected
            return delta_chosen, delta_rejected, delta_chosen - delta_rejected

    async def _compute_one_with_progress(
        row: Mapping[str, Any],
        progress: Any,
    ) -> tuple[float, float, float]:
        try:
            return await _compute_one(row)
        finally:
            progress.update(1)

    with tqdm(total=len(rows), desc="Computing logprob deltas", unit="row") as progress:
        all_results = await asyncio.gather(
            *[_compute_one_with_progress(row, progress) for row in rows]
        )
    for delta_chosen, delta_rejected, delta_margin in all_results:
        deltas_chosen.append(delta_chosen)
        deltas_rejected.append(delta_rejected)
        deltas_margin.append(delta_margin)

    return {
        "delta_logp_chosen": deltas_chosen,
        "delta_logp_rejected": deltas_rejected,
        "delta_margin": deltas_margin,
    }


async def _enrich_dataset_path(path: Path, args: argparse.Namespace) -> int:
    rows = load_jsonl(path)
    if not rows:
        return 0
    updated = 0
    dirty = 0

    def mark_updated() -> None:
        nonlocal updated, dirty
        updated += 1
        dirty += 1

    def maybe_flush() -> None:
        nonlocal dirty
        if dirty >= args.save_every:
            _save_records(path, rows)
            dirty = 0

    tokenizer = None
    if args.tokenizer_model_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_model_name, use_fast=True, trust_remote_code=True
        )

    if tokenizer is not None:
        missing_token_indices = _collect_missing_indices(rows, TOKEN_LENGTH_KEYS)
        if missing_token_indices:
            target_rows = [rows[idx] for idx in missing_token_indices]
            lengths = _extract_token_lengths(target_rows, tokenizer=tokenizer)
            for pos, row_idx in enumerate(missing_token_indices):
                row = dict(rows[row_idx])
                row["chosen_token_length"] = lengths["chosen"][pos]
                row["rejected_token_length"] = lengths["rejected"][pos]
                row["chosen_reasoning_token_length"] = lengths["chosen_reasoning"][pos]
                row["rejected_reasoning_token_length"] = lengths["rejected_reasoning"][pos]
                rows[row_idx] = row
                mark_updated()
                maybe_flush()

    missing_pass_indices = _collect_missing_indices(rows, PASS_COUNT_KEYS)
    if missing_pass_indices:
        if args.n_evaluation <= 0:
            for row_idx in missing_pass_indices:
                row = dict(rows[row_idx])
                row["chosen_all_constraints_pass_count"] = 0
                row["rejected_al_constraints_pass_count"] = 0
                rows[row_idx] = row
                mark_updated()
                maybe_flush()
        else:
            inventory = ConstraintInventory(
                constraint_set=cast("ConstraintSetName", args.constraint_set)
            )
            wrapper_map = _build_wrapper_map(inventory)
            target_rows = [rows[idx] for idx in missing_pass_indices]
            judge_spec = (
                _parse_model_spec(args.judge_model_spec, label="judge_model_spec")
                if args.judge_model_spec
                else None
            )
            if _has_llm_constraints(target_rows, wrapper_map) and judge_spec is None:
                raise ValueError(
                    "--judge-model-spec is required when evaluating rows with LLM-based constraints."
                )
            temperatures = [
                float(value.strip())
                for value in str(args.evaluation_temperatures).split(",")
                if value.strip()
            ]
            if not temperatures:
                temperatures = [0.0]
            chosen_passes, rejected_passes = await _evaluate_with_temperatures(
                target_rows,
                wrapper_map=wrapper_map,
                judge_spec=judge_spec,
                n_evaluation=args.n_evaluation,
                temperatures=temperatures,
                max_concurrency=args.analysis_concurrency,
            )
            for pos, row_idx in enumerate(missing_pass_indices):
                row = dict(rows[row_idx])
                row["chosen_all_constraints_pass_count"] = chosen_passes[pos]
                row["rejected_al_constraints_pass_count"] = rejected_passes[pos]
                rows[row_idx] = row
                mark_updated()
                maybe_flush()

    reward_ready = args.reward_model_spec is not None and tokenizer is not None
    if reward_ready:
        missing_reward_indices = _collect_missing_indices(rows, REWARD_KEYS)
        if missing_reward_indices:
            reward_spec = _parse_model_spec(args.reward_model_spec, label="reward_model_spec")
            target_rows = [rows[idx] for idx in missing_reward_indices]
            reward_values = await _compute_reward_distributions(
                target_rows,
                reward_spec=reward_spec,
                tokenizer=tokenizer,
                max_concurrency=args.analysis_concurrency,
            )
            for pos, row_idx in enumerate(missing_reward_indices):
                row = dict(rows[row_idx])
                row["chosen_reward"] = reward_values["chosen"][pos]
                row["rejected_reward"] = reward_values["rejected"][pos]
                row["chosen_reasoning_reward"] = reward_values["chosen_reasoning"][pos]
                row["rejected_reasoning_reward"] = reward_values["rejected_reasoning"][pos]
                row["sum_of_chosen_and_chosen_reasoning_reward"] = reward_values[
                    "chosen_plus_reasoning_individual"
                ][pos]
                row["sum_of_rejected_and_rejected_reasoning_reward"] = reward_values[
                    "rejected_plus_reasoning_individual"
                ][pos]
                row["concatenated_chosen_and_chosen_reasoning_reward"] = reward_values[
                    "chosen_plus_reasoning_chat_template"
                ][pos]
                row["concatenated_rejected_and_rejected_reasoning_reward"] = reward_values[
                    "rejected_plus_reasoning_chat_template"
                ][pos]
                rows[row_idx] = row
                mark_updated()
                maybe_flush()

    logprob_ready = (
        tokenizer is not None
        and args.before_training_model_name
        and args.after_training_model_name
        and args.before_training_model_base_url
        and args.after_training_model_base_url
    )
    if logprob_ready:
        missing_logprob_indices = _collect_missing_indices(rows, LOGPROB_KEYS)
        if missing_logprob_indices:
            target_rows = [rows[idx] for idx in missing_logprob_indices]
            delta_values = await _compute_logprob_deltas(
                target_rows,
                before_model_name=args.before_training_model_name,
                after_model_name=args.after_training_model_name,
                before_base_url=args.before_training_model_base_url,
                after_base_url=args.after_training_model_base_url,
                api_key=args.logprob_api_key,
                tokenizer=tokenizer,
                max_concurrency=args.analysis_concurrency,
            )
            for pos, row_idx in enumerate(missing_logprob_indices):
                row = dict(rows[row_idx])
                row["chosen_delta_logp"] = delta_values["delta_logp_chosen"][pos]
                row["rejected_delta_logp"] = delta_values["delta_logp_rejected"][pos]
                row["delta_margin"] = delta_values["delta_margin"][pos]
                rows[row_idx] = row
                mark_updated()
                maybe_flush()

    if dirty > 0:
        _save_records(path, rows)
    return updated


def format_generated_records(rows: Sequence[Mapping[str, Any]]) -> Iterable[str]:
    if not rows:
        yield "No generated records to display."
        return
    for idx, row in enumerate(rows, start=1):
        yield f"[{idx}]"
        yield "Prompt source:"
        yield _format_generated_value(
            row.get("prompt_source"),
            "<no prompt source>",
            "<empty prompt source>",
        )
        yield "Constraint count:"
        yield _format_generated_value(
            row.get("constraint_count"),
            "<no constraint count>",
            "<empty constraint count>",
        )
        yield "Constraint types:"
        yield _format_generated_value(
            row.get("constraint_types"),
            "<no constraint types>",
            "<empty constraint types>",
        )
        yield "Prompt:"
        yield _format_generated_value(row.get("prompt"), "<no prompt>", "<empty prompt>")
        yield "Chosen response:"
        yield _format_generated_value(
            _extract_chosen_response(row),
            "<no response>",
            "<empty response>",
        )
        yield "Chosen reasoning:"
        yield _format_generated_value(
            _extract_chosen_reasoning(row),
            "<no reasoning>",
            "<empty reasoning>",
        )
        yield "Rejected response:"
        yield _format_generated_value(
            _extract_rejected_response(row),
            "<no response>",
            "<empty response>",
        )
        yield "Rejected reasoning:"
        yield _format_generated_value(
            _extract_rejected_reasoning(row),
            "<no reasoning>",
            "<empty reasoning>",
        )
        yield ""


async def _run_extra_analyses(
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    *,
    plot_output_dir: Path | None = None,
) -> list[str]:
    if args.analysis_concurrency <= 0:
        raise ValueError("--analysis-concurrency must be positive.")
    output_dir = plot_output_dir or Path(args.analysis_plot_dir)

    def append_numeric_distribution(
        *,
        section_key: str,
        title: str,
        values: Sequence[float],
        bucket_size: float,
    ) -> None:
        sections.append(format_numeric_distribution(title, values, bucket_size=bucket_size))
        slug = _slugify_for_filename(title)
        output_path = output_dir / f"{section_key}_{slug}.html"
        plot_path = _write_numeric_distribution_histogram(
            title=title,
            values=values,
            bucket_size=bucket_size,
            output_path=output_path,
        )
        if plot_path is not None:
            sections.append(f"  plot: {plot_path}")

    def append_paired_numeric_distribution(
        *,
        section_key: str,
        title: str,
        chosen_title: str,
        chosen_values: Sequence[float],
        rejected_title: str,
        rejected_values: Sequence[float],
        bucket_size: float,
    ) -> None:
        sections.append(
            format_numeric_distribution(chosen_title, chosen_values, bucket_size=bucket_size)
        )
        sections.append(
            format_numeric_distribution(rejected_title, rejected_values, bucket_size=bucket_size)
        )
        slug = _slugify_for_filename(title)
        output_path = output_dir / f"{section_key}_{slug}.html"
        plot_path = _write_numeric_distribution_histogram_series(
            title=title,
            series=(
                {"name": "chosen", "values": chosen_values, "color": "#1f77b4"},
                {"name": "rejected", "values": rejected_values, "color": "#d62728"},
            ),
            bucket_size=bucket_size,
            output_path=output_path,
        )
        if plot_path is not None:
            sections.append(f"  plot: {plot_path}")

    sections: list[str] = []

    if rows and all(_row_has_keys(row, TOKEN_LENGTH_KEYS) for row in rows):
        sections.append("Token length distributions")
        append_paired_numeric_distribution(
            section_key="token_length",
            title="chosen vs rejected token lengths",
            chosen_title="chosen token lengths",
            chosen_values=[float(row["chosen_token_length"]) for row in rows],
            rejected_title="rejected token lengths",
            rejected_values=[float(row["rejected_token_length"]) for row in rows],
            bucket_size=512,
        )
        append_paired_numeric_distribution(
            section_key="token_length",
            title="chosen_reasoning vs rejected_reasoning token lengths",
            chosen_title="chosen_reasoning token lengths",
            chosen_values=[float(row["chosen_reasoning_token_length"]) for row in rows],
            rejected_title="rejected_reasoning token lengths",
            rejected_values=[float(row["rejected_reasoning_token_length"]) for row in rows],
            bucket_size=512,
        )

    if rows and all(_row_has_keys(row, PASS_COUNT_KEYS) for row in rows):
        chosen_counts = Counter(int(row["chosen_all_constraints_pass_count"]) for row in rows)
        rejected_counts = Counter(int(row["rejected_al_constraints_pass_count"]) for row in rows)
        sections.append("Constraint-pass-count distributions")
        sections.append(
            _format_count_distribution(
                "chosen all-constraints-pass count", dict(sorted(chosen_counts.items()))
            )
        )
        sections.append(
            _format_count_distribution(
                "rejected all-constraints-pass count", dict(sorted(rejected_counts.items()))
            )
        )
        append_numeric_distribution(
            section_key="constraint_pass_count",
            title="all-constraints-pass count difference (chosen - rejected)",
            values=[
                float(row["chosen_all_constraints_pass_count"])
                - float(row["rejected_al_constraints_pass_count"])
                for row in rows
            ],
            bucket_size=1.0,
        )

    if rows and all(_row_has_keys(row, REWARD_KEYS) for row in rows):
        sections.append("Reward distributions")
        append_numeric_distribution(
            section_key="reward",
            title="reward difference (chosen - rejected)",
            values=[float(row["chosen_reward"]) - float(row["rejected_reward"]) for row in rows],
            bucket_size=0.5,
        )
        append_numeric_distribution(
            section_key="reward",
            title="reasoning reward difference (chosen - rejected)",
            values=[
                float(row["chosen_reasoning_reward"]) - float(row["rejected_reasoning_reward"])
                for row in rows
            ],
            bucket_size=0.5,
        )
        append_numeric_distribution(
            section_key="reward",
            title="individual-sum reward difference ((chosen + chosen_reasoning) - (rejected + rejected_reasoning))",
            values=[
                float(row["sum_of_chosen_and_chosen_reasoning_reward"])
                - float(row["sum_of_rejected_and_rejected_reasoning_reward"])
                for row in rows
            ],
            bucket_size=0.5,
        )
        append_numeric_distribution(
            section_key="reward",
            title="chat-template reward difference ((chosen + chosen_reasoning) - (rejected + rejected_reasoning))",
            values=[
                float(row["concatenated_chosen_and_chosen_reasoning_reward"])
                - float(row["concatenated_rejected_and_rejected_reasoning_reward"])
                for row in rows
            ],
            bucket_size=0.25,
        )
    if rows and all(_row_has_keys(row, LOGPROB_KEYS) for row in rows):
        sections.append("Logprob delta distributions")
        append_numeric_distribution(
            section_key="logprob_delta",
            title="delta_logp(chosen) = logp_after(chosen) - logp_before(chosen)",
            values=[float(row["chosen_delta_logp"]) for row in rows],
            bucket_size=5.0,
        )
        append_numeric_distribution(
            section_key="logprob_delta",
            title="delta_logp(rejected) = logp_after(rejected) - logp_before(rejected)",
            values=[float(row["rejected_delta_logp"]) for row in rows],
            bucket_size=5.0,
        )
        append_numeric_distribution(
            section_key="logprob_delta",
            title="delta_margin = delta_logp(chosen) - delta_logp(rejected)",
            values=[float(row["delta_margin"]) for row in rows],
            bucket_size=5.0,
        )

    return sections


async def run_extra_analyses(
    args: argparse.Namespace,
    rows: Sequence[Mapping[str, Any]],
    *,
    plot_output_dir: Path | None = None,
) -> list[str]:
    return await _run_extra_analyses(args, rows, plot_output_dir=plot_output_dir)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.save_every <= 0:
        raise ValueError("--save-every must be positive.")
    raw_paths = args.dataset_paths or [DEFAULT_DATASET_PATH]
    dataset_paths = [Path(path) for path in raw_paths]
    for path in dataset_paths:
        updated = asyncio.run(_enrich_dataset_path(path, args))
        if updated:
            print(f"Updated {updated} rows in {path}")

    records, _details = load_datasets([str(path) for path in dataset_paths])

    stats = compute_constraint_stats(records)
    print(format_stats(stats))

    extra_sections = asyncio.run(
        _run_extra_analyses(
            args,
            records,
            plot_output_dir=Path(args.analysis_plot_dir),
        )
    )
    if extra_sections:
        print()
        print("Additional analyses:")
        print("-------------------")
        for idx, section in enumerate(extra_sections):
            if idx > 0:
                print()
            print(section)

    if args.show_generated:
        print()
        print("Generated responses:")
        print("-------------------")
        for line in format_generated_records(records):
            print(line)


__all__ = [
    "ConstraintStats",
    "NumericDistribution",
    "compute_constraint_stats",
    "compute_numeric_distribution",
    "format_generated_records",
    "format_numeric_distribution",
    "format_stats",
    "load_datasets",
    "load_jsonl",
    "main",
    "parse_args",
    "run_extra_analyses",
    "write_numeric_distribution_histogram",
]


if __name__ == "__main__":
    main()
