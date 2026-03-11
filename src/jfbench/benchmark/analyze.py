from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any
from typing import Iterable
from typing import Mapping
from typing import Sequence


DEFAULT_RESULTS_PATH = Path("data/benchmark_results.jsonl")


@dataclass(frozen=True)
class GroupKey:
    prompt_source: str
    n_constraints: int
    model_short: str


@dataclass
class GroupStats:
    total_count: int = 0
    generated_count: int = 0
    evaluated_count: int = 0
    success_count: int = 0
    empty_reasoning_count: int = 0

    def register(self, record: Mapping[str, Any]) -> None:
        self.total_count += 1
        if _is_empty_reasoning_content(record.get("reasoning_content")):
            self.empty_reasoning_count += 1
        if _has_generated_output(record):
            self.generated_count += 1
        results = record.get("results")
        if results is not None:
            self.evaluated_count += 1
            if (
                isinstance(results, Mapping)
                and results
                and all(bool(value) for value in results.values())
            ):
                self.success_count += 1

    @property
    def pending_count(self) -> int:
        return max(self.generated_count - self.evaluated_count, 0)

    @property
    def failure_count(self) -> int:
        return max(self.evaluated_count - self.success_count, 0)

    @property
    def success_rate(self) -> float:
        if self.evaluated_count == 0:
            return 0.0
        return (self.success_count / self.evaluated_count) * 100.0


def _coerce_prompt_source(value: Any) -> str:
    text = str(value).strip() if value is not None else ""
    return text or "unknown"


def _coerce_model_short(record: Mapping[str, Any]) -> str:
    model_short = record.get("model_short")
    if isinstance(model_short, str) and model_short.strip():
        return model_short.strip()
    model = record.get("model")
    if isinstance(model, str) and model.strip():
        return model.strip()
    return "unknown-model"


def _coerce_n_constraints(value: Any) -> int:
    if value is None:
        return -1
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _normalize_string_sequence(value: Any) -> list[str]:
    if isinstance(value, str) or isinstance(value, bytes):
        return [str(value)]
    if isinstance(value, Sequence):
        return [str(item) for item in value]
    return []


def _is_empty_reasoning_content(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip() == ""


def _has_generated_output(record: Mapping[str, Any]) -> bool:
    results = record.get("results")
    if results is not None:
        return True
    response = record.get("response")
    if response is None:
        return False
    if isinstance(response, str):
        return True
    return True


def _collect_results_paths(results_path: Path) -> list[Path]:
    if not results_path.exists():
        return []
    if results_path.is_dir():
        return sorted(path for path in results_path.rglob("*.jsonl") if path.is_file())
    if results_path.is_file() and results_path.stat().st_size > 0:
        return [results_path]
    return []


def load_records(results_path: Path) -> list[dict[str, Any]]:
    results_paths = _collect_results_paths(results_path)
    if not results_paths:
        print(f"No benchmark results found at {results_path}.")
        return []
    records: list[dict[str, Any]] = []
    for path in results_paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError as exc:  # pragma: no cover - unexpected format
                    raise ValueError(f"Invalid JSON line in {path}: {exc}") from exc
                if isinstance(record, dict):
                    records.append(record)
    extra = ""
    if results_path.is_dir():
        extra = f" ({len(results_paths)} files)"
    print(f"Loaded {len(records)} benchmark records from {results_path}{extra}.")
    return records


def filter_records_by_constraint(
    records: Sequence[Mapping[str, Any]],
    constraint: str | None,
) -> list[dict[str, Any]]:
    if constraint is None:
        return [dict(record) for record in records]
    normalized = constraint.strip().lower()
    if not normalized:
        return [dict(record) for record in records]

    filtered: list[dict[str, Any]] = []
    for record in records:
        if _record_matches_constraint(record, normalized):
            filtered.append(dict(record))
    if filtered:
        print(
            f"Filtered records to {len(filtered)} entries matching constraint '{constraint}'.",
        )
    else:
        print(f"No records matched constraint '{constraint}'.")
    return filtered


def _record_matches_constraint(record: Mapping[str, Any], normalized_constraint: str) -> bool:
    constraint_types = record.get("constraint_types") or []
    for type_name in _normalize_string_sequence(constraint_types):
        if type_name.strip().lower() == normalized_constraint:
            return True
    constraints_meta = record.get("constraints") or []
    if isinstance(constraints_meta, Sequence) and not isinstance(constraints_meta, (str, bytes)):
        for entry in constraints_meta:
            if not isinstance(entry, Mapping):
                continue
            entry_name = entry.get("name")
            if isinstance(entry_name, str) and entry_name.strip().lower() == normalized_constraint:
                return True
    return False


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[GroupKey, GroupStats]:
    summary: dict[GroupKey, GroupStats] = {}
    for record in records:
        prompt_source = _coerce_prompt_source(record.get("prompt_source"))
        n_constraints = _coerce_n_constraints(record.get("n_constraints"))
        model_short = _coerce_model_short(record)
        key = GroupKey(
            prompt_source=prompt_source,
            n_constraints=n_constraints,
            model_short=model_short,
        )
        stats = summary.setdefault(key, GroupStats())
        stats.register(record)
    return summary


def _format_summary(summary: Mapping[GroupKey, GroupStats]) -> Iterable[str]:
    if not summary:
        yield "No benchmark records found."
        return
    header = (
        f"{'prompt_source':<20}"
        f"{'n_constraints':>15}"
        f"{'model_short':>35}"
        f"{'generated':>12}"
        f"{'evaluated':>12}"
        f"{'empty_reasoning':>18}"
        f"{'pending':>10}"
        f"{'success':>10}"
        f"{'failure':>10}"
        f"{'success%':>10}"
    )
    yield header
    yield "-" * len(header)
    for key in sorted(
        summary.keys(),
        key=lambda item: (item.prompt_source, item.n_constraints, item.model_short),
    ):
        stats = summary[key]
        yield (
            f"{key.prompt_source:<20}"
            f"{key.n_constraints:>15}"
            f"{key.model_short:>35}"
            f"{stats.generated_count:>12}"
            f"{stats.evaluated_count:>12}"
            f"{stats.empty_reasoning_count:>18}"
            f"{stats.pending_count:>10}"
            f"{stats.success_count:>10}"
            f"{stats.failure_count:>10}"
            f"{stats.success_rate:>9.1f}%"
        )
    total_stats = GroupStats()
    for stats in summary.values():
        total_stats.total_count += stats.total_count
        total_stats.generated_count += stats.generated_count
        total_stats.evaluated_count += stats.evaluated_count
        total_stats.success_count += stats.success_count
        total_stats.empty_reasoning_count += stats.empty_reasoning_count
    yield "-" * len(header)
    yield (
        f"{'TOTAL':<20}"
        f"{'':>15}"
        f"{'':>35}"
        f"{total_stats.generated_count:>12}"
        f"{total_stats.evaluated_count:>12}"
        f"{total_stats.empty_reasoning_count:>18}"
        f"{total_stats.pending_count:>10}"
        f"{total_stats.success_count:>10}"
        f"{total_stats.failure_count:>10}"
        f"{total_stats.success_rate:>9.1f}%"
    )


def print_summary(summary: Mapping[GroupKey, GroupStats]) -> None:
    for line in _format_summary(summary):
        print(line)


def _format_prompt_preview(prompt: Any) -> str:
    if prompt is None:
        return "<no prompt>"
    text = str(prompt)
    if text == "":
        return "<empty prompt>"
    return text


def _format_response_preview(response: Any) -> str:
    if response is None:
        return "<no response>"
    text = str(response)
    if text == "":
        return "<empty response>"
    return text


def _format_reasoning_preview(reasoning_content: Any) -> str:
    if reasoning_content is None:
        return "<no reasoning>"
    text = str(reasoning_content)
    if text == "":
        return "<empty reasoning>"
    return text


def _format_pass_status(record: Mapping[str, Any]) -> str:
    results = record.get("results")
    if results is None:
        return "not evaluated"
    if isinstance(results, Mapping) and results and all(bool(value) for value in results.values()):
        return "passed"
    return "failed"


def format_generated_responses(records: Sequence[Mapping[str, Any]]) -> Iterable[str]:
    generated_records = [record for record in records if _has_generated_output(record)]
    if not generated_records:
        yield "No generated responses to display."
        return
    for idx, record in enumerate(generated_records, start=1):
        prompt_source = _coerce_prompt_source(record.get("prompt_source"))
        n_constraints = _coerce_n_constraints(record.get("n_constraints"))
        model_short = _coerce_model_short(record)
        data_id = record.get("data_id")
        header = f"[{idx}] {prompt_source} | n={n_constraints} | {model_short}"
        if isinstance(data_id, str) and data_id.strip():
            header = f"{header} | {data_id.strip()}"
        header = f"{header} | {_format_pass_status(record)}"
        yield header
        yield "Prompt:"
        yield f"{_format_prompt_preview(record.get('prompt'))}"
        yield "Response:"
        yield f"{_format_response_preview(record.get('response'))}"
        yield "Reasoning:"
        yield f"{_format_reasoning_preview(record.get('reasoning_content'))}"
        yield ""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze benchmark results.")
    parser.add_argument(
        "--results-path",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help="Path to benchmark results file or directory.",
    )
    parser.add_argument(
        "--constraint",
        type=str,
        default=None,
        help="Only analyze records that include the given constraint name.",
    )
    parser.add_argument(
        "--show-generated",
        action="store_true",
        help="Show normalized generated responses after the summary table.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    resolved_path = Path(args.results_path)
    records = load_records(resolved_path)
    records = filter_records_by_constraint(records, args.constraint)
    summary = summarize_records(records)
    print_summary(summary)
    if args.show_generated:
        print()
        print("Generated responses:")
        print("-------------------")
        for line in format_generated_responses(records):
            print(line)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
