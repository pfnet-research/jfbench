from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence


_ALLOWED_STATUSES = {"pass", "fail", "unknown"}
_STATUS_PATTERN = re.compile(
    r"\[\s*(?P<index>\d+)\s*\].*?chosen=(?P<chosen>\w+),\s*rejected=(?P<rejected>\w+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RecordStatus:
    index: int
    chosen: str
    rejected: str


def load_constraint_statuses(path: Path) -> list[RecordStatus]:
    text = path.read_text(encoding="utf-8")
    statuses = [
        RecordStatus(
            index=int(match.group("index")),
            chosen=match.group("chosen").lower(),
            rejected=match.group("rejected").lower(),
        )
        for match in _STATUS_PATTERN.finditer(text)
    ]
    if not statuses:
        raise ValueError("No constraint statuses found in analysis result.")
    statuses.sort(key=lambda status: status.index)
    for expected_index, status in enumerate(statuses):
        if status.index != expected_index:
            raise ValueError(
                f"Expected record index {expected_index} but found {status.index} in analysis result."
            )
        if status.chosen not in _ALLOWED_STATUSES or status.rejected not in _ALLOWED_STATUSES:
            raise ValueError(
                f"Unsupported status pair chosen={status.chosen}, rejected={status.rejected}."
            )
    print(f"Loaded {len(statuses)} constraint statuses from {path}.")
    return statuses


def _should_keep(status: RecordStatus) -> bool:
    if status.chosen in {"unknown", "fail"}:
        return False
    if status.rejected in {"unknown", "pass"}:
        return False
    return True


def screen_dataset(
    *,
    input_path: Path,
    analysis_result_path: Path,
    output_path: Path,
    allow_overwrite: bool = False,
) -> None:
    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if input_path.suffix.lower() != ".jsonl":
        raise ValueError("Input file must be a .jsonl file.")
    if output_path.exists() and output_path.is_dir():
        raise ValueError("Output path must be a file, not a directory.")
    analysis_result_path = analysis_result_path.resolve()
    statuses = load_constraint_statuses(analysis_result_path)
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    same_file = input_path == output_path
    if same_file and not allow_overwrite:
        raise ValueError("allow_overwrite must be set when screening in place.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    status_index = 0
    total_statuses = len(statuses)
    write_target = output_path
    tmp_file: Path | None = None
    if same_file:
        tmp_file = output_path.with_suffix(f"{output_path.suffix}.tmp")
        write_target = tmp_file
    n_screened = 0
    with (
        input_path.open("r", encoding="utf-8") as reader,
        write_target.open(
            "w",
            encoding="utf-8",
        ) as writer,
    ):
        for line in reader:
            if status_index >= total_statuses:
                raise ValueError("Analysis result has fewer entries than dataset records.")
            status = statuses[status_index]
            status_index += 1
            if _should_keep(status):
                writer.write(line)
                n_screened += 1
    print(f"Screened {n_screened} records out of {status_index} total records.")
    if tmp_file is not None:
        tmp_file.replace(output_path)
    if status_index != total_statuses:
        raise ValueError("Analysis result has more entries than dataset records.")
    print(f"Screened dataset saved to {output_path}.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--analysis-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    screen_dataset(
        input_path=args.input,
        analysis_result_path=args.analysis_result,
        output_path=args.output,
        allow_overwrite=bool(args.allow_overwrite),
    )


__all__ = [
    "RecordStatus",
    "load_constraint_statuses",
    "main",
    "parse_args",
    "screen_dataset",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
