from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Iterator
from typing import TYPE_CHECKING

import zstandard as zstd

from jfbench._data import DATA_DIR


if TYPE_CHECKING:
    from jfbench.protocol import Constraint


DATA_PATH = DATA_DIR / "ja_stackoverflow_train.jsonl.zst"


class JaStackoverflowPrompt:
    def __init__(self, instruction: str) -> None:
        self._instruction = instruction

    def text(self, constraints: list[Constraint], *, train_or_test: str = "train") -> str:
        constraints_instructions = "\n".join(
            f"- {constraint.instructions(train_or_test=train_or_test)}"
            for constraint in constraints
        )
        return (
            "# 指示文\n"
            f"{self._instruction}\n\n"
            "# 回答に関する注意事項\n"
            "特に指定がなければ、日本語で回答してください。ただし、英語での回答が求められている場合は英語で回答してください。\n"
            "ただし、以下の制約条件を全て守ってください。\n"
            f"{constraints_instructions}"
        )

    @property
    def document(self) -> str:
        return self._instruction


def get_all_ja_stackoverflow_prompts(
    dataset_path: str | None = None,
) -> list[JaStackoverflowPrompt]:
    path = Path(dataset_path) if dataset_path else Path(DATA_PATH)
    return [JaStackoverflowPrompt(instruction) for instruction in _iter_instructions(path)]


def _iter_instructions(path: Path) -> Iterator[str]:
    decompressor = zstd.ZstdDecompressor()
    with path.open("rb") as raw:
        with decompressor.stream_reader(raw) as reader:
            with io.TextIOWrapper(reader, encoding="utf-8") as text_stream:
                for line in text_stream:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    record = json.loads(stripped)
                    instruction = record.get("instruction")
                    if instruction:
                        yield str(instruction)
