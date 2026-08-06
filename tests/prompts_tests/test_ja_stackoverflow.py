from __future__ import annotations

import json
from typing import TYPE_CHECKING

import zstandard as zstd

from jfbench.prompts import ja_stackoverflow
from jfbench.prompts import JaStackoverflowPrompt


if TYPE_CHECKING:
    from pathlib import Path


class DummyConstraint:
    def evaluate(self, value: str) -> tuple[bool, None]:
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        return "follow the dummy constraint"

    @property
    def group(self) -> str:
        return "Test"

    def rewrite_instructions(self) -> str:
        return "please satisfy the dummy constraint."

    @property
    def competitives(self) -> list[str]:
        return []

    def to_serializable_kwargs(self) -> dict[str, object]:
        return {}


def test_ja_stackoverflow_prompt_text_includes_constraints() -> None:
    prompt = JaStackoverflowPrompt("answer carefully")
    rendered = prompt.text([DummyConstraint()])
    assert "answer carefully" in rendered
    assert "follow the dummy constraint" in rendered


def test_get_all_ja_stackoverflow_prompts_reads_zst_file(tmp_path: Path) -> None:
    data_file = tmp_path / "ja_stackoverflow.jsonl.zst"
    compressor = zstd.ZstdCompressor()
    with data_file.open("wb") as fh:
        with compressor.stream_writer(fh) as writer:
            for idx in range(3):
                payload = {
                    "id": f"sample-{idx}",
                    "instruction": f"サンプル指示{idx}",
                    "output": f"出力{idx}",
                }
                writer.write(json.dumps(payload).encode("utf-8"))
                writer.write(b"\n")

    original_path = ja_stackoverflow.DATA_PATH
    ja_stackoverflow.DATA_PATH = data_file
    try:
        prompts = ja_stackoverflow.get_all_ja_stackoverflow_prompts()
    finally:
        ja_stackoverflow.DATA_PATH = original_path

    assert [prompt.document for prompt in prompts] == [
        "サンプル指示0",
        "サンプル指示1",
        "サンプル指示2",
    ]
