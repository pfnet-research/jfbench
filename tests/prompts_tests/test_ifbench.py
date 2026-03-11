from __future__ import annotations

from pathlib import Path

from jfbench.prompts import ifbench as ifbench_module
from jfbench.prompts import IFBenchPrompt


class DummyConstraint:
    def __init__(self) -> None:
        self.last_mode: str | None = None

    def evaluate(self, value: str) -> tuple[bool, None]:
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        self.last_mode = train_or_test
        return f"ダミー制約条件の指示({train_or_test})"

    @property
    def group(self) -> str:
        return "Test"

    def rewrite_instructions(self) -> str:
        return "ダミー制約条件を満たすように修正してください。"

    @property
    def competitives(self) -> list[str]:
        return []

    def to_serializable_kwargs(self) -> dict[str, object]:
        return {}


def test_ifbench_prompt_text_returns_stored_value() -> None:
    prompt = IFBenchPrompt("テストプロンプト")
    assert "テストプロンプト" in prompt.text([DummyConstraint()])


def test_ifbench_prompt_uses_test_instructions() -> None:
    prompt = IFBenchPrompt("テストプロンプト")
    constraint = DummyConstraint()

    rendered = prompt.text([constraint], train_or_test="test")

    assert "(test)" in rendered
    assert constraint.last_mode == "test"


def test_ifbench_default_data_path_uses_repository_relative_location() -> None:
    expected_suffix = Path("data/ifbench_ja_translated.jsonl")
    assert str(ifbench_module.DATA_PATH).endswith(str(expected_suffix))
