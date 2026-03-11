import logging
from typing import Any

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints._utils import split_words
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class NewlineFormatIfbenchConstraint(ConstraintGroupMixin):
    def __init__(self, tokenizer: Any | None = None, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._tokenizer = tokenizer

    def _tokenize(self, line: str) -> list[str]:
        stripped = line.strip()
        if self._tokenizer is not None:
            return [token.surface for token in self._tokenizer.tokenize(stripped)]
        return [word for word in split_words(stripped) if word]

    def evaluate(self, value: str) -> ConstraintEvaluation:
        lines = [line for line in value.splitlines() if line.strip()]
        for line in lines:
            tokens = self._tokenize(line)
            if len(tokens) != 1:
                reason = (
                    "[Newline Format] Each line should contain exactly one word; "
                    f"found {len(tokens)}."
                )
                logger.info(reason)
                return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "各単語を改行して記入してください。",
            "単語ごとに改行し、一行に一語だけを配置してください。",
            "スペースで区切らず、単語ごとに改行してください。",
            "一行に一単語となるように改行を入れてください。",
            "各単語を別の行に配置してください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "単語ごとに改行し、一行に一語だけになるようにしてください。"
