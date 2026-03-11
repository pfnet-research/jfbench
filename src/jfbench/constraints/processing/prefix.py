import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class PrefixProcessingConstraint(ConstraintGroupMixin):
    def __init__(self, prefix: str, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.prefix = prefix

    def evaluate(self, value: str) -> ConstraintEvaluation:
        if not value.startswith(self.prefix):
            reason = (
                f"[Prefix Processing] Output {value!r} does not start with required prefix "
                f"{self.prefix!r}."
            )
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"出力の先頭を必ず「{self.prefix}」で始めてください。",
            f"回答は{self.prefix}というフレーズで始まる必要があります。",
            f"文字列冒頭に「{self.prefix}」を置き、その後に内容を続けてください。",
            f"最初の文字列を{self.prefix}に固定し、その後に本文を記述してください。",
            f"書き出しは常に{self.prefix}から始めてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return f"先頭は「{self.prefix}」で始めてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return f"先頭を「{self.prefix}」で始まるように書き換えてください。"
