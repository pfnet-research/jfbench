import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class SuffixProcessingConstraint(ConstraintGroupMixin):
    def __init__(self, suffix: str, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.suffix = suffix

    def evaluate(self, value: str) -> ConstraintEvaluation:
        if not value.endswith(self.suffix):
            reason = (
                f"[Suffix Processing] Output {value!r} does not end with required suffix "
                f"{self.suffix!r}."
            )
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"出力の末尾は必ず「{self.suffix}」で締めてください。",
            f"回答は{self.suffix}という接尾辞で終了させてください。",
            f"最後の文字列を「{self.suffix}」に固定してから本文を終えてください。",
            f"結果は「{self.suffix}」で終わるよう強制してください。",
            f"末尾に必ず{self.suffix}を置いた状態で出力してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return f"末尾は「{self.suffix}」で終えてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return f"出力の末尾を修正し、必ず「{self.suffix}」で終わるようにしてください。"
