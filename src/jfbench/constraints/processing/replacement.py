import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class ReplacementProcessingConstraint(ConstraintGroupMixin):
    def __init__(
        self, document: str, start: int, end: int, keyword: str, *, seed: int | None = None
    ) -> None:
        super().__init__(seed=seed)
        self.document = document.rstrip("\r\n")
        self.start = start
        self.end = end
        self.keyword = keyword

    def evaluate(self, value: str) -> ConstraintEvaluation:
        replaced = self.document[: self.start] + self.keyword + self.document[self.end + 1 :]
        if replaced not in value:
            reason = "[Replacement Processing] Expected replacement string not found."
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"回答には指示文の{self.start}文字目から{self.end}文字目までを{self.keyword}に置換した文字列全体を含めてください。",
            f"指示文の{self.start}文字目〜{self.end}文字目を{self.keyword}に置き換えた文章を出力してください。",
            f"指示文の指定範囲（{self.start}〜{self.end}文字目）を{self.keyword}に差し替えたテキスト全体を回答に含めてください。",
            f"指示文の{self.start}〜{self.end}文字目を{self.keyword}に変更した内容を示してください。",
            f"指示文の{self.start}文字目から{self.end}文字目までの該当箇所を{self.keyword}へ置換した完全な文字列を回答してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return f"指示文の{self.start}〜{self.end}文字目を{self.keyword}に置換した上で全文を含めてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return (
            f"指示文の指定範囲({self.start}文字目〜{self.end}文字目)を{self.keyword}に置き換えた文字列を回答に含めてください。\n"
            "### 指示文\n"
            f"{self.document}"
        )
