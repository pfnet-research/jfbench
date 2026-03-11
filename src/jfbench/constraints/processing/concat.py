import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class ConcatProcessingConstraint(ConstraintGroupMixin):
    def __init__(self, document: str, times: int, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.document = document.rstrip("\r\n")
        self.times = times

    def evaluate(self, value: str) -> ConstraintEvaluation:
        expected = self.document * self.times
        if expected not in value:
            reason = (
                "[Concat Processing] Concatenated document not found the required number of times."
            )
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"与えられた指示文を{self.times}回繋げた文字列を回答に含めてください。",
            f"テキストを{self.times}連結した内容をそのまま回答に含めて出力してください。",
            f"指定文書を{self.times}回続けて並べた文字列を回答に含めてください。",
            f"指示文を{self.times}回連続で結合した形で含めてください。",
            f"指示文と同じ内容を{self.times}回繰り返した連結文字列を含めた上で回答してください。",
        ]
        if train_or_test == "train":
            return (
                self._random_instruction(templates)
                + "特に改行やスペースといった区切り文字は含めないでください。"
            )
        if train_or_test == "test":
            return f"指示文を{self.times}回連結した文字列をそのまま含めてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return (
            f"指示文を{self.times}回連結した文字列を入れてください。\n### 指示文\n{self.document}"
        )
