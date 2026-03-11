import logging
import math

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class SplitProcessingConstraint(ConstraintGroupMixin):
    def __init__(self, document: str, parts: int, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        if parts <= 0:
            raise ValueError("parts must be greater than zero.")
        self.document = document.rstrip("\r\n")
        self.parts = parts

    def evaluate(self, value: str) -> ConstraintEvaluation:
        chunk = math.ceil(len(self.document) / self.parts)
        segments = [self.document[i : i + chunk] for i in range(0, len(self.document), chunk)]
        expected = "[\n" + ",\n".join(segments) + "\n]"
        if expected not in value:
            reason = "[Split Processing] Expected split representation not found in the output."
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"指示文を同じ長さで{self.parts}分割し、[\n<文字列1>,\n<文字列2>,\n…\n<文字列{self.parts}>] の形式の分割結果を回答に含めてください。",
            f"指示文を{self.parts}等分したリスト([\n<文字列1>,\n<文字列2>,\n…\n<文字列{self.parts}>])を回答に含めてください。",
            f"指示文を長さが等しくなるように{self.parts}分割した各断片を[\n<文字列1>,\n<文字列2>,\n…\n<文字列{self.parts}>] の書式で示し、そのリストを出力に含めてください。",
            f"指示の文章を{self.parts}つの同じ長さに分けた結果を[\n<文字列1>,\n<文字列2>,\n…\n<文字列{self.parts}>] 形式で示し、そのリストを回答に含めてください。",
            f"指示文を均等に{self.parts}分した文字列リスト([\n<文字列1>,\n<文字列2>,\n…\n<文字列{self.parts}>])を回答に含めてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return f"指示文を同じ長さで{self.parts}分割したリストを[\n<文字列1>,\n<文字列2>,\n…\n<文字列{self.parts}>]の形式で入れてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return (
            f"指示文を{self.parts}分割したリスト形式を回答に含めてください。\n"
            "### 指示文\n"
            f"{self.document}"
        )
