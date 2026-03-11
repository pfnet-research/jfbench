import logging
import re

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class UnitNotationConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        # SI prefix: da, Y, Z, E, P, T, G, M, k, h, d, c, m, u/μ, n, p, f, a, z, y
        prefix = r"(?:da|[YZEPTGMkhdcmunpfazyμu])?"
        # SI base units + g (to allow kg, mg, μg, etc.)
        unit = r"(?:m|g|s|A|K|mol|cd)"
        pattern = rf"\d+(?:\.\d+)?\s*{prefix}{unit}(?![A-Za-z0-9_])"

        if not re.search(pattern, value):
            reason = "[Unit Notation] Numbers must include SI units (with prefixes)."
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "出力には少なくとも1つ数値を含め、その数値には必ずmやkgなどのSI単位を付けてください。",
            "数字の後ろにSI単位（m, kg, s など）を必ず付けた形の数値を1件以上示してください。",
            "SI単位（m, kg, s など）を伴う数値を少なくとも一つ出力に含めてください。",
            "数値を出す際は対応するSI単位を続けて示し、そのような値を1件以上入れてください。",
            "m・kg・sなどのSI単位を付記した数値を最低1つは書いてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "SI単位付きの数値を含めてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "出力を修正し、少なくとも1つの数値の後ろにSI単位を付けてください。"
