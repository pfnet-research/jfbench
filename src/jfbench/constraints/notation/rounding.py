from decimal import Decimal
from decimal import ROUND_HALF_UP
import logging
import re

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class RoundingNotationConstraint(ConstraintGroupMixin):
    def __init__(self, digits: int, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.digits = digits

    def evaluate(self, value: str) -> ConstraintEvaluation:
        numbers = re.findall(r"-?\d+\.\d+", value)
        if not numbers:
            reason = "[Rounding Notation] No decimal numbers to check."
            logger.info(reason)
            return False, reason

        for number in numbers:
            rounded = Decimal(number).quantize(
                Decimal(f"1.{'0' * self.digits}"),
                rounding=ROUND_HALF_UP,
            )
            if Decimal(number) != rounded:
                reason = (
                    "[Rounding Notation] A number does not follow the requested rounding rule."
                )
                logger.info(reason)
                return False, reason

        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"出力には少なくとも1つ小数を含め、その小数は小数第{self.digits}位で四捨五入した値にしてください。",
            f"小数を1件以上示し、小数点以下{self.digits}桁目で四捨五入した結果に統一してください。",
            f"小数は四捨五入処理を行い、小数第{self.digits}位までで示したものを少なくとも一つ入れてください。",
            f"数値を四捨五入して小数第{self.digits}位の精度に整えた小数を1件以上含めてください。",
            f"小数点以下{self.digits}桁に四捨五入した結果となる小数を必ず出力に含めてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return f"小数第{self.digits}位で四捨五入した小数を入れてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return (
            f"小数を少なくとも1つ残し、その値を小数第{self.digits}位で四捨五入に揃えてください。"
        )
