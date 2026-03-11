import logging
import re
from typing import List
from typing import Tuple
from typing import TYPE_CHECKING

from jfbench.constraints._group import ConstraintGroupMixin


if TYPE_CHECKING:
    from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class DecimalPlacesNotationConstraint(ConstraintGroupMixin):
    def __init__(self, required_decimal_places: int, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.required_decimal_places = required_decimal_places

    def evaluate(self, value: str) -> "ConstraintEvaluation":
        valid, mismatches = _check_all_decimals_have_decimal_places(
            value, self.required_decimal_places
        )
        last_reason: str | None = None
        if not valid:
            for decimal_str, actual_decimal_places in mismatches:
                last_reason = (
                    "[Decimal Places Notation] Decimal "
                    f"{decimal_str} has {actual_decimal_places} decimal places, "
                    f"expected {self.required_decimal_places}"
                )
                logger.info(last_reason)
        if valid:
            return True, None
        if last_reason is None:
            last_reason = "[Decimal Places Notation] Validation failed."
            logger.info(last_reason)
        return False, last_reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"出力には少なくとも1つの小数を含め、その小数は小数点以下{self.required_decimal_places}桁にそろえてください。",
            f"小数を必ず1件以上示し、小数点以下の桁数を{self.required_decimal_places}桁に固定してください。",
            f"小数を記述する際は小数第{self.required_decimal_places}位までを維持し、値を少なくとも1つ入れてください。",
            f"回答には{self.required_decimal_places}桁目まで記載した小数を1件以上含めてください。",
            f"小数点以下の桁数は常に{self.required_decimal_places}桁に統一し、そうした小数を必ず出力に含めてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return f"小数は小数点以下{self.required_decimal_places}桁で1つ以上含めてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return (
            f"小数を少なくとも1つ含め、小数点以下{self.required_decimal_places}桁に"
            "そろえるように数値を書き直してください。"
        )


def _decimal_places_of_mantissa(mantissa: str) -> int:
    """
    Return the number of decimal places (digits after the decimal point) in the
    mantissa that contains a decimal point (e.g., '0.00340', '12.', '.50').
    """
    # Normalize full-width symbols to ASCII.
    mantissa = mantissa.replace("\u2212", "-").replace("\uff0e", ".")
    if "." not in mantissa:
        raise ValueError("mantissa must contain a decimal point")

    _, frac_part = mantissa.split(".", 1)
    frac_digits = re.sub(r"\D", "", frac_part)
    return len(frac_digits)


def _mantissa_from_match_token(token: str) -> str:
    """
    Return the mantissa (the part that contains the decimal point) without the
    exponent from a matched token.
    Example: '+3.00e-2' -> '3.00'
    """
    token = token.replace("\u2212", "-").replace("\uff0e", ".")
    # Drop the exponent part that follows the first e/E.
    m = re.match(r"^[+\-]?((?:\d+\.\d*|\.\d+))([eE][+\-]?\d+)?$", token)
    if not m:
        # Fallback: capture the first chunk that contains a decimal point.
        fallback = re.search(r"\d*\.\d*", token)
        if fallback is None:
            raise ValueError(f"Decimal mantissa not found in token: {token}")
        return fallback.group(0)
    return m.group(1)


def _find_decimals(text: str) -> List[str]:
    """Return every decimal (a number that contains a decimal point) in the text."""
    decimal_re = re.compile(
        r"(?<![\w/])"  # Do not start next to alphanumerics or '/'.
        r"[+\-\u2212]?"
        r"(?:\d+\.\d*|\.\d+)"  # Number with a decimal point (allows 12. or .5).
        r"(?:[eE][+\-\u2212]?\d+)?"  # Optional exponent for scientific notation.
        r"(?!\w)"  # Do not continue with alphanumerics.
    )
    text = text.replace("\uff0e", ".").replace("\u2212", "-")
    return [m.group(0) for m in decimal_re.finditer(text)]


def _check_all_decimals_have_decimal_places(
    text: str, required_decimal_places: int
) -> Tuple[bool, List[Tuple[str, int]]]:
    """
    Determine whether every number with a decimal point in text has the expected number
    of decimal places.
    Returns: (result, list of mismatches as [(token, actual decimal places)]).
    """
    mismatches: List[Tuple[str, int]] = []
    for tok in _find_decimals(text):
        mant = _mantissa_from_match_token(tok)
        k = _decimal_places_of_mantissa(mant)
        if k != required_decimal_places:
            mismatches.append((tok, k))
    return (len(mismatches) == 0, mismatches)
