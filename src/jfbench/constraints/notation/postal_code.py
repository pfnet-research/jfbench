import logging
import re
import unicodedata

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class NoHyphenJpPostalCodeNotationConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        ok, matches = _validate_jp_postal_code_presence(value, no_hyphen=True)
        if not ok:
            reason = (
                "[No Hyphen JP Postal Code Notation] Found no valid JP postal code "
                f"(no hyphen). Matches found: {matches}"
            )
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "出力にハイフン無しの日本の郵便番号（例: 1234567）を少なくとも1つ含めてください。",
            "7桁連続の郵便番号を掲載し、ハイフンは使わないでください。",
            "日本の郵便番号をハイフンなし7桁で表記して盛り込んでください。",
            "1234567のように7桁続く郵便番号を必ず書いてください。",
            "連続7桁の数字のみで表す日本の郵便番号を出力に含めてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "ハイフンなしの7桁郵便番号を入れてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "出力を修正し、7桁連続のハイフンなし郵便番号（例：1234567）を必ず含めてください。"


class WithHyphenJpPostalCodeNotationConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        ok, matches = _validate_jp_postal_code_presence(value, no_hyphen=False)
        if not ok:
            reason = (
                "[With Hyphen JP Postal Code Notation] Found no valid JP postal code "
                f"(with hyphen). Matches found: {matches}"
            )
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "出力にハイフン付きの日本の郵便番号（例: 123-4567）を少なくとも1つ含めてください。",
            "3桁-4桁形式で書かれた郵便番号を必ず掲載してください。",
            "日本の郵便番号を123-4567のようにハイフンで区切った形で含めてください。",
            "ハイフンありの7桁郵便番号を1つは入れて回答してください。",
            "日本の郵便番号はxxx-xxxx形式で記述し、その形式の値を出力してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "ハイフン付きの郵便番号を含めてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "出力を修正して、3桁-4桁形式の日本の郵便番号（例：123-4567）を必ず含めてください。"


# Dash-like characters to normalize into ASCII '-'
_DASHES = "−‐‒–—―－"  # minus/hyphen variants (not including ASCII '-')

# Strict JP postal-code regexes:
#  - Hyphenated: exactly 3 digits, '-', 4 digits; not adjacent to other digits
#  - Plain: exactly 7 consecutive digits; not adjacent to other digits
_JP_POSTAL_HYPHEN_RE = re.compile(r"(?<!\d)\d{3}-\d{4}(?!\d)")
_JP_POSTAL_PLAIN_RE = re.compile(r"(?<!\d)\d{7}(?!\d)")


def _normalize(text: str) -> str:
    """Normalize text: NFKC (full-width -> ASCII) and unify dash variants to '-'."""
    t = unicodedata.normalize("NFKC", text)
    for d in _DASHES:
        t = t.replace(d, "-")
    return t


def _validate_jp_postal_code_presence(text: str, *, no_hyphen: bool) -> tuple[bool, list[str]]:
    """
    Validate that the text contains at least one JP postal code, with format policy:
      - no_hyphen=True  -> requires plain 'xxxxxxx' (7 consecutive digits)
      - no_hyphen=False -> requires hyphenated 'xxx-xxxx'

    Returns:
      (ok, matches)
        ok: True if at least one match is found in the required format
        matches: list of matched postal-code strings (after normalization)
    """
    t = _normalize(text)
    pattern = _JP_POSTAL_PLAIN_RE if no_hyphen else _JP_POSTAL_HYPHEN_RE
    matches = [m.group(0) for m in pattern.finditer(t)]
    return (len(matches) >= 1, matches)
