import logging
import re
from typing import Iterable
from typing import TYPE_CHECKING
import unicodedata

from jfbench.constraints._group import ConstraintGroupMixin


if TYPE_CHECKING:
    from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class GroupingNotationConstraint(ConstraintGroupMixin):
    def __init__(self, max_group_size: int, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.max_group_size = max_group_size

    def evaluate(self, value: str) -> "ConstraintEvaluation":
        all_ok, mismatches = _check_all_integers_are_grouped(
            value,
            max_group_size=self.max_group_size,
            allowed_separators=[",", "_", " "],
        )
        last_reason: str | None = None
        for tok, reason in mismatches:
            last_reason = (
                "[Grouping Notation] Token "
                f"{tok!r} does not conform to {self.max_group_size}-digit grouping: {reason}"
            )
            logger.info(last_reason)
        if all_ok:
            return True, None
        if last_reason is None:
            last_reason = "[Grouping Notation] Validation failed."
            logger.info(last_reason)
        return False, last_reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"整数は右から{self.max_group_size}桁ごとにカンマ(,)またはアンダースコア(_)で区切ってください。",
            f"{self.max_group_size}桁単位で区切り記号(, or _)を入れ、整った位取りで出力してください。",
            f"桁数が{self.max_group_size}を超える整数には、右から{self.max_group_size}桁ごとに区切り記号を挿入してください。",
            f"整数は{self.max_group_size}桁区切りルール（,または_）を守って表記してください。",
            f"右端から数えて{self.max_group_size}桁ごとに同じ区切り記号を入れた表記のみを許可します。",
        ]
        if train_or_test == "train":
            return "出力には少なくとも一つの整数を含めてください。" + self._random_instruction(
                templates
            )
        if train_or_test == "test":
            return (
                f"整数は{self.max_group_size}桁区切りで書いてください。一つ以上は含めてください。"
            )
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return f"整数の桁区切りを見直し、右側から{self.max_group_size}桁ごとに同じ区切り記号(, or _)で整形してください。"


# Regex to extract integer (no decimal point) candidates
# - Allow optional leading sign (+/-/−)
# - Consists only of digits and allowed separators
_INT_TOKEN_RE = re.compile(
    r"[+\-−]?"
    r"(?:\d(?:[\d,_\s]*\d)?)"
)

# Normalize some space-like characters to a regular ASCII space
_SPACE_LIKE = {
    "\u00a0",  # NBSP
    "\u2009",  # Thin Space
    "\u202f",  # Narrow No-Break Space
}


def _normalize_spaces(s: str) -> str:
    for ch in _SPACE_LIKE:
        s = s.replace(ch, " ")
    return s


def _detect_separators(digits: str, allowed: set[str]) -> set[str]:
    return {ch for ch in digits if ch in allowed}


def _check_grouping_core(digits_only_and_seps: str, max_group_size: int, sep_char: str) -> bool:
    """
    Validate grouping from the right in chunks of max_group_size digits.
    Rule: each group from the right must have exactly max_group_size digits,
    and only the leftmost group may have 1..max_group_size digits.
    """
    # Disallow consecutive separators and leading/trailing separators
    if digits_only_and_seps.startswith(sep_char) or digits_only_and_seps.endswith(sep_char):
        return False
    if sep_char * 2 in digits_only_and_seps:
        return False

    parts = digits_only_and_seps.split(sep_char)
    # All parts must consist of digits only (non-empty)
    if not all(part.isdigit() and len(part) > 0 for part in parts):
        return False

    # Validate group sizes from the right
    for idx_from_right, part in enumerate(reversed(parts), start=1):
        if idx_from_right == len(parts):  # leftmost group
            if not (1 <= len(part) <= max_group_size):
                return False
        else:  # groups to the right must be exactly max_group_size
            if len(part) != max_group_size:
                return False
    return True


def _find_integer_tokens(text: str) -> list[str]:
    """
    Extract all "integer tokens" (no decimal point) from the text as strings.
    Examples: '+12,345', '-1_000', '1 234 567', '42'
    Note: decimals like '123.45' are not extracted even if digits are matched.
    """
    # Normalize (e.g., full-width to half-width)
    text = unicodedata.normalize("NFKC", text)
    text = _normalize_spaces(text)
    tokens: list[str] = []
    for m in _INT_TOKEN_RE.finditer(text):
        start, end = m.span()
        # Skip if the match looks like part of a decimal number
        if start > 0 and text[start - 1] == ".":
            continue
        if end < len(text) and text[end] == ".":
            continue
        tokens.append(m.group(0))
    return tokens


def _check_all_integers_are_grouped(
    text: str,
    max_group_size: int = 3,
    allowed_separators: Iterable[str] = (",", "_", " "),
) -> tuple[bool, list[tuple[str, str]]]:
    """
    Check whether all integers in the text follow max_group_size-digit grouping.
    - A token may use only one kind of separator from allowed_separators (no mixing)
    - If the digit length exceeds max_group_size, a separator is required.
    - If separators are present, enforce strict right-to-left grouping:
      every group has exactly max_group_size digits except the leftmost group,
      which may have 1..max_group_size digits.
    Returns: (True if all conform, list of (token, reason) mismatches otherwise)
    """
    text = unicodedata.normalize("NFKC", text)
    text = _normalize_spaces(text)
    allowed = set(allowed_separators)

    mismatches: list[tuple[str, str]] = []

    for tok in _find_integer_tokens(text):
        core = tok.lstrip("+−-")  # remove sign
        # Detect which separator(s) are used
        seps = _detect_separators(core, allowed)

        # Mixing separators is not allowed
        if len(seps) > 1:
            mismatches.append((tok, "mixed_separators"))
            continue

        if len(seps) == 0:
            # No separators
            digits_only = "".join(ch for ch in core if ch.isdigit())
            if not digits_only:  # just in case
                mismatches.append((tok, "not_digits"))
                continue
            if len(digits_only) > max_group_size:
                mismatches.append((tok, "missing_separators"))
            # Otherwise OK (digit length <= max_group_size)
            continue

        # With separators: validate grouping
        sep_char = next(iter(seps))
        if not _check_grouping_core(core, max_group_size, sep_char):
            mismatches.append((tok, "bad_grouping"))

    return (len(mismatches) == 0, mismatches)
