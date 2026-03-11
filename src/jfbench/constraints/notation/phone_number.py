import logging
import re
import unicodedata

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class NoHyphenPhoneNumberNotationConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        is_all_valid, mismatches = _check_phones_are_10or11digits_no_hyphen(value, no_hyphen=True)
        last_reason: str | None = None
        if not is_all_valid:
            for token, reason in mismatches:
                last_reason = (
                    f"[Phone Number Notation] Token {token!r} is invalid for reason: {reason}"
                )
                logger.info(last_reason)
        if is_all_valid:
            return True, None
        if last_reason is None:
            last_reason = "[Phone Number Notation] Validation failed."
            logger.info(last_reason)
        return False, last_reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "出力にハイフン無しの10桁または11桁の電話番号を少なくとも1つ含めてください。",
            "10～11桁の数字のみで構成された電話番号（ハイフンなし）を必ず掲載してください。",
            "数字だけで書かれた連続10桁/11桁の電話番号を出力に含めてください。",
            "電話番号はハイフンを使わず連続した10桁または11桁で表記し、最低1件示してください。",
            "少なくとも1つ、ハイフン無しで10桁か11桁の電話番号を掲載してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "ハイフンなしの10桁または11桁の電話番号を入れてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return (
            "出力を修正し、ハイフン無しの10桁または11桁の電話番号を少なくとも1つ含めてください。"
        )


class WithHyphenPhoneNumberNotationConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        is_all_valid, mismatches = _check_phones_are_10or11digits_no_hyphen(
            value,
            no_hyphen=False,
        )
        last_reason: str | None = None
        if not is_all_valid:
            for token, reason in mismatches:
                last_reason = (
                    f"[Phone Number Notation] Token {token!r} is invalid for reason: {reason}"
                )
                logger.info(last_reason)
        if is_all_valid:
            return True, None
        if last_reason is None:
            last_reason = "[Phone Number Notation] Validation failed."
            logger.info(last_reason)
        return False, last_reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "出力には10桁または11桁の数字をハイフンで区切った電話番号を最低1つ含めてください。",
            "電話番号は必ずハイフン区切りで記述し、10～11桁の数字を含む形にしてください。",
            "少なくとも1件、ハイフン付きの10桁/11桁電話番号を含めた出力にしてください。",
            "電話番号は桁数10または11で、ハイフンを含む形式のみ許可されます。必ず1つ以上書いてください。",
            "ハイフンで区切られた電話番号（10 or 11桁の数字）を回答に含めてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "ハイフン付きの10桁または11桁電話番号を入れてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "出力を修正して、桁数10または11の数字をハイフンで区切った電話番号を必ず含めてください。"


# Hyphen/dash variants to normalize into ASCII '-'
_DASHES = "-‐‒–—―－"  # hyphen-minus, hyphen, figure/en/em dash, horizontal bar, full-width, etc.


def _normalize(text: str) -> str:
    """
    Normalize text:
      - NFKC (e.g., full-width digits -> ASCII digits)
      - unify many dash variants to ASCII '-'
    """
    t = unicodedata.normalize("NFKC", text)
    for ch in _DASHES:
        t = t.replace(ch, "-")
    return t


# A loose "phone-like" candidate:
#  - starts with optional '+', then a digit
#  - contains digits / spaces / hyphens / parentheses
#  - ends with a digit
#  - long enough to plausibly be a phone (>= ~10 chars overall)
_PHONE_CANDIDATE_RE = re.compile(r"(?<!\w)\+?\d[\d\-\s()]{8,}\d(?!\w)")


def _find_phone_candidates(text: str) -> list[str]:
    """
    Extract raw phone-like tokens from text (after normalization).
    This intentionally over-collects; strict validation is done later.
    """
    t = _normalize(text)
    return [m.group(0) for m in _PHONE_CANDIDATE_RE.finditer(t)]


def _is_valid_token(token: str, no_hyphen: bool) -> tuple[bool, str]:
    """
    Validate a single token against the hyphen policy:
      - no_hyphen=True  -> must be exactly 10 or 11 consecutive digits
      - no_hyphen=False -> must contain at least one hyphen, consist of digits and hyphens only,
                           not start/end with '-', no '--', and total digits must be 10 or 11
      - all cases must start with '0' to match Japanese phone numbers
    Returns: (is_valid, reason_if_invalid)
    """
    # Fast path for strict "digits only"
    if no_hyphen is True:
        if re.fullmatch(r"0\d{9,10}", token):
            return True, ""
        # Diagnose
        digits_only = re.sub(r"\D", "", token)
        if len(digits_only) not in (10, 11):
            return False, "wrong_length"
        if digits_only and not digits_only.startswith("0"):
            return False, "must_start_with_0"
        if "-" in token:
            return False, "has_hyphen"
        if re.search(r"[^\d]", token):
            return False, "has_non_digit"
        return False, "invalid"

    # Allow only digits and hyphens (no spaces/parentheses/plus/etc.)
    if re.search(r"[^\d\-]", token):
        return False, "has_non_digit"

    # Basic hyphen structural checks (shared by no_hyphen in {False, None})
    if token.startswith("-") or token.endswith("-") or "--" in token:
        return False, "bad_hyphen_structure"

    digits_only = token.replace("-", "")
    if not digits_only.isdigit():
        return False, "has_non_digit"
    if len(digits_only) not in (10, 11):
        return False, "wrong_length"
    if not digits_only.startswith("0"):
        return False, "must_start_with_0"

    if no_hyphen is False and "-" not in token:
        return False, "missing_hyphen"

    return True, ""


def _check_phones_are_10or11digits_no_hyphen(
    text: str, no_hyphen: bool = True
) -> tuple[bool, list[tuple[str, str]]]:
    """
    Validate that every phone-like token in the text is a 10 or 11 digit phone number,
    with a selectable hyphen policy.

    Args:
      text: input text.
      no_hyphen:
        - True  : enforce "no hyphen" (exactly 10 or 11 consecutive digits)
        - False : enforce "must contain hyphen(s)" (digits + '-' only; grouping free)

    Returns:
      (is_all_valid, mismatches)
        - is_all_valid: True if no violations found
        - mismatches : list of (token, reason)
            reasons:
              'wrong_length'         -> digit count is not 10 or 11
              'has_hyphen'           -> hyphen present when forbidden
              'missing_hyphen'       -> hyphen required but not present
              'has_non_digit'        -> contains non digit/hyphen chars (space, parentheses, '+', etc.)
              'bad_hyphen_structure' -> starts/ends with '-' or contains '--'
              'no_phone_found'       -> no phone-like token found
              'must_start_with_0'    -> does not start with '0' (Japanese phone number requirement)
              'invalid'              -> catch-all (shouldn't normally happen)
    """
    t = _normalize(text)
    candidates = _find_phone_candidates(t)

    if not candidates:
        return (False, [("", "no_phone_found")])

    mismatches: list[tuple[str, str]] = []

    for tok in candidates:
        ok, reason = _is_valid_token(tok, no_hyphen)
        if not ok:
            mismatches.append((tok, reason))

    return (len(mismatches) == 0, mismatches)
