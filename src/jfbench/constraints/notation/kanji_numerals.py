import logging
import re
import unicodedata

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class KanjiNumeralsNotationConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        all_kansuji, kansuji_tokens = _all_numbers_are_kansuji(value)
        if not all_kansuji:
            reason = f"[Kanji Numerals Notation] Found Arabic numeral tokens: {kansuji_tokens}"
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "出力には少なくとも1つ数値を含め、その数値はすべて漢数字（例: 一二三）で表記してください。",
            "算用数字は使わず、数字を漢数字へ置き換えたものを1件以上示してください。",
            "文章内の数字は漢数字のみを用い、0-9の表記を避けた数値を少なくとも1つ入れてください。",
            "全ての数値を漢字表記に統一し、アラビア数字を残さない例を必ず含めてください。",
            "漢数字以外の表記は禁止です。漢数字で記した数値を最低1件は出力に含めてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "漢数字で表した数値を含めてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "出力を修正し、少なくとも1つの数値を漢数字に置き換えてください。"


class NoKanjiNumeralsNotationConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        no_kansuji, arabic_tokens = _all_numbers_are_not_kansuji(value)
        if not no_kansuji:
            reason = f"[No Kanji Numerals Notation] Found Kanji numeral tokens: {arabic_tokens}"
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "出力に少なくとも一つ数値を含め、その数値はすべて算用数字（0-9）で表記し、漢数字は使わないでください。",
            "数字は必ずアラビア数字に統一し、「一」「二」などの漢数字を避けた数値を1件以上示してください。",
            "任意の数値は算用数字で示し、漢数字を含めない文章にしてください。少なくとも1つは数値を入れてください。",
            "0～9の数字表記のみを使用し、漢数字を見つけたら算用数字へ直した例を出力に含めてください。",
            "数の表現には算用数字だけを採用し、漢字による表記を排した数値を最低1件は書いてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "算用数字の数値を含め、漢数字は避けてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "出力を修正し、少なくとも一つの数値を算用数字に統一してください。"


_ARABIC_NUMBER_RE = re.compile(
    r"[+\-−]?"
    r"(?:[0-9\uFF10-\uFF19]+(?:[._,，．・][0-9\uFF10-\uFF19]+)*)"
)


def _find_arabic_number_tokens(text: str) -> list[str]:
    return [m.group(0) for m in _ARABIC_NUMBER_RE.finditer(text)]


def _all_numbers_are_kansuji(text: str) -> tuple[bool, list[str]]:
    arabic_tokens = _find_arabic_number_tokens(text)
    return (len(arabic_tokens) == 0, arabic_tokens)


_KANSUJI_CLASS = "〇零一二三四五六七八九十百千"

# Regex to find contiguous runs of Kanji numerals
_KANSUJI_RE = re.compile(f"[{_KANSUJI_CLASS}]+")


def _find_kansuji_tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text)
    return [m.group(0) for m in _KANSUJI_RE.finditer(text)]


def _all_numbers_are_not_kansuji(text: str) -> tuple[bool, list[str]]:
    tokens = _find_kansuji_tokens(text)
    return (len(tokens) == 0, tokens)
