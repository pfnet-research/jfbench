from typing import Protocol
from typing import Type

import pytest

from jfbench.constraints.character import FullWidthCharacterConstraint
from jfbench.constraints.character import HalfWidthCharacterConstraint
from jfbench.constraints.character import HiraganaCharacterConstraint
from jfbench.constraints.character import KanjiCharacterConstraint
from jfbench.constraints.character import KatakanaCharacterConstraint
from jfbench.constraints.character import LowercaseCharacterConstraint
from jfbench.constraints.character import UppercaseCharacterConstraint


class _ConstraintWithEvaluate(Protocol):
    def evaluate(self, value: str) -> tuple[bool, str | None]: ...


def test_uppercase_character_constraint_requires_uppercase_letters() -> None:
    constraint = UppercaseCharacterConstraint()
    assert constraint.evaluate("ABC")[0] is True
    assert constraint.evaluate("AbC")[0] is False


def test_lowercase_character_constraint_requires_lowercase_letters() -> None:
    constraint = LowercaseCharacterConstraint()
    assert constraint.evaluate("abc")[0] is True
    assert constraint.evaluate("aBc")[0] is False


def test_hiragana_character_constraint_allows_whitespace() -> None:
    constraint = HiraganaCharacterConstraint()
    assert constraint.evaluate("あい う\nえ")[0] is True


def test_hiragana_character_constraint_allows_symbols() -> None:
    constraint = HiraganaCharacterConstraint()
    assert constraint.evaluate("あ★")[0] is True


def test_hiragana_character_constraint_allows_alphabet() -> None:
    constraint = HiraganaCharacterConstraint()
    assert constraint.evaluate("あA")[0] is True


def test_hiragana_character_constraint_allows_digits() -> None:
    constraint = HiraganaCharacterConstraint()
    assert constraint.evaluate("あ123")[0] is True


def test_hiragana_character_constraint_allows_long_vowel_mark() -> None:
    constraint = HiraganaCharacterConstraint()
    assert constraint.evaluate("あー")[0] is True


def test_katakana_character_constraint_allows_symbols() -> None:
    constraint = KatakanaCharacterConstraint()
    assert constraint.evaluate("ア★")[0] is True


def test_katakana_character_constraint_allows_alphabet() -> None:
    constraint = KatakanaCharacterConstraint()
    assert constraint.evaluate("アA")[0] is True


def test_katakana_character_constraint_allows_digits() -> None:
    constraint = KatakanaCharacterConstraint()
    assert constraint.evaluate("ア123")[0] is True


def test_katakana_character_constraint_allows_long_vowel_mark() -> None:
    constraint = KatakanaCharacterConstraint()
    assert constraint.evaluate("アー")[0] is True


def test_kanji_character_constraint_allows_symbols() -> None:
    constraint = KanjiCharacterConstraint()
    assert constraint.evaluate("漢★")[0] is True


def test_kanji_character_constraint_allows_alphabet() -> None:
    constraint = KanjiCharacterConstraint()
    assert constraint.evaluate("漢A")[0] is True


@pytest.mark.parametrize(
    ("constraint_cls", "prefix"),
    [
        (HiraganaCharacterConstraint, "あ"),
        (KatakanaCharacterConstraint, "ア"),
        (KanjiCharacterConstraint, "漢"),
    ],
)
def test_script_constraints_allow_ascii_punctuation(
    constraint_cls: Type[_ConstraintWithEvaluate], prefix: str
) -> None:
    constraint = constraint_cls()
    punctuation = "(){}-|!@#$%^&*-=_+'\"?/.,<>[]"

    assert constraint.evaluate(f"{prefix}{punctuation}")[0] is True


def test_full_width_character_constraint_allows_whitespace() -> None:
    constraint = FullWidthCharacterConstraint()
    assert constraint.evaluate("ＡＢＣ 　\n")[0] is True


def test_half_width_character_constraint_allows_known_halfwidth_chars() -> None:
    constraint = HalfWidthCharacterConstraint()

    assert constraint.evaluate("ABC\u201cdef\u201d")[0] is True
    assert constraint.evaluate("\uffa1\uffee")[0] is True


@pytest.mark.parametrize(
    ("constraint_cls", "value", "offending"),
    [
        (HiraganaCharacterConstraint, "あア", "ア"),
        (KatakanaCharacterConstraint, "アあ", "あ"),
        (KanjiCharacterConstraint, "漢ア", "ア"),
        (FullWidthCharacterConstraint, "ＡA", "A"),
        (HalfWidthCharacterConstraint, "AＡ", "Ａ"),
    ],
)
def test_script_constraints_include_offending_character_in_reason(
    constraint_cls: Type[_ConstraintWithEvaluate], value: str, offending: str
) -> None:
    constraint = constraint_cls()

    result, reason = constraint.evaluate(value)

    assert result is False
    assert reason is not None
    assert offending in reason
