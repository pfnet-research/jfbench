from jfbench.constraints.ifbench_words import ConsonantsWordsIfbenchConstraint
from jfbench.constraints.ifbench_words import OddEvenSyllablesWordsIfbenchConstraint
from jfbench.constraints.ifbench_words import PalindromeWordsIfbenchConstraint
from jfbench.constraints.ifbench_words import VowelWordsIfbenchConstraint


def test_consonants_words_accepts_emphasis_kana() -> None:
    constraint = ConsonantsWordsIfbenchConstraint()

    success, reason = constraint.evaluate("ざっし かっぱ けっこん")

    assert success is True
    assert reason is None


def test_consonants_words_uses_reading_for_hidden_emphasis() -> None:
    constraint = ConsonantsWordsIfbenchConstraint()

    success, reason = constraint.evaluate("学校 ざっし")

    assert success is True
    assert reason is None


def test_consonants_words_rejects_plain_terms() -> None:
    constraint = ConsonantsWordsIfbenchConstraint()

    success, reason = constraint.evaluate("かな ことば ねこ")

    assert success is False
    assert reason is not None


def test_odd_even_syllables_alternates_parity() -> None:
    constraint = OddEvenSyllablesWordsIfbenchConstraint()

    success, reason = constraint.evaluate("アイウ カキ サシス")

    assert success is True
    assert reason is None


def test_odd_even_syllables_detects_same_parity_neighbors() -> None:
    constraint = OddEvenSyllablesWordsIfbenchConstraint()

    success, reason = constraint.evaluate("アイウ カキ クケ")

    assert success is False
    assert reason is not None


def test_palindrome_words_counts_reading_palindromes() -> None:
    constraint = PalindromeWordsIfbenchConstraint(minimum_palindromes=2)

    success, reason = constraint.evaluate("level civic apple")

    assert success is True
    assert reason is None


def test_palindrome_words_rejects_when_insufficient() -> None:
    constraint = PalindromeWordsIfbenchConstraint(minimum_palindromes=2)

    success, reason = constraint.evaluate("apple banana carrot")

    assert success is False
    assert reason is not None


def test_vowel_words_limits_unique_vowels() -> None:
    constraint = VowelWordsIfbenchConstraint()

    success, reason = constraint.evaluate("かきく さしす たちつ")

    assert success is True
    assert reason is None


def test_vowel_words_rejects_more_than_three_vowel_types() -> None:
    constraint = VowelWordsIfbenchConstraint()

    success, reason = constraint.evaluate("aei ou uo")

    assert success is False
    assert reason is not None
