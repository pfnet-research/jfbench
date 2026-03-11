from jfbench.constraints.ifbench_words import LastFirstWordsIfbenchConstraint
from jfbench.constraints.ifbench_words import NoConsecutiveWordsIfbenchConstraint
from jfbench.constraints.ifbench_words import ParagraphLastFirstWordsIfbenchConstraint
from jfbench.constraints.ifbench_words import RepeatsWordsIfbenchConstraint
from jfbench.constraints.ifbench_words import StartVerbWordsIfbenchConstraint


def test_no_consecutive_words_accepts_different_initials() -> None:
    constraint = NoConsecutiveWordsIfbenchConstraint()

    success, reason = constraint.evaluate("apple banana cat")

    assert success is True
    assert reason is None


def test_no_consecutive_words_rejects_same_initials_even_if_cased() -> None:
    constraint = NoConsecutiveWordsIfbenchConstraint()

    success, reason = constraint.evaluate("apple Apple")

    assert success is False
    assert reason is not None


def test_last_first_words_chains_sentences() -> None:
    constraint = LastFirstWordsIfbenchConstraint()

    success, reason = constraint.evaluate("走れ 素早く。素早く 進め。")

    assert success is True
    assert reason is None


def test_last_first_words_rejects_when_chain_breaks() -> None:
    constraint = LastFirstWordsIfbenchConstraint()

    success, reason = constraint.evaluate("走れ 素早く。速さで進め。")

    assert success is False
    assert reason is not None


def test_paragraph_last_first_accepts_matching_edges() -> None:
    constraint = ParagraphLastFirstWordsIfbenchConstraint()

    success, reason = constraint.evaluate("same middle same\n\nagain middle again")

    assert success is True
    assert reason is None


def test_paragraph_last_first_rejects_mismatch() -> None:
    constraint = ParagraphLastFirstWordsIfbenchConstraint()

    success, reason = constraint.evaluate("first middle last\n\nagain middle again")

    assert success is False
    assert reason is not None


def test_repeats_words_rejects_exceeding_limit() -> None:
    constraint = RepeatsWordsIfbenchConstraint(max_repeats=2)

    success, reason = constraint.evaluate("apple apple banana")

    assert success is False
    assert reason is not None


def test_repeats_words_accepts_under_limit() -> None:
    constraint = RepeatsWordsIfbenchConstraint(max_repeats=2)

    success, reason = constraint.evaluate("apple banana cherry")

    assert success is True
    assert reason is None


def test_start_verb_accepts_japanese_verb() -> None:
    constraint = StartVerbWordsIfbenchConstraint()

    success, reason = constraint.evaluate("書く 計画をまとめる")

    assert success is True
    assert reason is None


def test_start_verb_rejects_noun_start() -> None:
    constraint = StartVerbWordsIfbenchConstraint()

    success, reason = constraint.evaluate("本 読む")

    assert success is False
    assert reason is not None
