import pytest

from jfbench.constraints.ifbench_sentence import KeywordSentenceIfbenchConstraint


def test_keyword_sentence_accepts_case_insensitive_match() -> None:
    constraint = KeywordSentenceIfbenchConstraint(sentence_index=2, keyword="Apple")

    text = "First sentence without keyword. the apple pie is tasty."

    success, reason = constraint.evaluate(text)

    assert success is True
    assert reason is None


def test_keyword_sentence_rejects_when_missing_in_target() -> None:
    constraint = KeywordSentenceIfbenchConstraint(sentence_index=2, keyword="apple")

    text = "apple shows up early. no keyword here."

    success, reason = constraint.evaluate(text)

    assert success is False
    assert reason is not None


def test_keyword_sentence_requires_enough_sentences() -> None:
    constraint = KeywordSentenceIfbenchConstraint(sentence_index=3, keyword="apple")

    success, reason = constraint.evaluate("only one sentence.")

    assert success is False
    assert reason is not None


def test_keyword_sentence_instructions_reject_train_mode() -> None:
    constraint = KeywordSentenceIfbenchConstraint(sentence_index=1, keyword="apple")
    with pytest.raises(ValueError, match="ifbench"):
        constraint.instructions(train_or_test="train")
