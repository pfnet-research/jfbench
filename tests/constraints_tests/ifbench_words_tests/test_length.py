import pytest

from jfbench.constraints.ifbench_words import PrimeLengthsWordsIfbenchConstraint


def test_prime_lengths_accepts_prime_surfaces() -> None:
    constraint = PrimeLengthsWordsIfbenchConstraint()

    success, reason = constraint.evaluate("hi wow")

    assert success is True
    assert reason is None


def test_prime_lengths_rejects_non_prime_length() -> None:
    constraint = PrimeLengthsWordsIfbenchConstraint()

    success, reason = constraint.evaluate("four ok")

    assert success is False
    assert reason is not None


def test_prime_lengths_instructions_reject_train_mode() -> None:
    constraint = PrimeLengthsWordsIfbenchConstraint()
    with pytest.raises(ValueError, match="ifbench"):
        constraint.instructions(train_or_test="train")
