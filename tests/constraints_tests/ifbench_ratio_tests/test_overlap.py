import pytest

from jfbench.constraints.ifbench_ratio import OverlapRatioIfbenchConstraint


def test_overlap_ratio_matches_target_within_tolerance() -> None:
    document = "alpha beta gamma delta"
    value = "alpha beta gamma zeta"
    constraint = OverlapRatioIfbenchConstraint(document, target_ratio_percent=50)

    success, reason = constraint.evaluate(value)

    assert success is True
    assert reason is None


def test_overlap_ratio_outside_tolerance_is_rejected() -> None:
    document = "alpha beta gamma delta"
    value = "alpha beta gamma delta epsilon zeta"
    constraint = OverlapRatioIfbenchConstraint(document, target_ratio_percent=80)

    success, reason = constraint.evaluate(value)

    assert success is False
    assert reason is not None
    assert "Trigram overlap" in reason


def test_overlap_ratio_instructions_reject_train_mode() -> None:
    constraint = OverlapRatioIfbenchConstraint("alpha beta gamma", target_ratio_percent=50)
    with pytest.raises(ValueError, match="ifbench"):
        constraint.instructions(train_or_test="train")
