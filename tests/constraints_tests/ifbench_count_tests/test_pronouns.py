from jfbench.constraints.ifbench_count import PronounsCountIfbenchConstraint


def test_pronouns_count_japanese_tokens() -> None:
    constraint = PronounsCountIfbenchConstraint(minimum_pronouns=3)
    value = "私は映画が好きです。あなたも私も彼も一緒に行きました。"

    success, reason = constraint.evaluate(value)

    assert success is True
    assert reason is None


def test_pronouns_count_insufficient() -> None:
    constraint = PronounsCountIfbenchConstraint(minimum_pronouns=1)
    value = "庭に猫がいる。"

    success, reason = constraint.evaluate(value)

    assert success is False
    assert reason is not None
    assert "Not enough pronouns" in reason
