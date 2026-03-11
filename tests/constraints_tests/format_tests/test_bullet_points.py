from jfbench.constraints.format import BulletPointsFormatConstraint


def test_bullet_constraint_accepts_mixed_content() -> None:
    value = "Summary:\n- point one\n- point two\nConclusion."
    assert BulletPointsFormatConstraint().evaluate(value)[0] is True


def test_bullet_constraint_requires_bullet_presence() -> None:
    value = "This paragraph has no bullet points."
    assert BulletPointsFormatConstraint().evaluate(value)[0] is False


def test_bullet_constraint_rejects_wrong_marker() -> None:
    value = "Intro\n* wrong starter bullet"
    assert BulletPointsFormatConstraint().evaluate(value)[0] is False


def test_bullet_constraint_accepts_compact_marker() -> None:
    value = "-missing space after dash"
    assert BulletPointsFormatConstraint().evaluate(value)[0] is True


def test_bullet_constraint_ignores_other_bullets_when_expected_present() -> None:
    value = "- first\n* other style bullet"
    assert BulletPointsFormatConstraint().evaluate(value)[0] is True


def test_bullet_constraint_rejects_empty_expected_bullet() -> None:
    success, reason = BulletPointsFormatConstraint().evaluate("-   ")
    assert success is False
    assert reason is not None and "has no content" in reason


def test_bullet_constraint_rejects_double_marker_without_content() -> None:
    success, _ = BulletPointsFormatConstraint().evaluate("--")
    assert success is False
