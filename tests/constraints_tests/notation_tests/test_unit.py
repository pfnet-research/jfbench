from jfbench.constraints.notation import UnitNotationConstraint


def test_unit_notation_constraint_allows_space() -> None:
    constraint = UnitNotationConstraint()
    assert constraint.evaluate("距離は 12 km です")[0] is True


def test_unit_notation_constraint_allows_no_space() -> None:
    constraint = UnitNotationConstraint()
    assert constraint.evaluate("距離は12kmです")[0] is True


def test_unit_notation_constraint_rejects_missing_unit() -> None:
    constraint = UnitNotationConstraint()
    assert constraint.evaluate("距離は 12 です")[0] is False
