from jfbench.constraints.notation import DecimalPlacesNotationConstraint


def test_decimal_places_accepts_matching_decimal_places() -> None:
    constraint = DecimalPlacesNotationConstraint(3)
    assert constraint.evaluate("values: 0.123, -3.400, 5.600e-2")[0]


def test_decimal_places_rejects_mismatched_decimal_places() -> None:
    constraint = DecimalPlacesNotationConstraint(3)
    assert not constraint.evaluate("values: 0.12, -3.400, 5.600e-2")[0]


def test_decimal_places_ignores_digits_before_decimal_point() -> None:
    constraint = DecimalPlacesNotationConstraint(2)
    assert constraint.evaluate("values: 12.30, .50, -0.10e1")[0]
