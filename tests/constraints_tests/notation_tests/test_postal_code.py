from jfbench.constraints.notation import NoHyphenJpPostalCodeNotationConstraint
from jfbench.constraints.notation import WithHyphenJpPostalCodeNotationConstraint


def test_postal_code_without_hyphen_requires_plain_digits() -> None:
    constraint = NoHyphenJpPostalCodeNotationConstraint()
    assert constraint.evaluate("Address: postal code 1000001 Tokyo.")[0]
    assert not constraint.evaluate("Address: postal code 100-0001 Tokyo.")[0]


def test_postal_code_with_hyphen_requires_hyphenated_form() -> None:
    constraint = WithHyphenJpPostalCodeNotationConstraint()
    assert constraint.evaluate("Address: postal code 100-0001 Tokyo.")[0]
    assert not constraint.evaluate("Address: postal code 1000001 Tokyo.")[0]
