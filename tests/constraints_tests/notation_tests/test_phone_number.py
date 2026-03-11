from jfbench.constraints.notation import NoHyphenPhoneNumberNotationConstraint
from jfbench.constraints.notation import WithHyphenPhoneNumberNotationConstraint


def test_phone_number_without_hyphen_requires_digits_only() -> None:
    constraint = NoHyphenPhoneNumberNotationConstraint()
    assert constraint.evaluate("Contact: 0312345678 for details.")[0]
    assert not constraint.evaluate("Contact: 03-1234-5678 for details.")[0]
    assert not constraint.evaluate("Contact: 1312345678 for details.")[0]


def test_phone_number_with_hyphen_requires_hyphenated_form() -> None:
    constraint = WithHyphenPhoneNumberNotationConstraint()
    assert constraint.evaluate("Contact: 03-1234-5678 for details.")[0]
    assert not constraint.evaluate("Contact: 0312345678 for details.")[0]
    assert not constraint.evaluate("Contact: 13-1234-5678 for details.")[0]
