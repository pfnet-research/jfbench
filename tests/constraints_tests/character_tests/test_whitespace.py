from jfbench.constraints.character import NoSuffixWhitespaceConstraint
from jfbench.constraints.character import NoWhitespaceConstraint


def test_no_whitespace_constraint_rejects_any_space() -> None:
    constraint = NoWhitespaceConstraint()
    assert constraint.evaluate("NoSpaceHere")[0] is True
    assert constraint.evaluate("Has space")[0] is False
    assert constraint.evaluate("Has\ttab")[0] is False


def test_no_suffix_whitespace_constraint_rejects_trailing_space() -> None:
    constraint = NoSuffixWhitespaceConstraint()
    assert constraint.evaluate("No trailing space")[0] is True
    assert constraint.evaluate("Ends with space ")[0] is False
    assert constraint.evaluate("Ends with newline\n")[0] is False
