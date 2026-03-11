from jfbench.constraints.notation import TitlecaseNotationConstraint


def test_titlecase_notation_constraint_allows_single_word() -> None:
    constraint = TitlecaseNotationConstraint()
    assert constraint.evaluate("Title")[0] is True


def test_titlecase_notation_constraint_allows_multiple_words() -> None:
    constraint = TitlecaseNotationConstraint()
    assert constraint.evaluate("Hello World")[0] is True


def test_titlecase_notation_constraint_rejects_lowercase() -> None:
    constraint = TitlecaseNotationConstraint()
    assert constraint.evaluate("hello world")[0] is False
