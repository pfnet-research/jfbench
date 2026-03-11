from jfbench.constraints.notation import GroupingNotationConstraint


def test_grouping_accepts_well_grouped_integers() -> None:
    constraint = GroupingNotationConstraint(max_group_size=3)
    text = "Population: 12,345,678 residents and -9 876 543 visitors."
    assert constraint.evaluate(text)[0]


def test_grouping_rejects_incorrect_grouping() -> None:
    constraint = GroupingNotationConstraint(max_group_size=3)
    assert not constraint.evaluate("Budget: 12,34,567 yen.")[0]


def test_grouping_ignores_decimals() -> None:
    constraint = GroupingNotationConstraint(max_group_size=3)
    text = "Price: 123.45 dollars, revenue 6,789."
    assert constraint.evaluate(text)[0]


def test_grouping_catches_long_integers_without_separators_anywhere() -> None:
    constraint = GroupingNotationConstraint(max_group_size=3)
    text = "TokenABC123456DEF should be grouped."
    assert not constraint.evaluate(text)[0]
