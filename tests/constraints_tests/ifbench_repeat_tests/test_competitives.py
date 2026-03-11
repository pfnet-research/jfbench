from jfbench.constraints._competitives import COMPETITIVE_CONSTRAINTS


def test_ifbench_repeat_constraints_are_competitive_with_all_others() -> None:
    all_constraints = set(COMPETITIVE_CONSTRAINTS)
    repeat_constraints = {
        "ChangeRepeatIfbenchConstraint",
        "SimpleRepeatIfbenchConstraint",
    }
    for name in repeat_constraints:
        expected = all_constraints - {name}
        assert expected == set(COMPETITIVE_CONSTRAINTS[name])
