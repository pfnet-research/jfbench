from jfbench.constraints.processing import SuffixProcessingConstraint


def test_suffix_constraint_requires_expected_suffix() -> None:
    constraint = SuffixProcessingConstraint("EOF")
    assert constraint.evaluate("データの終わり: EOF")[0]
    assert not constraint.evaluate("EOFでは終わらない")[0]
