from jfbench.constraints.processing import PrefixProcessingConstraint


def test_prefix_constraint_requires_expected_prefix() -> None:
    constraint = PrefixProcessingConstraint("Hello:")
    assert constraint.evaluate("Hello: 世界")[0]
    assert not constraint.evaluate("こんにちは Hello: 世界")[0]
