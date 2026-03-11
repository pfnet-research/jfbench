from jfbench.constraints.length.sections import SectionsLengthConstraint


def test_sections_constraint_custom_delimiter() -> None:
    constraint = SectionsLengthConstraint(2, 4, sections_delimiter="---")
    text = "sec1---sec2---sec3"
    assert constraint.evaluate(text)[0] is True
    assert constraint.evaluate("single section")[0] is False
    assert constraint.evaluate("a---b---c---d---e")[0] is False


def test_sections_length_instructions_test_variant() -> None:
    constraint = SectionsLengthConstraint(1, 3, sections_delimiter="--", seed=0)
    assert (
        constraint.instructions(train_or_test="test")
        == "セクションは1～3個にして、区切りは'--'で統一してください。"
    )
