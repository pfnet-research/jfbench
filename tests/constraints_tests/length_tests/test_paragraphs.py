from jfbench.constraints.length.paragraphs import ParagraphsLengthConstraint


def test_paragraphs_constraint_uses_double_newline() -> None:
    constraint = ParagraphsLengthConstraint(2, 3)
    text = "段落一\n\n段落二"
    assert constraint.evaluate(text)[0] is True
    assert constraint.evaluate("単独段落のみ")[0] is False
    assert constraint.evaluate("段落一\n\n段落二\n\n段落三\n\n段落四")[0] is False


def test_paragraphs_length_instructions_test_variant() -> None:
    constraint = ParagraphsLengthConstraint(1, 2, seed=0)
    assert (
        constraint.instructions(train_or_test="test")
        == "段落数は1～2にし、区切りは\\n\\nでお願いします。"
    )
