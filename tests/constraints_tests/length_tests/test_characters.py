from jfbench.constraints.length.characters import CharactersLengthConstraint


def test_characters_constraint_pass_and_fail() -> None:
    constraint = CharactersLengthConstraint(5, 10)
    assert constraint.evaluate("123456")[0] is True
    assert constraint.evaluate("1234")[0] is False
    assert constraint.evaluate("12345678901")[0] is False


def test_characters_length_instructions_test_variant() -> None:
    constraint = CharactersLengthConstraint(2, 4)
    assert (
        constraint.instructions(train_or_test="test") == "文字数は2字以上4字以下にしてください。"
    )
