from jfbench.constraints.length.sentences import SentencesLengthConstraint


def test_sentences_constraint_default_pysbd() -> None:
    constraint = SentencesLengthConstraint(2, 3)
    text = "文一。文二。文三。"
    assert constraint.evaluate(text)[0] is True
    assert constraint.evaluate("単文のみ。")[0] is False
    assert constraint.evaluate("文一。文二。文三。文四。")[0] is False


def test_sentences_constraint_counts_various_sentence_endings() -> None:
    constraint = SentencesLengthConstraint(3, 3)
    text = "これは一つの文です。これは別の文です！さらに確認しますか？"
    assert constraint.evaluate(text)[0] is True


def test_sentences_length_instructions_test_variant() -> None:
    constraint = SentencesLengthConstraint(2, 4, seed=0)
    assert (
        constraint.instructions(train_or_test="test")
        == "文は2～4文にし、pysbdで区切りやすい句読点と改行に整えてください。"
    )
