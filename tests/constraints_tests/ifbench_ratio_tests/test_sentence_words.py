from jfbench.constraints.ifbench_ratio import SentenceWordsRatioIfbenchConstraint
from jfbench.constraints.ifbench_ratio import StopWordsRatioIfbenchConstraint


def test_sentence_words_ratio_passes_for_balanced_three_sentences() -> None:
    constraint = SentenceWordsRatioIfbenchConstraint()
    value = "あか いちご たべた。あお もも のんだ。きいろ すいか かんだ。"

    success, reason = constraint.evaluate(value)

    assert success is True
    assert reason is None


def test_sentence_words_ratio_rejects_shared_words() -> None:
    constraint = SentenceWordsRatioIfbenchConstraint()
    value = "あか いちご たべた。あか もも のんだ。あか すいか かんだ。"

    success, reason = constraint.evaluate(value)

    assert success is False
    assert reason is not None
    assert "Sentences must use distinct words" in reason


def test_sentence_words_ratio_requires_three_sentences() -> None:
    constraint = SentenceWordsRatioIfbenchConstraint()
    value = "あか いちご たべた。あお もも のんだ。"

    success, reason = constraint.evaluate(value)

    assert success is False
    assert reason is not None
    assert "Exactly three sentences are required" in reason


def test_sentence_words_ratio_requires_same_word_count() -> None:
    constraint = SentenceWordsRatioIfbenchConstraint()
    value = "red apple eats now. blue cherry runs fast. green melon rests."

    success, reason = constraint.evaluate(value)

    assert success is False
    assert reason is not None
    assert "must have the same word count" in reason


def test_stop_words_ratio_limits_usage() -> None:
    constraint = StopWordsRatioIfbenchConstraint(50)
    value = "そして 革新的 な 研究 を した"

    success, reason = constraint.evaluate(value)

    assert success is True
    assert reason is None


def test_stop_words_ratio_rejects_high_percentage() -> None:
    constraint = StopWordsRatioIfbenchConstraint(30)
    value = "の に は が を と も で"

    success, reason = constraint.evaluate(value)

    assert success is False
    assert reason is not None
    assert "exceeds maximum" in reason
