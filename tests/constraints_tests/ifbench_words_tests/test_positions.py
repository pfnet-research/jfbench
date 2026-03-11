from jfbench.constraints.ifbench_words import KeywordsSpecificPositionWordsIfbenchConstraint
from jfbench.constraints.ifbench_words import WordsPositionWordsIfbenchConstraint


def test_keywords_specific_position_matches_target() -> None:
    constraint = KeywordsSpecificPositionWordsIfbenchConstraint(
        sentence_index=2,
        word_index=2,
        keyword="keyword",
    )

    success, reason = constraint.evaluate("intro words only. well keyword placed.")

    assert success is True
    assert reason is None


def test_keywords_specific_position_rejects_wrong_word() -> None:
    constraint = KeywordsSpecificPositionWordsIfbenchConstraint(
        sentence_index=2,
        word_index=2,
        keyword="keyword",
    )

    success, reason = constraint.evaluate("intro words only. well filler placed.")

    assert success is False
    assert reason is not None


def test_words_position_accepts_matching_locations() -> None:
    constraint = WordsPositionWordsIfbenchConstraint(
        word_index=1,
        from_end_index=1,
        word="Hello",
    )

    success, reason = constraint.evaluate("Hello middle hello")

    assert success is True
    assert reason is None


def test_words_position_rejects_when_positions_differ() -> None:
    constraint = WordsPositionWordsIfbenchConstraint(
        word_index=1,
        from_end_index=1,
        word="Hello",
    )

    success, reason = constraint.evaluate("Hello middle world")

    assert success is False
    assert reason is not None
