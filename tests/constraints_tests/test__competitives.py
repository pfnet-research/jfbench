import pytest

from jfbench.constraints._competitives import COMPETITIVE_CONSTRAINTS


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("BulletPointsFormatConstraint", "DiffFormatConstraint"),
        ("UppercaseCharacterConstraint", "HiraganaCharacterConstraint"),
        ("LowercaseCharacterConstraint", "KatakanaCharacterConstraint"),
        ("HalfWidthCharacterConstraint", "KanjiCharacterConstraint"),
        ("FullWidthCharacterConstraint", "CamelcaseNotationConstraint"),
        ("ImpersonalStyleConstraint", "FirstPersonSingularStyleConstraint"),
        ("EasyVocabularyStyleConstraint", "DifficultVocacularyStyleConstraint"),
        ("JoyfulEmotionalStyleConstraint", "SadEmotionalStyleConstraint"),
        ("CasualToneStyleConstraint", "BusinessToneStyleConstraint"),
        ("PoliteStyleConstraint", "PlainStyleConstraint"),
    ],
)
def test_competitive_pairs_are_registered(left: str, right: str) -> None:
    assert right in COMPETITIVE_CONSTRAINTS[left]
    assert left in COMPETITIVE_CONSTRAINTS[right]
