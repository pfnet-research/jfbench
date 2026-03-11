from jfbench.constraints.notation import KanjiNumeralsNotationConstraint
from jfbench.constraints.notation import NoKanjiNumeralsNotationConstraint


def test_kanji_numerals_constraint_accepts_only_kanji_numbers() -> None:
    constraint = KanjiNumeralsNotationConstraint()
    assert constraint.evaluate("Sales were 一千二百 units.")[0]


def test_no_kanji_numerals_constraint_rejects_kanji_numbers() -> None:
    constraint = NoKanjiNumeralsNotationConstraint()
    assert not constraint.evaluate("Revenue was 四千円.")[0]
    assert constraint.evaluate("Revenue was 4000 yen.")[0]
