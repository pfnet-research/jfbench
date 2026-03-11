from jfbench.constraints.format.tsv import TsvFormatConstraint


def test_tsv_evaluate_valid() -> None:
    tsv_text = "name\tage\nAlice\t30\nBob\t25\n"
    constraint = TsvFormatConstraint()
    assert constraint.evaluate(tsv_text)[0] is True


def test_tsv_evaluate_inconsistent_columns() -> None:
    tsv_text = "name\tage\nAlice\nBob\t25\n"
    constraint = TsvFormatConstraint()
    assert constraint.evaluate(tsv_text)[0] is False


def test_tsv_instructions() -> None:
    assert TsvFormatConstraint().instructions()
