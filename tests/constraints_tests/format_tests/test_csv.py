from jfbench.constraints.format.csv import CsvFormatConstraint


def test_csv_evaluate_valid() -> None:
    csv_text = "name,age\nAlice,30\nBob,25\n"
    constraint = CsvFormatConstraint()
    assert constraint.evaluate(csv_text)[0] is True


def test_csv_evaluate_with_inconsistent_columns() -> None:
    csv_text = "name,age\nAlice\nBob,25\n"
    constraint = CsvFormatConstraint()
    assert constraint.evaluate(csv_text)[0] is False


def test_csv_instructions() -> None:
    assert CsvFormatConstraint().instructions()
