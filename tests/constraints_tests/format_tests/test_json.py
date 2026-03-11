from jfbench.constraints.format.json import JsonFormatConstraint


def test_json_evaluate_valid() -> None:
    constraint = JsonFormatConstraint()
    assert constraint.evaluate('{"key": "value"}')[0] is True


def test_json_evaluate_invalid() -> None:
    constraint = JsonFormatConstraint()
    assert constraint.evaluate("{invalid json}")[0] is False


def test_json_instructions() -> None:
    assert JsonFormatConstraint().instructions()
