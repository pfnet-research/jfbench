from jfbench.constraints.format.yaml import YamlFormatConstraint


def test_yaml_evaluate_valid() -> None:
    yaml_text = "key: value\nitems:\n  - a\n  - b\n"
    constraint = YamlFormatConstraint()
    assert constraint.evaluate(yaml_text)[0] is True


def test_yaml_evaluate_invalid() -> None:
    yaml_text = "key: value: broken"
    constraint = YamlFormatConstraint()
    assert constraint.evaluate(yaml_text)[0] is False


def test_yaml_evaluate_strips_code_fence() -> None:
    yaml_text = "```yaml\nkey: value\nitems:\n  - a\n  - b\n```"
    constraint = YamlFormatConstraint()
    assert constraint.evaluate(yaml_text)[0] is True


def test_yaml_instructions() -> None:
    assert YamlFormatConstraint().instructions()
