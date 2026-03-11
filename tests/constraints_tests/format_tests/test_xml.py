from jfbench.constraints.format.xml import XmlFormatConstraint


def test_xml_evaluate_valid() -> None:
    xml = "<root><item id='1'/></root>"
    constraint = XmlFormatConstraint()
    assert constraint.evaluate(xml)[0] is True


def test_xml_evaluate_invalid() -> None:
    xml = "<root><unclosed></root>"
    constraint = XmlFormatConstraint()
    assert constraint.evaluate(xml)[0] is False


def test_xml_evaluate_strips_code_fence() -> None:
    xml = "```xml\n<root><item id='1'/></root>\n```"
    constraint = XmlFormatConstraint()
    assert constraint.evaluate(xml)[0] is True


def test_xml_instructions() -> None:
    assert XmlFormatConstraint().instructions()
