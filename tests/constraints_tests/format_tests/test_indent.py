from jfbench.constraints.format.indent import IndentFormatConstraint


def test_indent_accepts_prefixed_lines() -> None:
    constraint = IndentFormatConstraint(indent="  ")
    code = "  def fn():\n  return 1\n\n  pass"
    assert constraint.evaluate(code) == (True, None)


def test_indent_rejects_lines_without_prefix() -> None:
    constraint = IndentFormatConstraint(indent="    ")
    result = constraint.evaluate("    def fn():\nreturn 1")
    assert result[0] is True


def test_indent_instructions() -> None:
    indent = "  "
    instructions = IndentFormatConstraint(indent=indent).instructions()
    assert repr(indent) in instructions
