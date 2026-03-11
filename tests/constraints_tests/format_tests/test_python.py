from jfbench.constraints.format.python import NoCodeFencePythonFormatConstraint
from jfbench.constraints.format.python import PythonFormatConstraint
from jfbench.constraints.format.python import WithCodeFencePythonFormatConstraint


def test_python_evaluate_valid() -> None:
    code = "def add(a, b):\n    return a + b\n"
    constraint = PythonFormatConstraint()
    assert constraint.evaluate(code)[0] is True


def test_python_evaluate_invalid() -> None:
    code = "def broken(:\n    pass\n"
    constraint = PythonFormatConstraint()
    assert constraint.evaluate(code)[0] is False


def test_python_instructions() -> None:
    assert PythonFormatConstraint().instructions()


def test_no_code_fence_python_rejects_fenced_block() -> None:
    constraint = NoCodeFencePythonFormatConstraint()
    ok, _ = constraint.evaluate("```python\nprint('hi')\n```")
    assert ok is False


def test_no_code_fence_python_rejects_tilde_fence() -> None:
    constraint = NoCodeFencePythonFormatConstraint()
    ok, _ = constraint.evaluate("~~~python\nprint('hi')\n~~~")
    assert ok is False


def test_no_code_fence_python_requires_valid_syntax() -> None:
    constraint = NoCodeFencePythonFormatConstraint()
    ok, reason = constraint.evaluate("def broken(:\n    pass")
    assert ok is False
    assert reason and "Syntax error" in reason


def test_with_code_fence_python_accepts_labeled_block() -> None:
    constraint = WithCodeFencePythonFormatConstraint()
    ok, _ = constraint.evaluate("```python\nprint('hi')\n```")
    assert ok is True


def test_with_code_fence_python_rejects_non_leading_fence() -> None:
    constraint = WithCodeFencePythonFormatConstraint()
    ok, _ = constraint.evaluate("prefix\n```python\nprint('hi')\n```")
    assert ok is False


def test_with_code_fence_python_requires_valid_syntax() -> None:
    constraint = WithCodeFencePythonFormatConstraint()
    ok, reason = constraint.evaluate("```python\ndef broken(:\n```")
    assert ok is False
    assert reason and "Syntax error" in reason


def test_with_code_fence_python_accepts_python3_hint() -> None:
    constraint = WithCodeFencePythonFormatConstraint()
    ok, _ = constraint.evaluate("```python3\nprint('ok')\n```")
    assert ok is True
