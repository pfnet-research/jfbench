from collections import Counter

import pytest

from jfbench.constraints.format import javascript
from jfbench.constraints.format.javascript import JavascriptFormatConstraint
from jfbench.constraints.format.javascript import NoCodeFenceJavascriptFormatConstraint
from jfbench.constraints.format.javascript import WithCodeFenceJavascriptFormatConstraint


def test_javascript_evaluate_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: Counter[str] = Counter()

    def fake_parse_module(src: str, tolerant: bool = False) -> None:
        calls["parse_module"] += 1
        assert "console.log" in src

    def fail_parse_script(*_args: object, **_kwargs: object) -> None:
        pytest.fail("parseScript should not be called when module parsing succeeds")

    monkeypatch.setattr(javascript.esprima, "parseModule", fake_parse_module, raising=True)
    monkeypatch.setattr(javascript.esprima, "parseScript", fail_parse_script, raising=True)

    assert JavascriptFormatConstraint().evaluate("console.log(1);")[0] is True
    assert calls["parse_module"] == 1


def test_javascript_evaluate_invalid_reports_first_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyError(Exception):
        def __init__(self, description: str, line: int, column: int) -> None:
            super().__init__(description)
            self.description = description
            self.lineNumber = line
            self.column = column

    def raise_module_error(*_args: object, **_kwargs: object) -> None:
        raise DummyError("Unexpected token", 1, 5)

    def raise_script_error(*_args: object, **_kwargs: object) -> None:
        raise DummyError("Still invalid", 1, 5)

    monkeypatch.setattr(javascript.esprima, "Error", DummyError, raising=False)
    monkeypatch.setattr(javascript.esprima, "parseModule", raise_module_error, raising=True)
    monkeypatch.setattr(javascript.esprima, "parseScript", raise_script_error, raising=True)

    constraint = JavascriptFormatConstraint()
    ok = constraint.evaluate("!invalid js")[0]
    assert ok is False


def test_javascript_instructions() -> None:
    assert JavascriptFormatConstraint().instructions()


def test_no_code_fence_javascript_rejects_fenced_block() -> None:
    constraint = NoCodeFenceJavascriptFormatConstraint()
    ok, _ = constraint.evaluate("```js\nconst n = 1;\n```")
    assert ok is False


def test_with_code_fence_javascript_accepts_labeled_block() -> None:
    constraint = WithCodeFenceJavascriptFormatConstraint()
    ok, _ = constraint.evaluate("```javascript\nconst n = 1;\n```")
    assert ok is True


def test_with_code_fence_requires_leading_block() -> None:
    constraint = WithCodeFenceJavascriptFormatConstraint()
    ok, reason = constraint.evaluate("preamble\n```javascript\nconst n = 1;\n```")
    assert ok is False
    assert "Code fences are required" in (reason or "")


def test_with_code_fence_validates_inner_js(monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = 0
    last_src: str | None = None

    def fake_is_valid_js(src: str) -> tuple[bool, str | None]:
        nonlocal call_count, last_src
        call_count += 1
        last_src = src
        return False, "Bad JS"

    monkeypatch.setattr(javascript, "is_valid_js", fake_is_valid_js, raising=True)
    constraint = WithCodeFenceJavascriptFormatConstraint()
    ok, reason = constraint.evaluate("```js\nconst x = ;\n```")

    assert ok is False
    assert call_count == 1
    assert last_src == "const x = ;"
    assert "Syntax error" in (reason or "")


def test_no_code_fence_validates_js(monkeypatch: pytest.MonkeyPatch) -> None:
    call_count = 0
    last_src: str | None = None

    def fake_is_valid_js(src: str) -> tuple[bool, str | None]:
        nonlocal call_count, last_src
        call_count += 1
        last_src = src
        return True, None

    monkeypatch.setattr(javascript, "is_valid_js", fake_is_valid_js, raising=True)
    constraint = NoCodeFenceJavascriptFormatConstraint()
    ok, _ = constraint.evaluate("const value = 1;")

    assert ok is True
    assert call_count == 1
    assert last_src == "const value = 1;"
