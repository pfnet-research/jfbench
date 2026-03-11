from dataclasses import dataclass
from typing import Any

import pytest

from jfbench.constraints.format import latex as latex_module
from jfbench.constraints.format.latex import LatexFormatConstraint


def test_latex_evaluate_delegates_to_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, str] = {}

    def fake_check(_self: LatexFormatConstraint, value: str) -> tuple[bool, list[str]]:
        recorded["value"] = value
        return True, []

    monkeypatch.setattr(
        LatexFormatConstraint, "_pylatexenc_syntax_check", fake_check, raising=True
    )
    constraint = LatexFormatConstraint()
    assert constraint.evaluate(r"\LaTeX{}")[0] is True
    assert recorded["value"] == r"\LaTeX{}"


def test_latex_evaluate_strips_code_fence(monkeypatch: pytest.MonkeyPatch) -> None:
    recorded: dict[str, str] = {}

    def fake_check(_self: LatexFormatConstraint, value: str) -> tuple[bool, list[str]]:
        recorded["value"] = value
        return True, []

    monkeypatch.setattr(
        LatexFormatConstraint, "_pylatexenc_syntax_check", fake_check, raising=True
    )
    constraint = LatexFormatConstraint()
    assert constraint.evaluate("```latex\n\\LaTeX{}\n```")[0] is True
    assert recorded["value"] == "\\LaTeX{}"


def test_latex_evaluate_logs_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_check(_self: LatexFormatConstraint, _value: str) -> tuple[bool, list[str]]:
        return False, ["err one", "err two"]

    monkeypatch.setattr(
        LatexFormatConstraint, "_pylatexenc_syntax_check", fake_check, raising=True
    )
    assert LatexFormatConstraint().evaluate("bad")[0] is False


def test_pylatexenc_syntax_check_accepts_undefined_macros(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass
    class DummyMacroNode:
        macroname: str
        nodeargd: Any = None
        nodelist: list[Any] | None = None

    class DummyWalker:
        def __init__(self, value: str, latex_context: Any):
            self.value = value
            self.context = latex_context

        def get_latex_nodes(self) -> tuple[list[Any], None, None]:
            return [DummyMacroNode("undefined")], None, None

    monkeypatch.setattr(latex_module, "_build_context", lambda: object(), raising=False)
    monkeypatch.setattr(latex_module, "LatexWalker", DummyWalker, raising=False)
    monkeypatch.setattr(latex_module, "LatexMacroNode", DummyMacroNode, raising=False)

    ok, errors = LatexFormatConstraint._pylatexenc_syntax_check(  # noqa: SLF001
        r"\undefined"
    )
    assert ok is True
    assert errors == []


def test_pylatexenc_syntax_check_allows_arbitrary_macros(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass
    class DummyMacroNode:
        macroname: str
        nodeargd: Any = None
        nodelist: list[Any] | None = None

    @dataclass
    class DummyEnvironmentNode:
        environmentname: str
        nodelist: list[Any] | None = None
        nodeargd: Any = None

    class DummyWalker:
        def __init__(self, value: str, latex_context: Any):
            self.value = value
            self.context = latex_context

        def get_latex_nodes(self) -> tuple[list[Any], None, None]:
            nodes = [
                DummyMacroNode("defined"),
                DummyMacroNode("undefined"),
                DummyMacroNode("item"),
                DummyEnvironmentNode("itemize", nodelist=[DummyMacroNode("item")]),
            ]
            return nodes, None, None

    class DummyContext:
        def get_macro(self, name: str) -> object | None:
            return object() if name in {"defined", "item"} else None

    monkeypatch.setattr(latex_module, "_build_context", lambda: DummyContext(), raising=False)
    monkeypatch.setattr(latex_module, "LatexWalker", DummyWalker, raising=False)
    monkeypatch.setattr(latex_module, "LatexMacroNode", DummyMacroNode, raising=False)
    monkeypatch.setattr(latex_module, "LatexEnvironmentNode", DummyEnvironmentNode, raising=False)

    ok, errors = LatexFormatConstraint._pylatexenc_syntax_check(  # noqa: SLF001
        r"\defined \undefined \item"
    )
    assert ok is False
    assert r"\item used outside of a list environment" in errors


def test_pylatexenc_syntax_check_on_parse_error(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyParseError(Exception):
        pass

    class DummyWalker:
        def __init__(self, _value: str, latex_context: Any | None = None):
            pass

        def get_latex_nodes(self) -> tuple[list[Any], Any, Any]:
            raise DummyParseError("boom")

    monkeypatch.setattr(latex_module, "_build_context", lambda: object(), raising=False)
    monkeypatch.setattr(latex_module, "LatexWalkerParseError", DummyParseError, raising=False)
    monkeypatch.setattr(latex_module, "LatexWalker", DummyWalker, raising=False)

    ok, errors = LatexFormatConstraint._pylatexenc_syntax_check("bad")  # noqa: SLF001
    assert ok is False
    assert errors == ["Parse error: boom"]


def test_latex_instructions() -> None:
    assert LatexFormatConstraint().instructions()
