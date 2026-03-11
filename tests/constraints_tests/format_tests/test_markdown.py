from textwrap import dedent
from types import SimpleNamespace
from typing import cast
from typing import TYPE_CHECKING

import pytest

from jfbench.constraints.format import markdown as markdown_module
from jfbench.constraints.format.markdown import MarkdownClosedFencesConstraint
from jfbench.constraints.format.markdown import MarkdownFormatConstraint
from jfbench.constraints.format.markdown import MarkdownHeadingJumpsConstraint
from jfbench.constraints.format.markdown import MarkdownHeadingsStructureConstraint
from jfbench.constraints.format.markdown import MarkdownLinksAndImagesConstraint
from jfbench.constraints.format.markdown import MarkdownListStructureConstraint
from jfbench.constraints.format.markdown import MarkdownParseableConstraint
from jfbench.constraints.format.markdown import MarkdownReferenceLinksConstraint
from jfbench.constraints.format.markdown import MarkdownTableStructureConstraint
from jfbench.constraints.format.markdown import MarkdownUnconsumedEmphasisMarkersConstraint


if TYPE_CHECKING:
    from markdown_it.token import Token


def test_markdown_format_constraint_aggregates(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyConstraint:
        def __init__(self, result: bool):
            self.result = result
            self.calls = 0

        def evaluate(self, value: str) -> tuple[bool, str | None]:
            self.calls += 1
            reason = None if self.result else "failure"
            return self.result, reason

        def instructions(self, train_or_test: str = "train") -> str:
            return "dummy"

        @property
        def group(self) -> str:
            return "Test"

    true_constraint = DummyConstraint(True)
    false_constraint = DummyConstraint(False)

    monkeypatch.setattr(markdown_module, "MarkdownParseableConstraint", lambda: true_constraint)
    monkeypatch.setattr(markdown_module, "MarkdownClosedFencesConstraint", lambda: true_constraint)
    monkeypatch.setattr(markdown_module, "MarkdownHeadingJumpsConstraint", lambda: true_constraint)
    monkeypatch.setattr(
        markdown_module, "MarkdownLinksAndImagesConstraint", lambda: true_constraint
    )
    monkeypatch.setattr(
        markdown_module, "MarkdownHeadingsStructureConstraint", lambda: true_constraint
    )
    monkeypatch.setattr(
        markdown_module, "MarkdownReferenceLinksConstraint", lambda: true_constraint
    )
    monkeypatch.setattr(
        markdown_module, "MarkdownListStructureConstraint", lambda: true_constraint
    )
    monkeypatch.setattr(
        markdown_module, "MarkdownTableStructureConstraint", lambda: true_constraint
    )
    monkeypatch.setattr(
        markdown_module, "MarkdownUnconsumedEmphasisMarkersConstraint", lambda: true_constraint
    )

    assert MarkdownFormatConstraint().evaluate("# ok")[0] is True

    monkeypatch.setattr(
        markdown_module, "MarkdownUnconsumedEmphasisMarkersConstraint", lambda: false_constraint
    )
    assert MarkdownFormatConstraint().evaluate("# ok")[0] is False
    assert false_constraint.calls == 1


def test_markdown_format_constraint_instructions_indent_bullets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyConstraint:
        def __init__(self, instruction: str) -> None:
            self._instruction = instruction

        def instructions(self, train_or_test: str = "train") -> str:
            return self._instruction

    nested = DummyConstraint("概要:\n- bullet1\n- bullet2")
    plain = DummyConstraint("単体の説明")
    mapping = [
        ("MarkdownParseableConstraint", nested),
        ("MarkdownClosedFencesConstraint", plain),
        ("MarkdownHeadingJumpsConstraint", plain),
        ("MarkdownLinksAndImagesConstraint", plain),
        ("MarkdownHeadingsStructureConstraint", plain),
        ("MarkdownReferenceLinksConstraint", plain),
        ("MarkdownListStructureConstraint", plain),
        ("MarkdownTableStructureConstraint", plain),
        ("MarkdownUnconsumedEmphasisMarkersConstraint", plain),
    ]
    for name, instance in mapping:
        monkeypatch.setattr(markdown_module, name, lambda inst=instance: inst)

    instructions = MarkdownFormatConstraint(seed=123).instructions()

    assert "- 概要:" in instructions
    assert "\n  - bullet1" in instructions
    assert "\n  - bullet2" in instructions
    assert "\n- - bullet1" not in instructions
    assert "- 単体の説明" in instructions

    assert "mdformat" not in instructions


def test_markdown_parseable_constraint_valid() -> None:
    assert MarkdownParseableConstraint().evaluate("# Heading\n\nText")[0] is True


def test_markdown_parseable_constraint_handles_parser_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DummyParser:
        def parse(self, _value: str) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(markdown_module, "MarkdownIt", lambda *_args, **_kwargs: DummyParser())
    assert MarkdownParseableConstraint().evaluate("invalid")[0] is False


def test_markdown_closed_fences_constraint_detects_unclosed() -> None:
    value = "```python\nprint('hi')\n"
    assert MarkdownClosedFencesConstraint().evaluate(value)[0] is False


def test_markdown_closed_fences_constraint_accepts_closed() -> None:
    value = "```\ncode\n```"
    assert MarkdownClosedFencesConstraint().evaluate(value)[0] is True


def test_markdown_heading_jumps_constraint_detects_jump() -> None:
    value = dedent(
        """
        # Title
        ### Jump
        """
    )
    assert MarkdownHeadingJumpsConstraint().evaluate(value)[0] is False


def test_markdown_links_and_images_constraint_checks_scheme() -> None:
    token = SimpleNamespace(
        type="link_open", attrs=[("href", "webdav://example.com")], children=None
    )
    link_tokens: list["Token"] = [cast("Token", token)]
    issues = MarkdownLinksAndImagesConstraint._check_links_and_images(link_tokens)  # NOQA
    assert issues


def test_markdown_links_and_images_constraint_requires_alt_text() -> None:
    value = "![](image.png)"
    assert MarkdownLinksAndImagesConstraint().evaluate(value)[0] is False


def test_markdown_headings_structure_missing_h1() -> None:
    value = "## Subheading"
    assert MarkdownHeadingsStructureConstraint().evaluate(value)[0] is False


def test_markdown_reference_links_constraint_detects_undefined() -> None:
    value = "[link][missing]"
    assert MarkdownReferenceLinksConstraint().evaluate(value)[0] is False


def test_markdown_list_structure_constraint_detects_empty_item() -> None:
    tokens = [
        SimpleNamespace(type="list_item_open", map=(0, 1)),
        SimpleNamespace(type="list_item_close"),
    ]
    typed_tokens: list["Token"] = [cast("Token", t) for t in tokens]
    issues = MarkdownListStructureConstraint._check_no_empty_list_items(typed_tokens)  # NOQA
    assert issues


def test_markdown_list_structure_constraint_allows_year_list() -> None:
    text = "# Events\n2000. an event\n2001. another event"
    assert MarkdownListStructureConstraint().evaluate(text)[0] is True


def test_markdown_table_structure_constraint_detects_row_length() -> None:
    def tok(t_type: str, content: str | None = None) -> SimpleNamespace:
        return SimpleNamespace(type=t_type, content=content)

    tokens = [
        tok("table_open"),
        tok("tr_open"),
        tok("th_open"),
        tok("inline", "H1"),
        tok("th_close"),
        tok("th_open"),
        tok("inline", "H2"),
        tok("th_close"),
        tok("tr_close"),
        tok("tr_open"),
        tok("td_open"),
        tok("inline", "a"),
        tok("td_close"),
        tok("td_open"),
        tok("inline", "b"),
        tok("td_close"),
        tok("tr_close"),
        tok("tr_open"),
        tok("td_open"),
        tok("inline", "c"),
        tok("td_close"),
        tok("tr_close"),
        tok("table_close"),
    ]
    typed_table_tokens: list["Token"] = [cast("Token", t) for t in tokens]
    issues = MarkdownTableStructureConstraint.check_tables(typed_table_tokens)
    assert issues


def test_markdown_unconsumed_emphasis_markers_constraint_detects_issue() -> None:
    value = "word * text"
    assert MarkdownUnconsumedEmphasisMarkersConstraint().evaluate(value)[0] is False
