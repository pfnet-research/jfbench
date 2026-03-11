import pytest

from jfbench.constraints.format.html import HtmlFormatConstraint


@pytest.fixture
def constraint() -> HtmlFormatConstraint:
    return HtmlFormatConstraint()


def test_evaluate_returns_true_for_valid_html(constraint: HtmlFormatConstraint) -> None:
    html = "<!DOCTYPE html><html><head></head><body><p>Hello</p></body></html>"
    assert constraint.evaluate(html)[0] is True


def test_evaluate_returns_false_for_invalid_html(constraint: HtmlFormatConstraint) -> None:
    html = "plain text without markup"
    assert constraint.evaluate(html)[0] is False


def test_evaluate_unwraps_markdown_html_fence(constraint: HtmlFormatConstraint) -> None:
    html = "```html\n<!DOCTYPE html><html><head></head><body><p>Hello</p></body></html>\n```"
    assert constraint.evaluate(html)[0] is True


def test_evaluate_unwraps_tilde_fence_without_language(
    constraint: HtmlFormatConstraint,
) -> None:
    html = "~~~\n<!DOCTYPE html><html><head></head><body><p>Hello</p></body></html>\n~~~"
    assert constraint.evaluate(html)[0] is True


def test_instructions_returns_expected_message(
    constraint: HtmlFormatConstraint,
) -> None:
    options = {
        "出力全体を有効なHTML5文書として記述してください。",
        "HTMLタグで構成された完全なHTML5文書を生成し、不正な構造を避けてください。",
        "正しいHTML5構文のみを用い、単体でパースできるHTMLを出力してください。",
        "開閉タグや入れ子を整えたHTML5形式で回答を提出してください。",
        "HTML文書として完結するように記述し、構文上の崩れがないことを保証してください。",
    }
    assert constraint.instructions() in options
