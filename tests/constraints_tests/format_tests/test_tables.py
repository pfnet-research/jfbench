from jfbench.constraints.format.tables import HtmlTableFormatConstraint
from jfbench.constraints.format.tables import LatexTableFormatConstraint
from jfbench.constraints.format.tables import MarkdownTableFormatConstraint
from jfbench.constraints.format.tables import MediawikiTableFormatConstraint


def test_markdown_table_constraint_valid() -> None:
    table = "col1|col2\n---|---\na|b\nc|d\n"
    assert MarkdownTableFormatConstraint().evaluate(table)[0] is True


def test_markdown_table_constraint_inconsistent_columns() -> None:
    table = "col1|col2\n---|---\na|b|c\n"
    constraint = MarkdownTableFormatConstraint()
    assert constraint.evaluate(table)[0] is False


def test_html_table_constraint_valid() -> None:
    html = "<table><tbody><tr><td>a</td></tr></tbody></table>"
    assert HtmlTableFormatConstraint().evaluate(html)[0] is True


def test_html_table_constraint_requires_single_table() -> None:
    html = "<div>wrap</div><table><tr><td>a</td></tr></table>"
    constraint = HtmlTableFormatConstraint()
    assert constraint.evaluate(html)[0] is False


def test_latex_table_constraint_valid() -> None:
    latex = (
        r"\begin{tabular}{c}"
        "\n"
        r"A \\"
        "\n"
        r"\end{tabular}"
    )
    assert LatexTableFormatConstraint().evaluate(latex)[0] is True


def test_latex_table_constraint_allows_math_wrap() -> None:
    latex = (
        r"\["
        "\n"
        r"\begin{tabular}{c}"
        "\n"
        r"A \\"
        "\n"
        r"\end{tabular}"
        "\n"
        r"\]"
    )
    assert LatexTableFormatConstraint().evaluate(latex)[0] is True


def test_latex_table_constraint_requires_tabular() -> None:
    latex = r"\begin{array}{c}A \\ \end{array}"
    constraint = LatexTableFormatConstraint()
    assert constraint.evaluate(latex)[0] is False


def test_mediawiki_table_constraint_valid() -> None:
    wiki = "{|\n! Header\n|-\n| Cell\n|}\n"
    assert MediawikiTableFormatConstraint().evaluate(wiki)[0] is True


def test_mediawiki_table_constraint_requires_closing() -> None:
    wiki = "{|\n! Header\n|-\n| Cell\n"
    constraint = MediawikiTableFormatConstraint()
    assert constraint.evaluate(wiki)[0] is False
