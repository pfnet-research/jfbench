from jfbench.constraints.format.citation import CitationFormatConstraint


def test_parenthetical_citation_with_references() -> None:
    value = (
        "Recent studies highlight the importance of context (Smith, 2020).\n\n"
        "References\n"
        "Smith, J. 2020. Understanding Context. Academic Press."
    )
    assert CitationFormatConstraint().evaluate(value)[0] is True


def test_numeric_citation_with_references() -> None:
    value = (
        "The approach improves accuracy [1].\n\n"
        "References\n"
        "[1] Smith, J., 2020. Understanding Context. Academic Press."
    )
    assert CitationFormatConstraint().evaluate(value)[0] is True


def test_footnote_citation_with_references() -> None:
    value = (
        "Detailed analysis is provided elsewhere [^1].\n\n"
        "References\n"
        "[^1] Doe, J., 2021. Detailed Analysis. Research House."
    )
    assert CitationFormatConstraint().evaluate(value)[0] is True


def test_missing_references_entries_fails() -> None:
    value = "The approach improves accuracy (Smith, 2020)."
    assert CitationFormatConstraint().evaluate(value)[0] is False


def test_references_without_header_are_accepted() -> None:
    value = (
        "The approach improves accuracy [1].\n\n"
        "[1] Smith, J., 2020. Understanding Context. Academic Press."
    )
    assert CitationFormatConstraint().evaluate(value)[0] is True


def test_reference_without_year_fails() -> None:
    value = (
        "The approach improves accuracy [1].\n\n"
        "References\n"
        "[1] Smith, J. Understanding Context. Academic Press."
    )
    assert CitationFormatConstraint().evaluate(value)[0] is False


def test_numeric_citation_missing_entry_fails() -> None:
    value = (
        "The approach improves accuracy [2].\n\n"
        "References\n"
        "[1] Smith, J., 2020. Understanding Context. Academic Press."
    )
    assert CitationFormatConstraint().evaluate(value)[0] is False


def test_additional_content_allowed_after_references() -> None:
    value = (
        "The approach improves accuracy [1].\n\n"
        "References\n"
        "[1] Smith, J., 2020. Understanding Context. Academic Press.\n\n"
        "Appendix A: Further discussion."
    )
    assert CitationFormatConstraint().evaluate(value)[0] is True
