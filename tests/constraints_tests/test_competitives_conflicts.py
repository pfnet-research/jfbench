from jfbench.constraints._competitives import COMPETITIVE_CONSTRAINTS


def _assert_conflict(a: str, b: str) -> None:
    assert b in COMPETITIVE_CONSTRAINTS[a]
    assert a in COMPETITIVE_CONSTRAINTS[b]


def test_width_and_script_conflicts() -> None:
    _assert_conflict("FullWidthCharacterConstraint", "CamelcaseNotationConstraint")
    _assert_conflict("HalfWidthCharacterConstraint", "KanjiCharacterConstraint")


def test_uppercase_and_format_conflicts() -> None:
    _assert_conflict("UppercaseCharacterConstraint", "HtmlFormatConstraint")
    _assert_conflict("UppercaseCharacterConstraint", "JavascriptFormatConstraint")
    _assert_conflict("UppercaseCharacterConstraint", "LatexFormatConstraint")
    _assert_conflict("UppercaseCharacterConstraint", "LatexTableFormatConstraint")
    _assert_conflict("UppercaseCharacterConstraint", "MarkdownLinksAndImagesConstraint")
    _assert_conflict("UppercaseCharacterConstraint", "MediawikiTableFormatConstraint")
    _assert_conflict("UppercaseCharacterConstraint", "NoCodeFenceJavascriptFormatConstraint")
    _assert_conflict("UppercaseCharacterConstraint", "NoCodeFencePythonFormatConstraint")
    _assert_conflict("UppercaseCharacterConstraint", "PythonFormatConstraint")
    _assert_conflict("UppercaseCharacterConstraint", "WithCodeFenceJavascriptFormatConstraint")
    _assert_conflict("UppercaseCharacterConstraint", "WithCodeFencePythonFormatConstraint")
    _assert_conflict("UppercaseCharacterConstraint", "XmlFormatConstraint")


def test_pronoun_and_tense_conflicts() -> None:
    _assert_conflict("ImpersonalStyleConstraint", "FirstPersonSingularStyleConstraint")
    _assert_conflict("PastTenseStyleConstraint", "PresentTenseStyleConstraint")


def test_vocabulary_and_whitespace_conflicts() -> None:
    _assert_conflict("EasyVocabularyStyleConstraint", "DifficultVocacularyStyleConstraint")
    _assert_conflict("TitlecaseNotationConstraint", "NoWhitespaceConstraint")


def test_language_and_script_conflicts() -> None:
    _assert_conflict("JapaneseStyleConstraint", "CamelcaseNotationConstraint")
    _assert_conflict("EnglishStyleConstraint", "HiraganaCharacterConstraint")
    _assert_conflict("NoJapaneseStyleConstraint", "KanjiNumeralsNotationConstraint")


def test_structure_and_whitespace_conflicts() -> None:
    _assert_conflict("NoWhitespaceConstraint", "NewlineFormatIfbenchConstraint")
    _assert_conflict("NoWhitespaceConstraint", "ParagraphsLengthConstraint")


def test_additional_whitespace_conflicts() -> None:
    _assert_conflict("NoWhitespaceConstraint", "DiffFormatConstraint")
    _assert_conflict("NoWhitespaceConstraint", "HtmlFormatConstraint")
    _assert_conflict("NoWhitespaceConstraint", "HtmlTableFormatConstraint")
    _assert_conflict("NoWhitespaceConstraint", "JavascriptFormatConstraint")
    _assert_conflict("NoWhitespaceConstraint", "LatexTableFormatConstraint")
    _assert_conflict("NoWhitespaceConstraint", "MediawikiTableFormatConstraint")
    _assert_conflict("NoWhitespaceConstraint", "NoCodeFenceJavascriptFormatConstraint")
    _assert_conflict("NoWhitespaceConstraint", "NoCodeFencePythonFormatConstraint")
    _assert_conflict("NoWhitespaceConstraint", "PythonFormatConstraint")
    _assert_conflict("NoWhitespaceConstraint", "QuoteUnquoteFormatIfbenchConstraint")
    _assert_conflict("NoWhitespaceConstraint", "TsvFormatConstraint")
    _assert_conflict("NoWhitespaceConstraint", "WithCodeFenceJavascriptFormatConstraint")
    _assert_conflict("NoWhitespaceConstraint", "WithCodeFencePythonFormatConstraint")
    _assert_conflict("NoWhitespaceConstraint", "XmlFormatConstraint")


def test_document_scope_conflicts() -> None:
    _assert_conflict("IntrinsicContentConstraint", "IrrevantContentConstraint")
    _assert_conflict("AbstractIncludedContentConstraint", "IrrevantContentConstraint")


def test_romaji_and_format_conflicts() -> None:
    _assert_conflict("RomajiCharacterConstraint", "HtmlFormatConstraint")
    _assert_conflict("RomajiCharacterConstraint", "JavascriptFormatConstraint")
    _assert_conflict("RomajiCharacterConstraint", "LatexFormatConstraint")
    _assert_conflict("RomajiCharacterConstraint", "NoCodeFenceJavascriptFormatConstraint")
    _assert_conflict("RomajiCharacterConstraint", "NoCodeFencePythonFormatConstraint")
    _assert_conflict("RomajiCharacterConstraint", "PythonFormatConstraint")
    _assert_conflict("RomajiCharacterConstraint", "WithCodeFenceJavascriptFormatConstraint")
    _assert_conflict("RomajiCharacterConstraint", "WithCodeFencePythonFormatConstraint")
    _assert_conflict("RomajiCharacterConstraint", "XmlFormatConstraint")


def test_hiragana_and_format_conflicts() -> None:
    _assert_conflict("HiraganaCharacterConstraint", "HtmlFormatConstraint")
    _assert_conflict("HiraganaCharacterConstraint", "JavascriptFormatConstraint")
    _assert_conflict("HiraganaCharacterConstraint", "LatexFormatConstraint")
    _assert_conflict("HiraganaCharacterConstraint", "NoCodeFenceJavascriptFormatConstraint")
    _assert_conflict("HiraganaCharacterConstraint", "NoCodeFencePythonFormatConstraint")
    _assert_conflict("HiraganaCharacterConstraint", "PythonFormatConstraint")
    _assert_conflict("HiraganaCharacterConstraint", "WithCodeFenceJavascriptFormatConstraint")
    _assert_conflict("HiraganaCharacterConstraint", "WithCodeFencePythonFormatConstraint")
    _assert_conflict("HiraganaCharacterConstraint", "XmlFormatConstraint")


def test_katakana_and_format_conflicts() -> None:
    _assert_conflict("KatakanaCharacterConstraint", "HtmlFormatConstraint")
    _assert_conflict("KatakanaCharacterConstraint", "JavascriptFormatConstraint")
    _assert_conflict("KatakanaCharacterConstraint", "LatexFormatConstraint")
    _assert_conflict("KatakanaCharacterConstraint", "NoCodeFenceJavascriptFormatConstraint")
    _assert_conflict("KatakanaCharacterConstraint", "NoCodeFencePythonFormatConstraint")
    _assert_conflict("KatakanaCharacterConstraint", "PythonFormatConstraint")
    _assert_conflict("KatakanaCharacterConstraint", "WithCodeFenceJavascriptFormatConstraint")
    _assert_conflict("KatakanaCharacterConstraint", "WithCodeFencePythonFormatConstraint")
    _assert_conflict("KatakanaCharacterConstraint", "XmlFormatConstraint")


def test_kanji_and_format_conflicts() -> None:
    _assert_conflict("KanjiCharacterConstraint", "HtmlFormatConstraint")
    _assert_conflict("KanjiCharacterConstraint", "JavascriptFormatConstraint")
    _assert_conflict("KanjiCharacterConstraint", "LatexFormatConstraint")
    _assert_conflict("KanjiCharacterConstraint", "NoCodeFenceJavascriptFormatConstraint")
    _assert_conflict("KanjiCharacterConstraint", "NoCodeFencePythonFormatConstraint")
    _assert_conflict("KanjiCharacterConstraint", "PythonFormatConstraint")
    _assert_conflict("KanjiCharacterConstraint", "WithCodeFenceJavascriptFormatConstraint")
    _assert_conflict("KanjiCharacterConstraint", "WithCodeFencePythonFormatConstraint")
    _assert_conflict("KanjiCharacterConstraint", "XmlFormatConstraint")


def test_fullwidth_and_format_conflicts() -> None:
    _assert_conflict("FullWidthCharacterConstraint", "NoJapanesePunctuationConstraint")
    _assert_conflict("FullWidthCharacterConstraint", "CsvFormatConstraint")
    _assert_conflict("FullWidthCharacterConstraint", "DiffFormatConstraint")
    _assert_conflict("FullWidthCharacterConstraint", "HtmlFormatConstraint")
    _assert_conflict("FullWidthCharacterConstraint", "HtmlTableFormatConstraint")
    _assert_conflict("FullWidthCharacterConstraint", "JavascriptFormatConstraint")
    _assert_conflict("FullWidthCharacterConstraint", "JsonFormatConstraint")
    _assert_conflict("FullWidthCharacterConstraint", "LatexFormatConstraint")
    _assert_conflict("FullWidthCharacterConstraint", "LatexTableFormatConstraint")
    _assert_conflict("FullWidthCharacterConstraint", "MediawikiTableFormatConstraint")
    _assert_conflict("FullWidthCharacterConstraint", "NoCodeFenceJavascriptFormatConstraint")
    _assert_conflict("FullWidthCharacterConstraint", "NoCodeFencePythonFormatConstraint")
    _assert_conflict("FullWidthCharacterConstraint", "PythonFormatConstraint")
    _assert_conflict("FullWidthCharacterConstraint", "SentenceDelimiterFormatConstraint")
    _assert_conflict("FullWidthCharacterConstraint", "TsvFormatConstraint")
    _assert_conflict("FullWidthCharacterConstraint", "WithCodeFenceJavascriptFormatConstraint")
    _assert_conflict("FullWidthCharacterConstraint", "WithCodeFencePythonFormatConstraint")
    _assert_conflict("FullWidthCharacterConstraint", "XmlFormatConstraint")
    _assert_conflict("FullWidthCharacterConstraint", "YamlFormatConstraint")
