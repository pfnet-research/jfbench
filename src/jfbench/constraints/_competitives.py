import ast
import itertools
from pathlib import Path


def _discover_constraint_names() -> set[str]:
    root = Path(__file__).resolve().parent
    names: set[str] = set()
    for path in root.rglob("*.py"):
        if path.name in {"_group.py", "_competitives.py"}:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and (
                node.name.endswith("Constraint") or node.name.endswith("Constraints")
            ):
                names.add(node.name)
    return names


ALL_CONSTRAINT_NAMES: set[str] = _discover_constraint_names()


TEXTUAL_FORMATS = {
    "BulletPointsFormatConstraint",
    "CitationFormatConstraint",
    "DiffFormatConstraint",
}

SERIALIZED_FORMATS = {
    "CsvFormatConstraint",
    "JsonFormatConstraint",
    "TsvFormatConstraint",
    "XmlFormatConstraint",
    "YamlFormatConstraint",
}

CODE_FORMATS = {
    "JavascriptFormatConstraint",
    "NoCodeFenceJavascriptFormatConstraint",
    "PythonFormatConstraint",
    "NoCodeFencePythonFormatConstraint",
    "WithCodeFenceJavascriptFormatConstraint",
    "WithCodeFencePythonFormatConstraint",
}

MARKUP_FORMATS = {
    "HtmlFormatConstraint",
    "LatexFormatConstraint",
}

TABLE_FORMATS = {
    "HtmlTableFormatConstraint",
    "LatexTableFormatConstraint",
    "MarkdownTableFormatConstraint",
    "MediawikiTableFormatConstraint",
}

NON_MARKDOWN_TABLE_FORMATS = {
    "HtmlTableFormatConstraint",
    "LatexTableFormatConstraint",
    "MediawikiTableFormatConstraint",
}

MARKDOWN_FORMATS = {
    "MarkdownClosedFencesConstraint",
    "MarkdownFormatConstraint",
    "MarkdownHeadingJumpsConstraint",
    "MarkdownHeadingsStructureConstraint",
    "MarkdownLinksAndImagesConstraint",
    "MarkdownListStructureConstraint",
    "MarkdownParseableConstraint",
    "MarkdownReferenceLinksConstraint",
    "MarkdownTableStructureConstraint",
    "MarkdownUnconsumedEmphasisMarkersConstraint",
}

STRUCTURE_LLMS = {
    "HeadingStructureConstraint",
    "SectionStructureConstraint",
}

SCRIPT_ONLY = {
    "HiraganaCharacterConstraint",
    "KatakanaCharacterConstraint",
    "KanjiCharacterConstraint",
    "RomajiCharacterConstraint",
}

WIDTH_EXCLUSIVE = {
    "FullWidthCharacterConstraint",
    "HalfWidthCharacterConstraint",
}

ASCII_CASE = {
    "UppercaseCharacterConstraint",
    "LowercaseCharacterConstraint",
}

ASCII_REQUIRED = {
    "CamelcaseNotationConstraint",
    "SnakecaseNotationConstraint",
    "TitlecaseNotationConstraint",
    "RomajiCharacterConstraint",
    *ASCII_CASE,
}

TENSE_EXCLUSIVE = {
    "PastTenseStyleConstraint",
    "PresentTenseStyleConstraint",
}

POLITENESS_EXCLUSIVE = {
    "PlainStyleConstraint",
    "PoliteStyleConstraint",
}

VOCABULARY_EXCLUSIVE = {
    "EasyVocabularyStyleConstraint",
    "DifficultVocacularyStyleConstraint",
}

EMOTIONAL_STYLES_POSITIVE = {
    "HappyEmotionalStyleConstraint",
    "JoyfulEmotionalStyleConstraint",
}

EMOTIONAL_STYLES_NEGATIVE = {
    "AngryEmotionalStyleConstraint",
    "SadEmotionalStyleConstraint",
}

MUTUALLY_EXCLUSIVE_GROUPS: tuple[set[str], ...] = (
    SERIALIZED_FORMATS,
    CODE_FORMATS,
    MARKUP_FORMATS,
    TABLE_FORMATS,
    TEXTUAL_FORMATS,
    SCRIPT_ONLY,
    WIDTH_EXCLUSIVE,
    ASCII_CASE,
    TENSE_EXCLUSIVE,
    POLITENESS_EXCLUSIVE,
    VOCABULARY_EXCLUSIVE,
)


CROSS_GROUP_CONFLICTS: tuple[tuple[set[str], set[str]], ...] = (
    (SERIALIZED_FORMATS, CODE_FORMATS),
    (SERIALIZED_FORMATS, MARKUP_FORMATS),
    (SERIALIZED_FORMATS, TABLE_FORMATS),
    (SERIALIZED_FORMATS, TEXTUAL_FORMATS),
    (SERIALIZED_FORMATS, MARKDOWN_FORMATS),
    (CODE_FORMATS, MARKUP_FORMATS),
    (CODE_FORMATS, TABLE_FORMATS),
    (CODE_FORMATS, TEXTUAL_FORMATS),
    (CODE_FORMATS, MARKDOWN_FORMATS),
    (MARKUP_FORMATS, TABLE_FORMATS),
    (MARKUP_FORMATS, TEXTUAL_FORMATS),
    (MARKUP_FORMATS, MARKDOWN_FORMATS),
    (TABLE_FORMATS, TEXTUAL_FORMATS),
    (TABLE_FORMATS, STRUCTURE_LLMS),
    (NON_MARKDOWN_TABLE_FORMATS, MARKDOWN_FORMATS),
    (STRUCTURE_LLMS, SERIALIZED_FORMATS),
    (STRUCTURE_LLMS, CODE_FORMATS),
    (STRUCTURE_LLMS, MARKUP_FORMATS),
    (STRUCTURE_LLMS, MARKUP_FORMATS),
    (EMOTIONAL_STYLES_POSITIVE, EMOTIONAL_STYLES_NEGATIVE),
    ({"FullWidthCharacterConstraint"}, ASCII_REQUIRED),
    ({"HalfWidthCharacterConstraint"}, SCRIPT_ONLY - {"RomajiCharacterConstraint"}),
    (WIDTH_EXCLUSIVE, ASCII_REQUIRED),
    (ASCII_CASE, SCRIPT_ONLY - {"RomajiCharacterConstraint"}),
    (ASCII_REQUIRED, SCRIPT_ONLY - {"RomajiCharacterConstraint"}),
    ({"JapaneseStyleConstraint"}, ASCII_REQUIRED),
    ({"NoJapaneseStyleConstraint"}, SCRIPT_ONLY),
)


LANGUAGE_CONFLICTS = {
    ("JapaneseStyleConstraint", "EnglishStyleConstraint"),
    ("JapaneseStyleConstraint", "BritishEnglishStyleConstraint"),
    ("JapaneseStyleConstraint", "AmericanEnglishStyleConstraint"),
    ("BritishEnglishStyleConstraint", "AmericanEnglishStyleConstraint"),
    ("EnglishStyleConstraint", "NoEnglishStyleConstraint"),
    ("BritishEnglishStyleConstraint", "NoEnglishStyleConstraint"),
    ("AmericanEnglishStyleConstraint", "NoEnglishStyleConstraint"),
    ("JapaneseStyleConstraint", "NoJapaneseStyleConstraint"),
}


WHITESPACE_FORBIDDEN = {"NoWhitespaceConstraint"}
WHITESPACE_REQUIRED = {
    "BlankLinesLengthConstraint",
    "CitationFormatConstraint",
    "DiffFormatConstraint",
    "HtmlFormatConstraint",
    "HtmlTableFormatConstraint",
    "IndentFormatConstraint",
    "JavascriptFormatConstraint",
    "LatexTableFormatConstraint",
    "LineIndentFormatIfbenchConstraint",
    "MarkdownHeadingJumpsConstraint",
    "MarkdownHeadingsStructureConstraint",
    "MarkdownListStructureConstraint",
    "MarkdownTableFormatConstraint",
    "MarkdownTableStructureConstraint",
    "MediawikiTableFormatConstraint",
    "NewLinesLengthConstraint",
    "NewlineFormatIfbenchConstraint",
    "NoCodeFenceJavascriptFormatConstraint",
    "NoCodeFencePythonFormatConstraint",
    "ParagraphsLengthConstraint",
    "PythonFormatConstraint",
    "QuoteUnquoteFormatIfbenchConstraint",
    "SectionsLengthConstraint",
    "SubBulletsFormatIfbenchConstraint",
    "TitlecaseNotationConstraint",
    "TsvFormatConstraint",
    "WithCodeFenceJavascriptFormatConstraint",
    "WithCodeFencePythonFormatConstraint",
    "XmlFormatConstraint",
}

DOC_GROUNDED_CONSTRAINTS = {
    "AbstractIncludedContentConstraint",
    "IntrinsicContentConstraint",
}
DOC_IRRELEVANT_CONSTRAINTS = {"IrrevantContentConstraint", "SimpleRepeatIfbenchConstraint"}
CONTENT_CONSTRAINTS = {
    "AbstractExcludedContentConstraint",
    "AbstractIncludedContentConstraint",
    "IntrinsicContentConstraint",
    "IrrevantContentConstraint",
    "KeywordExcludedContentConstraint",
    "KeywordIncludedContentConstraint",
    "NoReasonContentConstraint",
    "ReasonContentConstraint",
}
IFBENCH_REPEAT_CONSTRAINTS = {
    "ChangeRepeatIfbenchConstraint",
    "SimpleRepeatIfbenchConstraint",
}


EXPLICIT_CONFLICTS = {
    frozenset(pair)
    for pair in {
        ("MarkdownFormatConstraint", "MarkdownTableFormatConstraint"),
        ("MarkdownHeadingJumpsConstraint", "MarkdownTableFormatConstraint"),
        ("MarkdownHeadingsStructureConstraint", "MarkdownTableFormatConstraint"),
        ("KanjiNumeralsNotationConstraint", "NoKanjiNumeralsNotationConstraint"),
        ("NoHyphenJpPostalCodeNotationConstraint", "WithHyphenJpPostalCodeNotationConstraint"),
        ("NoHyphenPhoneNumberNotationConstraint", "WithHyphenPhoneNumberNotationConstraint"),
        ("JapanesePunctuationConstraint", "NoJapanesePunctuationConstraint"),
        ("JapanesePunctuationConstraint", "CsvFormatConstraint"),
        ("JapanesePunctuationConstraint", "JsonFormatConstraint"),
        ("NoCommasConstraint", "CsvFormatConstraint"),
        ("NoCommasConstraint", "JsonFormatConstraint"),
        ("NoJapaneseStyleConstraint", "JapanesePunctuationConstraint"),
        ("NoJapaneseStyleConstraint", "KanjiNumeralsNotationConstraint"),
        ("NoJapaneseStyleConstraint", "FuriganaNotationConstraint"),
        ("ReasonContentConstraint", "NoExplanationConstraint"),
        ("SelfAttestationConstraint", "NoSelfReferenceConstraint"),
        ("ReasonContentConstraint", "NoReasonContentConstraint"),
        ("ExplanationConstraint", "NoExplanationConstraint"),
        ("GreetingOutputConstraint", "NoGreetingOutputConstraint"),
        ("SelfReferenceConstraint", "NoSelfReferenceConstraint"),
        ("SelfAttestationConstraint", "NoSelfAttestationConstraint"),
        ("NoCodeFenceJavascriptFormatConstraint", "WithCodeFenceJavascriptFormatConstraint"),
        ("NoCodeFencePythonFormatConstraint", "WithCodeFencePythonFormatConstraint"),
        ("PoliteStyleConstraint", "PlainStyleConstraint"),
        ("CasualToneStyleConstraint", "AcademicToneStyleConstraint"),
        ("CasualToneStyleConstraint", "BusinessToneStyleConstraint"),
        ("EasyVocabularyStyleConstraint", "DifficultVocacularyStyleConstraint"),
        ("ImpersonalStyleConstraint", "FirstPersonPluralStyleConstraint"),
        ("ImpersonalStyleConstraint", "FirstPersonSingularStyleConstraint"),
        ("UppercaseCharacterConstraint", "LowercaseCharacterConstraint"),
        ("UppercaseCharacterConstraint", "HtmlFormatConstraint"),
        ("UppercaseCharacterConstraint", "JavascriptFormatConstraint"),
        ("UppercaseCharacterConstraint", "LatexFormatConstraint"),
        ("UppercaseCharacterConstraint", "LatexTableFormatConstraint"),
        ("UppercaseCharacterConstraint", "MarkdownLinksAndImagesConstraint"),
        ("UppercaseCharacterConstraint", "MediawikiTableFormatConstraint"),
        ("UppercaseCharacterConstraint", "NoCodeFenceJavascriptFormatConstraint"),
        ("UppercaseCharacterConstraint", "NoCodeFencePythonFormatConstraint"),
        ("UppercaseCharacterConstraint", "PythonFormatConstraint"),
        ("UppercaseCharacterConstraint", "WithCodeFenceJavascriptFormatConstraint"),
        ("UppercaseCharacterConstraint", "WithCodeFencePythonFormatConstraint"),
        ("UppercaseCharacterConstraint", "XmlFormatConstraint"),
        ("FullWidthCharacterConstraint", "HalfWidthCharacterConstraint"),
        ("FormalToneStyleConstraint", "CasualToneStyleConstraint"),
        ("PoliteStyleConstraint", "CasualToneStyleConstraint"),
        ("CamelcaseNotationConstraint", "LowercaseCharacterConstraint"),
        ("CamelcaseNotationConstraint", "UppercaseCharacterConstraint"),
        ("TitlecaseNotationConstraint", "LowercaseCharacterConstraint"),
        ("TitlecaseNotationConstraint", "UppercaseCharacterConstraint"),
        ("SnakecaseNotationConstraint", "UppercaseCharacterConstraint"),
        ("EnglishStyleConstraint", "HiraganaCharacterConstraint"),
        ("EnglishStyleConstraint", "KatakanaCharacterConstraint"),
        ("EnglishStyleConstraint", "KanjiCharacterConstraint"),
        ("EnglishStyleConstraint", "JapaneseStyleConstraint"),
        ("EnglishStyleConstraint", "JapanesePunctuationConstraint"),
        ("RomajiCharacterConstraint", "EnglishStyleConstraint"),
        ("RomajiCharacterConstraint", "JapaneseStyleConstraint"),
        ("RomajiCharacterConstraint", "HtmlFormatConstraint"),
        ("RomajiCharacterConstraint", "JavascriptFormatConstraint"),
        ("RomajiCharacterConstraint", "LatexFormatConstraint"),
        ("RomajiCharacterConstraint", "NoCodeFenceJavascriptFormatConstraint"),
        ("RomajiCharacterConstraint", "NoCodeFencePythonFormatConstraint"),
        ("RomajiCharacterConstraint", "PythonFormatConstraint"),
        ("RomajiCharacterConstraint", "WithCodeFenceJavascriptFormatConstraint"),
        ("RomajiCharacterConstraint", "WithCodeFencePythonFormatConstraint"),
        ("RomajiCharacterConstraint", "XmlFormatConstraint"),
        ("HiraganaCharacterConstraint", "HtmlFormatConstraint"),
        ("HiraganaCharacterConstraint", "JavascriptFormatConstraint"),
        ("HiraganaCharacterConstraint", "LatexFormatConstraint"),
        ("HiraganaCharacterConstraint", "NoCodeFenceJavascriptFormatConstraint"),
        ("HiraganaCharacterConstraint", "NoCodeFencePythonFormatConstraint"),
        ("HiraganaCharacterConstraint", "PythonFormatConstraint"),
        ("HiraganaCharacterConstraint", "WithCodeFenceJavascriptFormatConstraint"),
        ("HiraganaCharacterConstraint", "WithCodeFencePythonFormatConstraint"),
        ("HiraganaCharacterConstraint", "XmlFormatConstraint"),
        ("KatakanaCharacterConstraint", "HtmlFormatConstraint"),
        ("KatakanaCharacterConstraint", "JavascriptFormatConstraint"),
        ("KatakanaCharacterConstraint", "LatexFormatConstraint"),
        ("KatakanaCharacterConstraint", "NoCodeFenceJavascriptFormatConstraint"),
        ("KatakanaCharacterConstraint", "NoCodeFencePythonFormatConstraint"),
        ("KatakanaCharacterConstraint", "PythonFormatConstraint"),
        ("KatakanaCharacterConstraint", "WithCodeFenceJavascriptFormatConstraint"),
        ("KatakanaCharacterConstraint", "WithCodeFencePythonFormatConstraint"),
        ("KatakanaCharacterConstraint", "XmlFormatConstraint"),
        ("KanjiCharacterConstraint", "HtmlFormatConstraint"),
        ("KanjiCharacterConstraint", "JavascriptFormatConstraint"),
        ("KanjiCharacterConstraint", "LatexFormatConstraint"),
        ("KanjiCharacterConstraint", "NoCodeFenceJavascriptFormatConstraint"),
        ("KanjiCharacterConstraint", "NoCodeFencePythonFormatConstraint"),
        ("KanjiCharacterConstraint", "PythonFormatConstraint"),
        ("KanjiCharacterConstraint", "WithCodeFenceJavascriptFormatConstraint"),
        ("KanjiCharacterConstraint", "WithCodeFencePythonFormatConstraint"),
        ("KanjiCharacterConstraint", "XmlFormatConstraint"),
        ("FullWidthCharacterConstraint", "NoJapanesePunctuationConstraint"),
        ("FullWidthCharacterConstraint", "CsvFormatConstraint"),
        ("FullWidthCharacterConstraint", "DiffFormatConstraint"),
        ("FullWidthCharacterConstraint", "HtmlFormatConstraint"),
        ("FullWidthCharacterConstraint", "HtmlTableFormatConstraint"),
        ("FullWidthCharacterConstraint", "JavascriptFormatConstraint"),
        ("FullWidthCharacterConstraint", "JsonFormatConstraint"),
        ("FullWidthCharacterConstraint", "LatexFormatConstraint"),
        ("FullWidthCharacterConstraint", "LatexTableFormatConstraint"),
        ("FullWidthCharacterConstraint", "MediawikiTableFormatConstraint"),
        ("FullWidthCharacterConstraint", "NoCodeFenceJavascriptFormatConstraint"),
        ("FullWidthCharacterConstraint", "NoCodeFencePythonFormatConstraint"),
        ("FullWidthCharacterConstraint", "PythonFormatConstraint"),
        ("FullWidthCharacterConstraint", "SentenceDelimiterFormatConstraint"),
        ("FullWidthCharacterConstraint", "TsvFormatConstraint"),
        ("FullWidthCharacterConstraint", "WithCodeFenceJavascriptFormatConstraint"),
        ("FullWidthCharacterConstraint", "WithCodeFencePythonFormatConstraint"),
        ("FullWidthCharacterConstraint", "XmlFormatConstraint"),
        ("FullWidthCharacterConstraint", "YamlFormatConstraint"),
        ("IntrinsicContentConstraint", "IrrevantContentConstraint"),
        ("AbstractIncludedContentConstraint", "IrrevantContentConstraint"),
        *LANGUAGE_CONFLICTS,
    }
}


def _in_same_exclusive_group(left: str, right: str) -> bool:
    return any(left in group and right in group for group in MUTUALLY_EXCLUSIVE_GROUPS)


def _in_cross_group_conflict(left: str, right: str) -> bool:
    return any(
        (left in lhs and right in rhs) or (left in rhs and right in lhs)
        for lhs, rhs in CROSS_GROUP_CONFLICTS
    )


def _requires_whitespace_conflict(left: str, right: str) -> bool:
    return (left in WHITESPACE_FORBIDDEN and right in WHITESPACE_REQUIRED) or (
        right in WHITESPACE_FORBIDDEN and left in WHITESPACE_REQUIRED
    )


def _doc_scope_conflict(left: str, right: str) -> bool:
    return (left in DOC_GROUNDED_CONSTRAINTS and right in DOC_IRRELEVANT_CONSTRAINTS) or (
        right in DOC_GROUNDED_CONSTRAINTS and left in DOC_IRRELEVANT_CONSTRAINTS
    )


def _are_competitive_pair(left: str, right: str) -> bool:
    if left == right:
        return False
    if left in IFBENCH_REPEAT_CONSTRAINTS or right in IFBENCH_REPEAT_CONSTRAINTS:
        return True
    if left in CONTENT_CONSTRAINTS and right in CONTENT_CONSTRAINTS:
        return True
    pair = frozenset((left, right))
    if pair in EXPLICIT_CONFLICTS:
        return True
    if _in_same_exclusive_group(left, right):
        return True
    if _in_cross_group_conflict(left, right):
        return True
    if _requires_whitespace_conflict(left, right):
        return True
    if _doc_scope_conflict(left, right):
        return True
    return False


def _build_graph() -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {name: set() for name in ALL_CONSTRAINT_NAMES}
    for left, right in itertools.combinations(sorted(ALL_CONSTRAINT_NAMES), 2):
        if _are_competitive_pair(left, right):
            graph[left].add(right)
            graph[right].add(left)
    return graph


COMPETITIVE_CONSTRAINTS: dict[str, tuple[str, ...]] = {
    name: tuple(sorted(conflicts)) for name, conflicts in _build_graph().items()
}


__all__ = ["COMPETITIVE_CONSTRAINTS"]
