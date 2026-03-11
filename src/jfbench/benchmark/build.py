import asyncio
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import inspect
import logging
from typing import Any
from typing import Awaitable
from typing import cast
from typing import Literal

import numpy as np
from tqdm import tqdm

from jfbench.constraints.character import FullWidthCharacterConstraint
from jfbench.constraints.character import HalfWidthCharacterConstraint
from jfbench.constraints.character import HiraganaCharacterConstraint
from jfbench.constraints.character import JapanesePunctuationConstraint
from jfbench.constraints.character import KanjiCharacterConstraint
from jfbench.constraints.character import KatakanaCharacterConstraint
from jfbench.constraints.character import LowercaseCharacterConstraint
from jfbench.constraints.character import NoCommasConstraint
from jfbench.constraints.character import NoJapanesePunctuationConstraint
from jfbench.constraints.character import NoSuffixWhitespaceConstraint
from jfbench.constraints.character import NoWhitespaceConstraint
from jfbench.constraints.character import RomajiCharacterConstraint
from jfbench.constraints.character import UppercaseCharacterConstraint
from jfbench.constraints.content import AbstractExcludedContentConstraint
from jfbench.constraints.content import AbstractIncludedContentConstraint
from jfbench.constraints.content import IntrinsicContentConstraint
from jfbench.constraints.content import IrrevantContentConstraint
from jfbench.constraints.content import KeywordExcludedContentConstraint
from jfbench.constraints.content import KeywordIncludedContentConstraint
from jfbench.constraints.content import NoReasonContentConstraint
from jfbench.constraints.content import ReasonContentConstraint
from jfbench.constraints.format import BulletPointsFormatConstraint
from jfbench.constraints.format import CitationFormatConstraint
from jfbench.constraints.format import CsvFormatConstraint
from jfbench.constraints.format import DiffFormatConstraint
from jfbench.constraints.format import HtmlFormatConstraint
from jfbench.constraints.format import HtmlTableFormatConstraint
from jfbench.constraints.format import IndentFormatConstraint
from jfbench.constraints.format import JavascriptFormatConstraint
from jfbench.constraints.format import JsonFormatConstraint
from jfbench.constraints.format import LatexFormatConstraint
from jfbench.constraints.format import LatexTableFormatConstraint
from jfbench.constraints.format import MarkdownClosedFencesConstraint
from jfbench.constraints.format import MarkdownFormatConstraint
from jfbench.constraints.format import MarkdownHeadingJumpsConstraint
from jfbench.constraints.format import MarkdownHeadingsStructureConstraint
from jfbench.constraints.format import MarkdownLinksAndImagesConstraint
from jfbench.constraints.format import MarkdownListStructureConstraint
from jfbench.constraints.format import MarkdownParseableConstraint
from jfbench.constraints.format import MarkdownReferenceLinksConstraint
from jfbench.constraints.format import MarkdownTableFormatConstraint
from jfbench.constraints.format import MarkdownTableStructureConstraint
from jfbench.constraints.format import MarkdownUnconsumedEmphasisMarkersConstraint
from jfbench.constraints.format import MediawikiTableFormatConstraint
from jfbench.constraints.format import NoCodeFenceJavascriptFormatConstraint
from jfbench.constraints.format import NoCodeFencePythonFormatConstraint
from jfbench.constraints.format import PythonFormatConstraint
from jfbench.constraints.format import SentenceDelimiterFormatConstraint
from jfbench.constraints.format import TsvFormatConstraint
from jfbench.constraints.format import WithCodeFenceJavascriptFormatConstraint
from jfbench.constraints.format import WithCodeFencePythonFormatConstraint
from jfbench.constraints.format import XmlFormatConstraint
from jfbench.constraints.format import YamlFormatConstraint
from jfbench.constraints.ifbench_count import ConjunctionCountIfbenchConstraint
from jfbench.constraints.ifbench_count import NumbersCountIfbenchConstraint
from jfbench.constraints.ifbench_count import PersonNamesCountIfbenchConstraint
from jfbench.constraints.ifbench_count import PronounsCountIfbenchConstraint
from jfbench.constraints.ifbench_count import PunctuationCountIfbenchConstraint
from jfbench.constraints.ifbench_count import UniqueWordCountIfbenchConstraint
from jfbench.constraints.ifbench_format import EmojiFormatIfbenchConstraint
from jfbench.constraints.ifbench_format import LineIndentFormatIfbenchConstraint
from jfbench.constraints.ifbench_format import NewlineFormatIfbenchConstraint
from jfbench.constraints.ifbench_format import OutputTemplateFormatIfbenchConstraint
from jfbench.constraints.ifbench_format import ParenthesesFormatIfbenchConstraint
from jfbench.constraints.ifbench_format import QuotesFormatIfbenchConstraint
from jfbench.constraints.ifbench_format import QuoteUnquoteFormatIfbenchConstraint
from jfbench.constraints.ifbench_format import SubBulletsFormatIfbenchConstraint
from jfbench.constraints.ifbench_format import ThesisFormatIfbenchConstraint
from jfbench.constraints.ifbench_ratio import OverlapRatioIfbenchConstraint
from jfbench.constraints.ifbench_ratio import SentenceBalanceRatioIfbenchConstraint
from jfbench.constraints.ifbench_ratio import SentenceTypeRatioIfbenchConstraint
from jfbench.constraints.ifbench_ratio import SentenceWordsRatioIfbenchConstraint
from jfbench.constraints.ifbench_ratio import StopWordsRatioIfbenchConstraint
from jfbench.constraints.ifbench_repeat import ChangeRepeatIfbenchConstraint
from jfbench.constraints.ifbench_repeat import SimpleRepeatIfbenchConstraint
from jfbench.constraints.ifbench_sentence import AlliterationIncrementSentenceIfbenchConstraint
from jfbench.constraints.ifbench_sentence import IncrementSentenceIfbenchConstraint
from jfbench.constraints.ifbench_sentence import KeywordSentenceIfbenchConstraint
from jfbench.constraints.ifbench_words import ConsonantsWordsIfbenchConstraint
from jfbench.constraints.ifbench_words import KeywordsSpecificPositionWordsIfbenchConstraint
from jfbench.constraints.ifbench_words import LastFirstWordsIfbenchConstraint
from jfbench.constraints.ifbench_words import NoConsecutiveWordsIfbenchConstraint
from jfbench.constraints.ifbench_words import OddEvenSyllablesWordsIfbenchConstraint
from jfbench.constraints.ifbench_words import PalindromeWordsIfbenchConstraint
from jfbench.constraints.ifbench_words import ParagraphLastFirstWordsIfbenchConstraint
from jfbench.constraints.ifbench_words import PrimeLengthsWordsIfbenchConstraint
from jfbench.constraints.ifbench_words import RepeatsWordsIfbenchConstraint
from jfbench.constraints.ifbench_words import StartVerbWordsIfbenchConstraint
from jfbench.constraints.ifbench_words import VowelWordsIfbenchConstraint
from jfbench.constraints.ifbench_words import WordsPositionWordsIfbenchConstraint
from jfbench.constraints.length import BlankLinesLengthConstraint
from jfbench.constraints.length import CharactersLengthConstraint
from jfbench.constraints.length import NewLinesLengthConstraint
from jfbench.constraints.length import ParagraphsLengthConstraint
from jfbench.constraints.length import SectionsLengthConstraint
from jfbench.constraints.length import SentencesLengthConstraint
from jfbench.constraints.length import WordsLengthConstraint
from jfbench.constraints.logic import DoubleNegationLogicConstraint
from jfbench.constraints.logic import NegationLogicConstraint
from jfbench.constraints.meta_output import ExplanationConstraint
from jfbench.constraints.meta_output import GreetingOutputConstraint
from jfbench.constraints.meta_output import NoExplanationConstraint
from jfbench.constraints.meta_output import NoGreetingOutputConstraint
from jfbench.constraints.meta_output import NoSelfAttestationConstraint
from jfbench.constraints.meta_output import NoSelfReferenceConstraint
from jfbench.constraints.meta_output import SelfAttestationConstraint
from jfbench.constraints.meta_output import SelfReferenceConstraint
from jfbench.constraints.notation import CamelcaseNotationConstraint
from jfbench.constraints.notation import CurrencyNotationConstraint
from jfbench.constraints.notation import DateNotationConstraint
from jfbench.constraints.notation import DecimalPlacesNotationConstraint
from jfbench.constraints.notation import EmailAddressNotationConstraint
from jfbench.constraints.notation import FuriganaNotationConstraint
from jfbench.constraints.notation import GroupingNotationConstraint
from jfbench.constraints.notation import KanjiNumeralsNotationConstraint
from jfbench.constraints.notation import NoHyphenJpPostalCodeNotationConstraint
from jfbench.constraints.notation import NoHyphenPhoneNumberNotationConstraint
from jfbench.constraints.notation import NoKanjiNumeralsNotationConstraint
from jfbench.constraints.notation import RoundingNotationConstraint
from jfbench.constraints.notation import SnakecaseNotationConstraint
from jfbench.constraints.notation import TimeNotationConstraint
from jfbench.constraints.notation import TitlecaseNotationConstraint
from jfbench.constraints.notation import UnitNotationConstraint
from jfbench.constraints.notation import WithHyphenJpPostalCodeNotationConstraint
from jfbench.constraints.notation import WithHyphenPhoneNumberNotationConstraint
from jfbench.constraints.notation import ZeroPaddingNotationConstraint
from jfbench.constraints.processing import ConcatProcessingConstraint
from jfbench.constraints.processing import DictionarySortProcessingConstraint
from jfbench.constraints.processing import ExtractionProcessingConstraint
from jfbench.constraints.processing import LengthSortProcessingConstraint
from jfbench.constraints.processing import NumberSortProcessingConstraint
from jfbench.constraints.processing import PrefixExtractionProcessingConstraint
from jfbench.constraints.processing import PrefixProcessingConstraint
from jfbench.constraints.processing import RangeExtractionProcessingConstraint
from jfbench.constraints.processing import ReplacementProcessingConstraint
from jfbench.constraints.processing import SplitProcessingConstraint
from jfbench.constraints.processing import StatisticsProcessingConstraint
from jfbench.constraints.processing import SuffixExtractionProcessingConstraint
from jfbench.constraints.processing import SuffixProcessingConstraint
from jfbench.constraints.structure import HeadingStructureConstraint
from jfbench.constraints.structure import SectionStructureConstraint
from jfbench.constraints.style import AcademicToneStyleConstraint
from jfbench.constraints.style import AmericanEnglishStyleConstraint
from jfbench.constraints.style import AngryEmotionalStyleConstraint
from jfbench.constraints.style import BritishEnglishStyleConstraint
from jfbench.constraints.style import BusinessToneStyleConstraint
from jfbench.constraints.style import CasualToneStyleConstraint
from jfbench.constraints.style import DifficultVocacularyStyleConstraint
from jfbench.constraints.style import EasyVocabularyStyleConstraint
from jfbench.constraints.style import EnglishStyleConstraint
from jfbench.constraints.style import FirstPersonPluralStyleConstraint
from jfbench.constraints.style import FirstPersonSingularStyleConstraint
from jfbench.constraints.style import FormalToneStyleConstraint
from jfbench.constraints.style import HappyEmotionalStyleConstraint
from jfbench.constraints.style import ImpersonalStyleConstraint
from jfbench.constraints.style import JapaneseStyleConstraint
from jfbench.constraints.style import JoyfulEmotionalStyleConstraint
from jfbench.constraints.style import NoEnglishStyleConstraint
from jfbench.constraints.style import NoJapaneseStyleConstraint
from jfbench.constraints.style import NoTypoStyleConstraint
from jfbench.constraints.style import PastTenseStyleConstraint
from jfbench.constraints.style import PlainStyleConstraint
from jfbench.constraints.style import PoliteStyleConstraint
from jfbench.constraints.style import PresentTenseStyleConstraint
from jfbench.constraints.style import SadEmotionalStyleConstraint
from jfbench.constraints.style import TaigendomeStyleConstraint
from jfbench.llm import extract_reasoning_content
from jfbench.llm import LLMClient
from jfbench.prompts.ifbench import get_all_ifbench_prompts
from jfbench.protocol import Constraint
from jfbench.protocol import ConstraintEvaluation
from jfbench.protocol import Prompt


logger = logging.getLogger(__name__)


def _stable_seed(*parts: str) -> int:
    payload = "||".join(parts)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


@dataclass(frozen=True, eq=False)
class RuleConstraintFactory:
    constructor: Callable[..., Constraint]
    uses_document: bool
    kwargs: dict[str, Any]
    accepts_seed: bool
    accepts_document: bool

    def __call__(
        self,
        *,
        seed: int | None = None,
        document: str | None = None,
    ) -> Constraint:
        params = dict(self.kwargs)
        if self.uses_document and self.accepts_document:
            params["document"] = document if document is not None else ""
        if self.accepts_seed:
            params["seed"] = seed
        return self.constructor(**params)


@dataclass(frozen=True, eq=False)
class LLMConstraintFactory:
    constructor: Callable[..., Constraint]
    uses_document: bool
    kwargs: dict[str, Any]
    accepts_seed: bool
    accepts_document: bool

    def __call__(
        self,
        client: LLMClient,
        document: str,
        seed: int | None = None,
    ) -> Constraint:
        params = dict(self.kwargs)
        if self.uses_document and self.accepts_document:
            params["document"] = document
        if self.accepts_seed:
            params["seed"] = seed
        return self.constructor(client=client, **params)


ConstraintFactory = Callable[..., Constraint]
ConstraintSetName = Literal["train", "test"]
InstructionMode = Literal["train", "test"]


@dataclass(frozen=True)
class ConstraintCollections:
    character: list[ConstraintFactory]
    rule_content: list[ConstraintFactory]
    llm_content: list[ConstraintFactory]
    format: list[ConstraintFactory]
    length: list[ConstraintFactory]
    logic: list[ConstraintFactory]
    meta_output: list[ConstraintFactory]
    notation: list[ConstraintFactory]
    rule_processing: list[ConstraintFactory]
    llm_processing: list[ConstraintFactory]
    structure: list[ConstraintFactory]
    rule_style: list[ConstraintFactory]
    llm_style: list[ConstraintFactory]
    ifbench_count: list[ConstraintFactory]
    ifbench_format: list[ConstraintFactory]
    ifbench_ratio: list[ConstraintFactory]
    ifbench_repeat: list[ConstraintFactory]
    ifbench_sentence: list[ConstraintFactory]
    ifbench_words: list[ConstraintFactory]

    @property
    def rule_based(self) -> list[ConstraintFactory]:
        return (
            self.character
            + self.rule_content
            + self.format
            + self.length
            + self.logic
            + self.notation
            + self.rule_processing
            + self.rule_style
            + self.ifbench_count
            + self.ifbench_format
            + self.ifbench_ratio
            + self.ifbench_repeat
            + self.ifbench_sentence
            + self.ifbench_words
        )

    @property
    def llm_based(self) -> list[ConstraintFactory]:
        return (
            self.llm_content
            + self.meta_output
            + self.llm_processing
            + self.structure
            + self.llm_style
        )


def _instruction_mode(constraint_set: ConstraintSetName) -> InstructionMode:
    return "test" if constraint_set == "test" else "train"


def _rule_constraint_factory(
    constructor: Callable[..., Constraint],
    *,
    uses_document: bool = False,
    **kwargs: Any,
) -> RuleConstraintFactory:
    signature = inspect.signature(constructor)
    return RuleConstraintFactory(
        constructor=constructor,
        uses_document=uses_document,
        kwargs=dict(kwargs),
        accepts_seed="seed" in signature.parameters,
        accepts_document="document" in signature.parameters,
    )


def _llm_constraint_factory(
    constructor: Callable[..., Constraint],
    *,
    uses_document: bool = False,
    **kwargs: Any,
) -> LLMConstraintFactory:
    signature = inspect.signature(constructor)
    return LLMConstraintFactory(
        constructor=constructor,
        uses_document=uses_document,
        kwargs=dict(kwargs),
        accepts_seed="seed" in signature.parameters,
        accepts_document="document" in signature.parameters,
    )


TRAINING_CHARACTER_CONSTRAINTS: list[ConstraintFactory] = [
    _rule_constraint_factory(JapanesePunctuationConstraint),
    _rule_constraint_factory(NoCommasConstraint),
    _rule_constraint_factory(NoJapanesePunctuationConstraint),
    _rule_constraint_factory(NoSuffixWhitespaceConstraint),
    _rule_constraint_factory(NoWhitespaceConstraint),
]
TRAINING_CHARACTER_CONSTRAINTS += [
    _rule_constraint_factory(FullWidthCharacterConstraint),
    _rule_constraint_factory(HalfWidthCharacterConstraint),
    _rule_constraint_factory(HiraganaCharacterConstraint),
    _rule_constraint_factory(KanjiCharacterConstraint),
    _rule_constraint_factory(KatakanaCharacterConstraint),
    _rule_constraint_factory(LowercaseCharacterConstraint),
    _rule_constraint_factory(RomajiCharacterConstraint),
    _rule_constraint_factory(UppercaseCharacterConstraint),
]
TRAINING_RULE_BASED_CONTENT_CONSTRAINTS: list[ConstraintFactory] = [
    _rule_constraint_factory(
        KeywordExcludedContentConstraint,
        keywords=["月", "火", "水", "木", "金", "土", "日"],
    ),
    _rule_constraint_factory(
        KeywordIncludedContentConstraint,
        keywords={"日本": 1, "東京": 10},
    ),
]
TRAINING_LLM_CONTENT_CONSTRAINTS: list[ConstraintFactory] = [
    _llm_constraint_factory(
        AbstractExcludedContentConstraint,
        uses_document=True,
        content="日本に関する説明",
    ),
    _llm_constraint_factory(
        AbstractIncludedContentConstraint,
        uses_document=True,
        content="日本に関する説明",
    ),
    _llm_constraint_factory(IntrinsicContentConstraint, uses_document=True),
    _llm_constraint_factory(ReasonContentConstraint, uses_document=True),
    _llm_constraint_factory(IrrevantContentConstraint, uses_document=True),
    _llm_constraint_factory(NoReasonContentConstraint),
]
TRAINING_FORMAT_CONSTRAINTS: list[ConstraintFactory] = [
    _rule_constraint_factory(BulletPointsFormatConstraint, starter_token="*_*"),
    _rule_constraint_factory(CitationFormatConstraint),
    _rule_constraint_factory(CsvFormatConstraint),
    _rule_constraint_factory(HtmlFormatConstraint),
    _rule_constraint_factory(HtmlTableFormatConstraint),
    _rule_constraint_factory(JavascriptFormatConstraint),
    _rule_constraint_factory(JsonFormatConstraint),
    _rule_constraint_factory(LatexFormatConstraint),
    _rule_constraint_factory(LatexTableFormatConstraint),
    _rule_constraint_factory(MarkdownClosedFencesConstraint),
    _rule_constraint_factory(MarkdownFormatConstraint),
    _rule_constraint_factory(MarkdownHeadingJumpsConstraint),
    _rule_constraint_factory(MarkdownHeadingsStructureConstraint),
    _rule_constraint_factory(MarkdownLinksAndImagesConstraint),
    _rule_constraint_factory(MarkdownListStructureConstraint),
    _rule_constraint_factory(MarkdownParseableConstraint),
    _rule_constraint_factory(MarkdownReferenceLinksConstraint),
    _rule_constraint_factory(MarkdownTableFormatConstraint),
    _rule_constraint_factory(MarkdownTableStructureConstraint),
    _rule_constraint_factory(MarkdownUnconsumedEmphasisMarkersConstraint),
    _rule_constraint_factory(MediawikiTableFormatConstraint),
    _rule_constraint_factory(PythonFormatConstraint),
    _rule_constraint_factory(TsvFormatConstraint),
    _rule_constraint_factory(XmlFormatConstraint),
    _rule_constraint_factory(YamlFormatConstraint),
]
TRAINING_FORMAT_CONSTRAINTS += [
    _rule_constraint_factory(DiffFormatConstraint),
    _rule_constraint_factory(IndentFormatConstraint, indent="    "),
    _rule_constraint_factory(NoCodeFenceJavascriptFormatConstraint),
    _rule_constraint_factory(WithCodeFenceJavascriptFormatConstraint),
    _rule_constraint_factory(NoCodeFencePythonFormatConstraint),
    _rule_constraint_factory(WithCodeFencePythonFormatConstraint),
    _rule_constraint_factory(SentenceDelimiterFormatConstraint, delimiter="(これは文末です)"),
]
TRAINING_LENGTH_CONSTRAINTS: list[ConstraintFactory] = [
    _rule_constraint_factory(
        CharactersLengthConstraint,
        min_length=100,
        max_length=200,
    ),
    _rule_constraint_factory(
        ParagraphsLengthConstraint,
        min_paragraphs=7,
        max_paragraphs=10,
    ),
    _rule_constraint_factory(
        SectionsLengthConstraint,
        min_sections=10,
        max_sections=12,
    ),
    _rule_constraint_factory(
        SentencesLengthConstraint,
        min_sentences=3,
        max_sentences=4,
    ),
]
TRAINING_LENGTH_CONSTRAINTS += [
    _rule_constraint_factory(WordsLengthConstraint, minimum=10, maximum=20),
    _rule_constraint_factory(NewLinesLengthConstraint, newlines=3),
    _rule_constraint_factory(BlankLinesLengthConstraint, blank_lines=2),
]
TRAINING_LOGIC_CONSTRAINTS: list[ConstraintFactory] = [
    lambda seed=None, document=None: NegationLogicConstraint(
        positive_constraint=NoWhitespaceConstraint(seed=seed),
        seed=seed,
    ),
    lambda seed=None, document=None: DoubleNegationLogicConstraint(
        positive_constraint=NoCommasConstraint(seed=seed),
        seed=seed,
    ),
    lambda seed=None, document=None: NegationLogicConstraint(
        positive_constraint=BulletPointsFormatConstraint(starter_token="*_*", seed=seed),
        seed=seed,
    ),
    lambda seed=None, document=None: DoubleNegationLogicConstraint(
        positive_constraint=CharactersLengthConstraint(min_length=100, max_length=200, seed=seed),
        seed=seed,
    ),
    lambda seed=None, document=None: NegationLogicConstraint(
        positive_constraint=DecimalPlacesNotationConstraint(required_decimal_places=5, seed=seed),
        seed=seed,
    ),
    lambda seed=None, document=None: DoubleNegationLogicConstraint(
        positive_constraint=PrefixProcessingConstraint(prefix="<これは文頭です>", seed=seed),
        seed=seed,
    ),
    lambda seed=None, document=None: NegationLogicConstraint(
        positive_constraint=WordsLengthConstraint(minimum=10, maximum=20, seed=seed),
        seed=seed,
    ),
    lambda seed=None, document=None: DoubleNegationLogicConstraint(
        positive_constraint=IndentFormatConstraint(indent="    ", seed=seed),
        seed=seed,
    ),
    lambda seed=None, document=None: NegationLogicConstraint(
        positive_constraint=SentenceDelimiterFormatConstraint(delimiter="。", seed=seed),
        seed=seed,
    ),
    lambda seed=None, document=None: DoubleNegationLogicConstraint(
        positive_constraint=RoundingNotationConstraint(digits=2, seed=seed),
        seed=seed,
    ),
]
TRAINING_META_OUTPUT_CONSTRAINTS: list[ConstraintFactory] = [
    _llm_constraint_factory(NoExplanationConstraint),
    _llm_constraint_factory(NoGreetingOutputConstraint),
    _llm_constraint_factory(NoSelfReferenceConstraint),
    _llm_constraint_factory(SelfAttestationConstraint),
    _llm_constraint_factory(ExplanationConstraint),
    _llm_constraint_factory(GreetingOutputConstraint),
    _llm_constraint_factory(NoSelfAttestationConstraint),
    _llm_constraint_factory(SelfReferenceConstraint),
]
TRAINING_NOTATION_CONSTRAINTS: list[ConstraintFactory] = [
    _rule_constraint_factory(DecimalPlacesNotationConstraint, required_decimal_places=5),
    _rule_constraint_factory(EmailAddressNotationConstraint),
    _rule_constraint_factory(GroupingNotationConstraint, max_group_size=3),
    _rule_constraint_factory(KanjiNumeralsNotationConstraint),
    _rule_constraint_factory(NoKanjiNumeralsNotationConstraint),
    _rule_constraint_factory(NoHyphenJpPostalCodeNotationConstraint),
    _rule_constraint_factory(NoHyphenPhoneNumberNotationConstraint),
    _rule_constraint_factory(WithHyphenJpPostalCodeNotationConstraint),
    _rule_constraint_factory(WithHyphenPhoneNumberNotationConstraint),
    _rule_constraint_factory(CamelcaseNotationConstraint),
    _rule_constraint_factory(CurrencyNotationConstraint),
    _rule_constraint_factory(DateNotationConstraint),
    _rule_constraint_factory(FuriganaNotationConstraint),
    _rule_constraint_factory(RoundingNotationConstraint, digits=2),
    _rule_constraint_factory(SnakecaseNotationConstraint),
    _rule_constraint_factory(TimeNotationConstraint),
    _rule_constraint_factory(TitlecaseNotationConstraint),
    _rule_constraint_factory(UnitNotationConstraint),
    _rule_constraint_factory(ZeroPaddingNotationConstraint, width=5),
]
TRAINING_RULE_BASED_PROCESSING_CONSTRAINTS: list[ConstraintFactory] = [
    _rule_constraint_factory(PrefixProcessingConstraint, prefix="<これは文頭です>"),
    _rule_constraint_factory(SuffixProcessingConstraint, suffix="<これは文末です>"),
    _rule_constraint_factory(ConcatProcessingConstraint, uses_document=True, times=2),
    _rule_constraint_factory(DictionarySortProcessingConstraint, uses_document=True),
    _rule_constraint_factory(LengthSortProcessingConstraint, uses_document=True),
    _rule_constraint_factory(NumberSortProcessingConstraint, uses_document=True),
    _rule_constraint_factory(
        ReplacementProcessingConstraint,
        uses_document=True,
        start=1,
        end=3,
        keyword="###",
    ),
    _rule_constraint_factory(SplitProcessingConstraint, uses_document=True, parts=3),
    _rule_constraint_factory(PrefixExtractionProcessingConstraint, uses_document=True, length=3),
    _rule_constraint_factory(
        RangeExtractionProcessingConstraint, uses_document=True, start=1, end=4
    ),
    _rule_constraint_factory(SuffixExtractionProcessingConstraint, uses_document=True, length=4),
]
TRAINING_LLM_PROCESSING_CONSTRAINTS: list[ConstraintFactory] = [
    _llm_constraint_factory(
        ExtractionProcessingConstraint,
        uses_document=True,
        condition="挨拶の言葉",
    ),
    _llm_constraint_factory(
        StatisticsProcessingConstraint,
        uses_document=True,
        statistic="含まれる固有名詞の数",
    ),
]
TRAINING_STRUCTURE_CONSTRAINTS: list[ConstraintFactory] = [
    _llm_constraint_factory(HeadingStructureConstraint),
    _llm_constraint_factory(SectionStructureConstraint),
]
TRAINING_RULE_BASED_STYLE_CONSTRAINTS: list[ConstraintFactory] = []
TRAINING_LLM_STYLE_CONSTRAINTS: list[ConstraintFactory] = [
    _llm_constraint_factory(EnglishStyleConstraint),
    _llm_constraint_factory(JapaneseStyleConstraint),
    _llm_constraint_factory(BritishEnglishStyleConstraint),
    _llm_constraint_factory(AmericanEnglishStyleConstraint),
    _llm_constraint_factory(PoliteStyleConstraint),
    _llm_constraint_factory(PastTenseStyleConstraint),
    _llm_constraint_factory(PresentTenseStyleConstraint),
    _llm_constraint_factory(FormalToneStyleConstraint),
    _llm_constraint_factory(CasualToneStyleConstraint),
    _llm_constraint_factory(BusinessToneStyleConstraint),
    _llm_constraint_factory(AcademicToneStyleConstraint),
    _llm_constraint_factory(NoTypoStyleConstraint),
    _llm_constraint_factory(AngryEmotionalStyleConstraint),
    _llm_constraint_factory(DifficultVocacularyStyleConstraint),
    _llm_constraint_factory(EasyVocabularyStyleConstraint),
    _llm_constraint_factory(FirstPersonSingularStyleConstraint),
    _llm_constraint_factory(FirstPersonPluralStyleConstraint),
    _llm_constraint_factory(HappyEmotionalStyleConstraint),
    _llm_constraint_factory(ImpersonalStyleConstraint),
    _llm_constraint_factory(JoyfulEmotionalStyleConstraint),
    _llm_constraint_factory(NoEnglishStyleConstraint),
    _llm_constraint_factory(NoJapaneseStyleConstraint),
    _llm_constraint_factory(PlainStyleConstraint),
    _llm_constraint_factory(SadEmotionalStyleConstraint),
    _llm_constraint_factory(TaigendomeStyleConstraint),
]
TRAINING_IFBENCH_COUNT_CONSTRAINTS: list[ConstraintFactory] = []
TRAINING_IFBENCH_FORMAT_CONSTRAINTS: list[ConstraintFactory] = []
TRAINING_IFBENCH_RATIO_CONSTRAINTS: list[ConstraintFactory] = []
TRAINING_IFBENCH_REPEAT_CONSTRAINTS: list[ConstraintFactory] = []
TRAINING_IFBENCH_SENTENCE_CONSTRAINTS: list[ConstraintFactory] = []
TRAINING_IFBENCH_WORDS_CONSTRAINTS: list[ConstraintFactory] = []

TEST_CHARACTER_CONSTRAINTS: list[ConstraintFactory] = list(TRAINING_CHARACTER_CONSTRAINTS)
TEST_RULE_BASED_CONTENT_CONSTRAINTS: list[ConstraintFactory] = list(
    TRAINING_RULE_BASED_CONTENT_CONSTRAINTS
)
TEST_LLM_CONTENT_CONSTRAINTS: list[ConstraintFactory] = list(TRAINING_LLM_CONTENT_CONSTRAINTS)
TEST_FORMAT_CONSTRAINTS: list[ConstraintFactory] = list(TRAINING_FORMAT_CONSTRAINTS)
TEST_LENGTH_CONSTRAINTS: list[ConstraintFactory] = list(TRAINING_LENGTH_CONSTRAINTS)
TEST_LOGIC_CONSTRAINTS: list[ConstraintFactory] = list(TRAINING_LOGIC_CONSTRAINTS)
TEST_META_OUTPUT_CONSTRAINTS: list[ConstraintFactory] = list(TRAINING_META_OUTPUT_CONSTRAINTS)
TEST_NOTATION_CONSTRAINTS: list[ConstraintFactory] = list(TRAINING_NOTATION_CONSTRAINTS)
TEST_RULE_BASED_PROCESSING_CONSTRAINTS: list[ConstraintFactory] = list(
    TRAINING_RULE_BASED_PROCESSING_CONSTRAINTS
)
TEST_LLM_PROCESSING_CONSTRAINTS: list[ConstraintFactory] = list(
    TRAINING_LLM_PROCESSING_CONSTRAINTS
)
TEST_STRUCTURE_CONSTRAINTS: list[ConstraintFactory] = list(TRAINING_STRUCTURE_CONSTRAINTS)
TEST_RULE_BASED_STYLE_CONSTRAINTS: list[ConstraintFactory] = list(
    TRAINING_RULE_BASED_STYLE_CONSTRAINTS
)
TEST_LLM_STYLE_CONSTRAINTS: list[ConstraintFactory] = list(TRAINING_LLM_STYLE_CONSTRAINTS)
TEST_IFBENCH_COUNT_CONSTRAINTS: list[ConstraintFactory] = [
    *TRAINING_IFBENCH_COUNT_CONSTRAINTS,
    _rule_constraint_factory(ConjunctionCountIfbenchConstraint, minimum_kinds=4),
    _rule_constraint_factory(NumbersCountIfbenchConstraint, expected_count=13),
    _rule_constraint_factory(PersonNamesCountIfbenchConstraint, minimum_names=10),
    _rule_constraint_factory(PronounsCountIfbenchConstraint, minimum_pronouns=22),
    _rule_constraint_factory(PunctuationCountIfbenchConstraint),
    _rule_constraint_factory(UniqueWordCountIfbenchConstraint, minimum_unique=360),
]
TEST_IFBENCH_FORMAT_CONSTRAINTS: list[ConstraintFactory] = [
    *TRAINING_IFBENCH_FORMAT_CONSTRAINTS,
    _rule_constraint_factory(EmojiFormatIfbenchConstraint),
    _rule_constraint_factory(LineIndentFormatIfbenchConstraint),
    _rule_constraint_factory(NewlineFormatIfbenchConstraint),
    _rule_constraint_factory(OutputTemplateFormatIfbenchConstraint),
    _rule_constraint_factory(ParenthesesFormatIfbenchConstraint),
    _rule_constraint_factory(QuoteUnquoteFormatIfbenchConstraint),
    _rule_constraint_factory(QuotesFormatIfbenchConstraint),
    _rule_constraint_factory(SubBulletsFormatIfbenchConstraint),
    _rule_constraint_factory(ThesisFormatIfbenchConstraint),
]
TEST_IFBENCH_RATIO_CONSTRAINTS: list[ConstraintFactory] = [
    *TRAINING_IFBENCH_RATIO_CONSTRAINTS,
    _rule_constraint_factory(
        OverlapRatioIfbenchConstraint, uses_document=True, target_ratio_percent=72
    ),
    _rule_constraint_factory(SentenceBalanceRatioIfbenchConstraint),
    _rule_constraint_factory(SentenceTypeRatioIfbenchConstraint),
    _rule_constraint_factory(SentenceWordsRatioIfbenchConstraint),
    _rule_constraint_factory(StopWordsRatioIfbenchConstraint, max_ratio_percent=27),
]
TEST_IFBENCH_REPEAT_CONSTRAINTS: list[ConstraintFactory] = [
    *TRAINING_IFBENCH_REPEAT_CONSTRAINTS,
    _rule_constraint_factory(ChangeRepeatIfbenchConstraint, uses_document=True),
    _rule_constraint_factory(SimpleRepeatIfbenchConstraint),
]
TEST_IFBENCH_SENTENCE_CONSTRAINTS: list[ConstraintFactory] = [
    *TRAINING_IFBENCH_SENTENCE_CONSTRAINTS,
    _rule_constraint_factory(AlliterationIncrementSentenceIfbenchConstraint),
    _rule_constraint_factory(IncrementSentenceIfbenchConstraint, increment=3),
    _rule_constraint_factory(KeywordSentenceIfbenchConstraint, sentence_index=7, keyword="AI"),
]
TEST_IFBENCH_WORDS_CONSTRAINTS: list[ConstraintFactory] = [
    *TRAINING_IFBENCH_WORDS_CONSTRAINTS,
    _rule_constraint_factory(ConsonantsWordsIfbenchConstraint),
    _rule_constraint_factory(
        KeywordsSpecificPositionWordsIfbenchConstraint,
        sentence_index=22,
        word_index=33,
        keyword="AI",
    ),
    _rule_constraint_factory(LastFirstWordsIfbenchConstraint),
    _rule_constraint_factory(NoConsecutiveWordsIfbenchConstraint),
    _rule_constraint_factory(OddEvenSyllablesWordsIfbenchConstraint),
    _rule_constraint_factory(PalindromeWordsIfbenchConstraint, minimum_palindromes=10),
    _rule_constraint_factory(ParagraphLastFirstWordsIfbenchConstraint),
    _rule_constraint_factory(PrimeLengthsWordsIfbenchConstraint),
    _rule_constraint_factory(RepeatsWordsIfbenchConstraint, max_repeats=4),
    _rule_constraint_factory(StartVerbWordsIfbenchConstraint),
    _rule_constraint_factory(VowelWordsIfbenchConstraint),
    _rule_constraint_factory(
        WordsPositionWordsIfbenchConstraint,
        word_index=2,
        from_end_index=2,
        word="テスト",
    ),
]


def get_constraint_collections(constraint_set: ConstraintSetName) -> ConstraintCollections:
    try:
        return CONSTRAINT_COLLECTIONS[constraint_set]
    except KeyError as exc:
        raise ValueError(f"Unsupported constraint set: {constraint_set}") from exc


TRAINING_CONSTRAINT_COLLECTIONS = ConstraintCollections(
    character=TRAINING_CHARACTER_CONSTRAINTS,
    rule_content=TRAINING_RULE_BASED_CONTENT_CONSTRAINTS,
    llm_content=TRAINING_LLM_CONTENT_CONSTRAINTS,
    format=TRAINING_FORMAT_CONSTRAINTS,
    length=TRAINING_LENGTH_CONSTRAINTS,
    logic=TRAINING_LOGIC_CONSTRAINTS,
    meta_output=TRAINING_META_OUTPUT_CONSTRAINTS,
    notation=TRAINING_NOTATION_CONSTRAINTS,
    rule_processing=TRAINING_RULE_BASED_PROCESSING_CONSTRAINTS,
    llm_processing=TRAINING_LLM_PROCESSING_CONSTRAINTS,
    structure=TRAINING_STRUCTURE_CONSTRAINTS,
    rule_style=TRAINING_RULE_BASED_STYLE_CONSTRAINTS,
    llm_style=TRAINING_LLM_STYLE_CONSTRAINTS,
    ifbench_count=TRAINING_IFBENCH_COUNT_CONSTRAINTS,
    ifbench_format=TRAINING_IFBENCH_FORMAT_CONSTRAINTS,
    ifbench_ratio=TRAINING_IFBENCH_RATIO_CONSTRAINTS,
    ifbench_repeat=TRAINING_IFBENCH_REPEAT_CONSTRAINTS,
    ifbench_sentence=TRAINING_IFBENCH_SENTENCE_CONSTRAINTS,
    ifbench_words=TRAINING_IFBENCH_WORDS_CONSTRAINTS,
)
TEST_CONSTRAINT_COLLECTIONS = ConstraintCollections(
    character=TEST_CHARACTER_CONSTRAINTS,
    rule_content=TEST_RULE_BASED_CONTENT_CONSTRAINTS,
    llm_content=TEST_LLM_CONTENT_CONSTRAINTS,
    format=TEST_FORMAT_CONSTRAINTS,
    length=TEST_LENGTH_CONSTRAINTS,
    logic=TEST_LOGIC_CONSTRAINTS,
    meta_output=TEST_META_OUTPUT_CONSTRAINTS,
    notation=TEST_NOTATION_CONSTRAINTS,
    rule_processing=TEST_RULE_BASED_PROCESSING_CONSTRAINTS,
    llm_processing=TEST_LLM_PROCESSING_CONSTRAINTS,
    structure=TEST_STRUCTURE_CONSTRAINTS,
    rule_style=TEST_RULE_BASED_STYLE_CONSTRAINTS,
    llm_style=TEST_LLM_STYLE_CONSTRAINTS,
    ifbench_count=TEST_IFBENCH_COUNT_CONSTRAINTS,
    ifbench_format=TEST_IFBENCH_FORMAT_CONSTRAINTS,
    ifbench_ratio=TEST_IFBENCH_RATIO_CONSTRAINTS,
    ifbench_repeat=TEST_IFBENCH_REPEAT_CONSTRAINTS,
    ifbench_sentence=TEST_IFBENCH_SENTENCE_CONSTRAINTS,
    ifbench_words=TEST_IFBENCH_WORDS_CONSTRAINTS,
)
CONSTRAINT_COLLECTIONS: dict[ConstraintSetName, ConstraintCollections] = {
    "train": TRAINING_CONSTRAINT_COLLECTIONS,
    "test": TEST_CONSTRAINT_COLLECTIONS,
}


@dataclass
class MetaData:
    prompt_source: str
    data_id: str
    n_constraints: int
    constraint_types: list[str]
    constraint_groups: list[str]
    constraint_instructions: list[str]
    prompt: str
    constraint_set: ConstraintSetName = "train"


@dataclass
class RewriteTrace:
    value: str
    evaluation: dict[str, bool]
    previous_value: str | None
    previous_evaluation: dict[str, bool] | None
    steps: int


class BenchmarkData:
    def __init__(
        self,
        prompt: Prompt,
        constraints: list[Constraint],
        meta_data: MetaData,
    ) -> None:
        self.prompt = prompt
        self.constraints = constraints
        self.meta_data = meta_data
        self.max_rewrite_attempts = len(constraints) + 1

    @staticmethod
    def build_meta_data(
        *,
        prompt_source: str,
        data_id: str,
        prompt: Prompt,
        constraints: list[Constraint],
        constraint_set: ConstraintSetName,
    ) -> MetaData:
        instruction_mode = _instruction_mode(constraint_set)
        return MetaData(
            prompt_source=prompt_source,
            data_id=data_id,
            n_constraints=len(constraints),
            constraint_types=[c.__class__.__name__ for c in constraints],
            constraint_groups=[c.group for c in constraints],
            constraint_instructions=[
                c.instructions(train_or_test=instruction_mode) for c in constraints
            ],
            prompt=prompt.text(constraints, train_or_test=instruction_mode),
            constraint_set=constraint_set,
        )

    async def evaluate(self, value: str) -> dict[str, bool]:
        _, summary = await self._evaluate_constraints_with_details(value)
        return summary

    async def score(self, value: str) -> float:
        results = await self.evaluate(value)
        if all(results.values()):
            return 1.0
        else:
            return 0.0

    def text(self) -> str:
        return self.meta_data.prompt

    async def rewrite(
        self,
        value: str,
        client: LLMClient,
        *,
        reasoning_history: list[str] | None = None,
    ) -> tuple[str, dict[str, bool]]:
        trace = await self.rewrite_with_trace(
            value,
            client,
            reasoning_history=reasoning_history,
        )
        return trace.value, trace.evaluation

    async def rewrite_with_trace(
        self,
        value: str,
        client: LLMClient,
        *,
        reasoning_history: list[str] | None = None,
    ) -> RewriteTrace:
        attempts: dict[int, int] = {id(constraint): 0 for constraint in self.constraints}
        current_value = value
        evaluations, summary = await self._evaluate_constraints_with_details(current_value)
        if self._all_passed(evaluations):
            return RewriteTrace(
                value=current_value,
                evaluation=summary,
                previous_value=None,
                previous_evaluation=None,
                steps=0,
            )
        previous_value: str | None = None
        previous_evaluation: dict[str, bool] | None = None
        steps = 0

        while True:
            failing_constraints: list[Constraint] = []
            failure_reasons: dict[Constraint, str | None] = {}
            for constraint, (passed, reason) in zip(self.constraints, evaluations, strict=True):
                if passed:
                    continue
                failing_constraints.append(constraint)
                failure_reasons[constraint] = reason
                key = id(constraint)
                count = attempts[key]
                if count >= self.max_rewrite_attempts:
                    raise RuntimeError(
                        f"Failed to satisfy {constraint.__class__.__name__} after "
                        f"{self.max_rewrite_attempts} rewrites."
                    )
                attempts[key] = count + 1

            if not failing_constraints:
                break

            previous_value = current_value
            previous_evaluation = dict(summary)
            current_value = await self._rewrite_once(
                current_value,
                self.constraints,
                client,
                failure_reasons,
                reasoning_history=reasoning_history,
            )
            steps += 1
            evaluations, summary = await self._evaluate_constraints_with_details(current_value)
            if self._all_passed(evaluations):
                return RewriteTrace(
                    value=current_value,
                    evaluation=summary,
                    previous_value=previous_value,
                    previous_evaluation=previous_evaluation,
                    steps=steps,
                )

        raise RuntimeError(
            f"Failed to rewrite output for BenchmarkData ID {self.meta_data.data_id}."
        )

    async def _evaluate_constraints_with_details(
        self,
        value: str,
    ) -> tuple[list[ConstraintEvaluation], dict[str, bool]]:
        per_constraint: list[ConstraintEvaluation] = []
        summary: dict[str, bool] = {}
        for constraint in self.constraints:
            result = await self._evaluate_single_constraint(constraint, value)
            per_constraint.append(result)
            summary[constraint.__class__.__name__] = result[0]
        return per_constraint, summary

    async def _evaluate_single_constraint(
        self,
        constraint: Constraint,
        value: str,
    ) -> ConstraintEvaluation:
        reason: str | None
        try:
            evaluation = constraint.evaluate(value)
            if inspect.isawaitable(evaluation):
                evaluations: list[Any] = [evaluation]
                for _ in range(2):
                    evaluations.append(constraint.evaluate(value))
                awaitable_indices = [
                    index
                    for index, candidate in enumerate(evaluations)
                    if inspect.isawaitable(candidate)
                ]
                awaitable_values = await asyncio.gather(
                    *[cast("Awaitable[Any]", evaluations[index]) for index in awaitable_indices]
                )
                for index, awaited in zip(awaitable_indices, awaitable_values, strict=True):
                    evaluations[index] = awaited
                evaluation_candidates = [
                    self._coerce_constraint_evaluation(candidate) for candidate in evaluations
                ]
                passed_count = sum(1 for passed, _ in evaluation_candidates if passed)
                failed_count = len(evaluation_candidates) - passed_count
                majority_passed = passed_count > failed_count
                reason = next(
                    (
                        candidate_reason
                        for candidate_passed, candidate_reason in evaluation_candidates
                        if candidate_passed == majority_passed
                    ),
                    None,
                )
                passed_bool = bool(majority_passed)
                normalized_reason = None if reason is None else str(reason)
                if not passed_bool:
                    if normalized_reason is None:
                        normalized_reason = (
                            f"[{constraint.__class__.__name__}] Constraint evaluation failed."
                        )
                    logging.getLogger(constraint.__class__.__module__).info(normalized_reason)
                return passed_bool, normalized_reason
        except Exception as e:
            reason = f"Evaluation fails for ID {self.meta_data.data_id} due to {e}."
            tqdm.write(reason)
            return False, reason
        passed, reason = self._coerce_constraint_evaluation(evaluation)
        passed_bool = bool(passed)
        normalized_reason = None if reason is None else str(reason)
        if not passed_bool:
            if normalized_reason is None:
                normalized_reason = (
                    f"[{constraint.__class__.__name__}] Constraint evaluation failed."
                )
            logging.getLogger(constraint.__class__.__module__).info(normalized_reason)
        return passed_bool, normalized_reason

    @staticmethod
    def _coerce_constraint_evaluation(evaluation: Any) -> ConstraintEvaluation:
        if isinstance(evaluation, tuple) and len(evaluation) == 2:
            passed, reason = evaluation
        else:
            passed = bool(evaluation)
            reason = None
        return bool(passed), None if reason is None else str(reason)

    async def _rewrite_once(
        self,
        value: str,
        constraints: Sequence[Constraint],
        client: LLMClient,
        failure_reasons: Mapping[Constraint, str | None],
        *,
        reasoning_history: list[str] | None = None,
    ) -> str:
        failing_constraints = [c for c in constraints if c in failure_reasons]
        rewrite_sequence: list[tuple[Constraint, Callable[[str], str]]] = []
        requires_llm_rewrite = False
        for constraint in failing_constraints:
            rewrite_method = getattr(constraint, "rewrite_value", None)
            if callable(rewrite_method):
                rewrite_sequence.append((constraint, rewrite_method))
            else:
                requires_llm_rewrite = True

        current_value = value
        if requires_llm_rewrite:
            prompt = self._build_rewrite_prompt(value, constraints, failure_reasons)
            responses, response_details = await client.async_ask([prompt])
            current_value = responses[0].strip()
            reasoning_content = extract_reasoning_content(
                getattr(client, "provider", ""),
                response_details[0] if response_details else None,
            )
            if reasoning_content and reasoning_history is not None:
                reasoning_history.append(reasoning_content)

        for constraint, rewrite_method in rewrite_sequence:
            logger.info(f"Applying rewrite_value for constraint {constraint.__class__.__name__}")
            current_value = rewrite_method(current_value)
        return current_value

    def _build_rewrite_prompt(
        self,
        value: str,
        constraints: Sequence[Constraint],
        failure_reasons: Mapping[Constraint, str | None],
    ) -> str:
        instructions = self.text()
        instruction_mode = _instruction_mode(self.meta_data.constraint_set)
        rewrite_sections: list[str] = []
        for constraint in constraints:
            try:
                rewrite_text = constraint.rewrite_instructions()
            except ValueError:
                rewrite_text = constraint.instructions(train_or_test=instruction_mode)
            reason = failure_reasons.get(constraint)
            entry = f"- [{constraint.__class__.__name__}] {rewrite_text}"
            if reason:
                entry += f"\n  理由: {reason}"
            rewrite_sections.append(entry)
        constraint_rewrite = "\n".join(rewrite_sections)
        return (
            "以下の指示と追加の修正条件を満たすように既存の出力を書き直してください。\n"
            "=== 指示 ===\n"
            f"{instructions}\n"
            "=== 追加の修正条件 ===\n"
            f"{constraint_rewrite}\n"
            "=== 現在の出力 ===\n"
            f"{value}\n"
            "=== 要求 ===\n"
            "修正後の出力のみをそのまま出力してください。"
        )

    @staticmethod
    def _all_passed(results: list[ConstraintEvaluation]) -> bool:
        return all(passed for passed, _ in results)


def get_benchmark_data_with_single_constraint(
    client: LLMClient,
    prompts: Sequence[Prompt],
    prompt_source: str,
    seed: int,
    constraint_set: ConstraintSetName = "train",
) -> list[BenchmarkData]:
    benchmark_data_list = []
    constraint_collections = get_constraint_collections(constraint_set)

    # Add data for all single constraints for each prompt
    for i, prompt in enumerate(prompts):
        for factory_idx, rule_based_constraint_factory in enumerate(
            constraint_collections.rule_based
        ):
            seed_value = _stable_seed(prompt_source, str(i), "rule", str(factory_idx))
            constraint = rule_based_constraint_factory(
                seed=seed_value,
                document=prompt.document,
            )
            constraints_list = [constraint]
            meta_data = BenchmarkData.build_meta_data(
                prompt_source=prompt_source,
                data_id=f"{i}-{constraint.__class__.__name__}",
                prompt=prompt,
                constraints=constraints_list,
                constraint_set=constraint_set,
            )
            benchmark_data = BenchmarkData(
                prompt=prompt,
                constraints=constraints_list,
                meta_data=meta_data,
            )
            benchmark_data_list.append(benchmark_data)
        for factory_idx, llm_based_constraint_factory in enumerate(
            constraint_collections.llm_based
        ):
            seed_value = _stable_seed(prompt_source, str(i), "llm", str(factory_idx))
            constraint = llm_based_constraint_factory(
                client,
                prompt.document,
                seed=seed_value,
            )
            constraints_list = [constraint]
            meta_data = BenchmarkData.build_meta_data(
                prompt_source=prompt_source,
                data_id=f"{i}-{constraint.__class__.__name__}",
                prompt=prompt,
                constraints=constraints_list,
                constraint_set=constraint_set,
            )
            benchmark_data = BenchmarkData(
                prompt=prompt,
                constraints=constraints_list,
                meta_data=meta_data,
            )
            benchmark_data_list.append(benchmark_data)

    np.random.default_rng(seed).shuffle(benchmark_data_list)
    return benchmark_data_list


def get_ifbench_benchmark_data(
    client: LLMClient,
    seed: int,
    constraint_set: ConstraintSetName = "train",
    dataset_path: str | None = None,
) -> list[BenchmarkData]:
    prompts = get_all_ifbench_prompts(dataset_path=dataset_path)
    return get_benchmark_data_with_single_constraint(
        client,
        prompts,
        prompt_source="ifbench",
        seed=seed,
        constraint_set=constraint_set,
    )


def get_benchmark_data_with_multiple_constraints(
    client: LLMClient,
    prompts: Sequence[Prompt],
    n_constraints: int,
    n_benchmark_data: int,
    seed: int,
    prompt_source: str,
    constraint_set: ConstraintSetName = "train",
) -> list[BenchmarkData]:
    assert n_constraints >= 2, "n_constraints must be at least 2."
    benchmark_data_list: list[BenchmarkData] = []
    benchmark_data_identity_set: set[tuple[str, tuple[str, ...]]] = set()
    rng = np.random.default_rng(seed)
    constraint_collections = get_constraint_collections(constraint_set)
    max_sampling_attempts = 1000
    duplicate_attempts = 0

    if not constraint_collections.format:
        raise ValueError(f"No format constraints available for set {constraint_set}.")

    progress = tqdm(
        total=n_benchmark_data,
        desc="Building benchmark data",
        leave=False,
    )
    try:
        while len(benchmark_data_list) < n_benchmark_data:
            # Sample a prompt uniformly
            prompt = rng.choice(prompts)
            constraints: list[Constraint] = []
            sampling_attempts = 0

            def _record_failure() -> None:
                nonlocal sampling_attempts
                sampling_attempts += 1
                if sampling_attempts >= max_sampling_attempts:
                    raise ValueError(
                        f"Unable to sample non-conflicting constraints after "
                        f"{max_sampling_attempts} attempts."
                    )

            def _conflicts_with_existing(candidate: Constraint) -> bool:
                candidate_name = candidate.__class__.__name__
                candidate_competitives = set(candidate.competitives or [])
                for existing_constraint in constraints:
                    existing_name = existing_constraint.__class__.__name__
                    existing_competitives = set(existing_constraint.competitives or [])
                    if (
                        candidate.__class__ == existing_constraint.__class__
                        or existing_name in candidate_competitives
                        or candidate_name in existing_competitives
                    ):
                        return True
                return False

            def _build_constraint(
                factory: ConstraintFactory,
            ) -> Constraint:
                if factory in constraint_collections.rule_based:
                    return factory(
                        seed=int(rng.integers(0, 2**63 - 1)),
                        document=prompt.document,
                    )
                return factory(
                    client,
                    prompt.document,
                    seed=int(rng.integers(0, 2**63 - 1)),
                )

            def _try_add_constraint(factory: ConstraintFactory) -> bool:
                nonlocal sampling_attempts
                candidate = _build_constraint(factory)
                if _conflicts_with_existing(candidate):
                    _record_failure()
                    return False
                constraints.append(candidate)
                sampling_attempts = 0
                return True

            format_constraint_factory = rng.choice(constraint_collections.format)
            _try_add_constraint(format_constraint_factory)

            group_to_factories: dict[str, list[ConstraintFactory]] = {
                "character": list(constraint_collections.character),
                "content": constraint_collections.rule_content
                + constraint_collections.llm_content,
                "format": list(constraint_collections.format),
                "length": list(constraint_collections.length),
                "logic": list(constraint_collections.logic),
                "meta_output": list(constraint_collections.meta_output),
                "notation": list(constraint_collections.notation),
                "processing": constraint_collections.rule_processing
                + constraint_collections.llm_processing,
                "structure": list(constraint_collections.structure),
                "style": constraint_collections.rule_style + constraint_collections.llm_style,
                "ifbench_count": list(constraint_collections.ifbench_count),
                "ifbench_format": list(constraint_collections.ifbench_format),
                "ifbench_ratio": list(constraint_collections.ifbench_ratio),
                "ifbench_repeat": list(constraint_collections.ifbench_repeat),
                "ifbench_sentence": list(constraint_collections.ifbench_sentence),
                "ifbench_words": list(constraint_collections.ifbench_words),
            }
            available_factories = [
                factory for factories in group_to_factories.values() for factory in factories
            ]
            if not available_factories:
                raise ValueError(f"No available constraints for set {constraint_set}.")

            while len(constraints) < n_constraints:
                constraint_factory = rng.choice(available_factories)
                if not _try_add_constraint(constraint_factory):
                    continue
            # Shuffle the order of constraints to avoid ordering bias.
            rng.shuffle(constraints)
            constraint_names = "-".join(
                constraint.__class__.__name__ for constraint in constraints
            )
            constraint_signature = tuple(
                sorted(constraint.__class__.__name__ for constraint in constraints)
            )
            meta_data = BenchmarkData.build_meta_data(
                prompt_source=prompt_source,
                data_id=f"multi-{constraint_names}",
                prompt=prompt,
                constraints=constraints,
                constraint_set=constraint_set,
            )
            identity = (prompt.document, constraint_signature)
            if identity in benchmark_data_identity_set:
                duplicate_attempts += 1
                if duplicate_attempts >= max_sampling_attempts:
                    raise ValueError(
                        "Unable to sample unique benchmark data without prompt duplication."
                    )
                continue
            duplicate_attempts = 0
            benchmark_data = BenchmarkData(
                prompt=prompt,
                constraints=constraints,
                meta_data=meta_data,
            )
            benchmark_data_list.append(benchmark_data)
            benchmark_data_identity_set.add(identity)
            progress.update(1)
    finally:
        progress.close()

    return benchmark_data_list


def get_ifbench_benchmark_data_with_multiple_constraints(
    client: LLMClient,
    n_constraints: int,
    n_benchmark_data: int,
    seed: int,
    constraint_set: ConstraintSetName = "train",
    dataset_path: str | None = None,
) -> list[BenchmarkData]:
    prompts = get_all_ifbench_prompts(dataset_path=dataset_path)
    return get_benchmark_data_with_multiple_constraints(
        client,
        prompts,
        n_constraints,
        n_benchmark_data,
        seed,
        prompt_source="ifbench",
        constraint_set=constraint_set,
    )
