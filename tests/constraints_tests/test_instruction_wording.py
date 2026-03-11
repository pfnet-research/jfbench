from typing import Callable

import pytest

from jfbench.constraints.content.intrinsic import IntrinsicContentConstraint
from jfbench.constraints.content.irrevant import IrrevantContentConstraint
from jfbench.constraints.content.reason import ReasonContentConstraint
from jfbench.constraints.format.html import HtmlFormatConstraint
from jfbench.constraints.format.yaml import YamlFormatConstraint
from jfbench.constraints.ifbench_ratio.overlap import OverlapRatioIfbenchConstraint
from jfbench.constraints.ifbench_repeat.repeat import ChangeRepeatIfbenchConstraint
from jfbench.constraints.processing.concat import ConcatProcessingConstraint
from jfbench.constraints.processing.extraction import ExtractionProcessingConstraint
from jfbench.constraints.processing.extraction import PrefixExtractionProcessingConstraint
from jfbench.constraints.processing.extraction import RangeExtractionProcessingConstraint
from jfbench.constraints.processing.extraction import SuffixExtractionProcessingConstraint
from jfbench.constraints.processing.replacement import ReplacementProcessingConstraint
from jfbench.constraints.processing.sort import DictionarySortProcessingConstraint
from jfbench.constraints.processing.sort import LengthSortProcessingConstraint
from jfbench.constraints.processing.sort import NumberSortProcessingConstraint
from jfbench.constraints.processing.split import SplitProcessingConstraint
from jfbench.constraints.processing.statistics import StatisticsProcessingConstraint
from jfbench.llm import LLMClient
from jfbench.protocol import Constraint


@pytest.mark.parametrize(
    ("builder", "label"),
    [
        (
            lambda client: IntrinsicContentConstraint(client, "sample text", seed=0),
            "intrinsic",
        ),
        (
            lambda client: IrrevantContentConstraint(client, "sample text", seed=0),
            "irrelevant",
        ),
        (
            lambda client: ReasonContentConstraint(client, "sample text", seed=0),
            "reason",
        ),
        (
            lambda client: StatisticsProcessingConstraint(
                client,
                "sample text",
                "total",
                seed=0,
            ),
            "statistics",
        ),
        (
            lambda client: ExtractionProcessingConstraint(
                client,
                "sample text",
                "condition",
                seed=0,
            ),
            "extraction",
        ),
    ],
)
def test_instruction_wording_client_constraints(
    builder: Callable[[LLMClient], Constraint],
    label: str,
    true_client: LLMClient,
) -> None:
    instance = builder(true_client)
    assert "ドキュメント" not in instance.instructions(), label
    assert "ドキュメント" not in instance.rewrite_instructions(), label


def test_instruction_wording_misc_constraints() -> None:
    constraints: list[Constraint] = [
        ChangeRepeatIfbenchConstraint("first second", seed=0),
        OverlapRatioIfbenchConstraint("one two three", 50, seed=0),
        ConcatProcessingConstraint("abc", 2, seed=0),
        SplitProcessingConstraint("abcdef", 2, seed=0),
        ReplacementProcessingConstraint("abcdef", 0, 2, "XYZ", seed=0),
        PrefixExtractionProcessingConstraint("abcdef", 2, seed=0),
        SuffixExtractionProcessingConstraint("abcdef", 2, seed=0),
        RangeExtractionProcessingConstraint("abcdef", 0, 2, seed=0),
        DictionarySortProcessingConstraint("a. b.", seed=0),
        LengthSortProcessingConstraint("a. bb.", seed=0),
        NumberSortProcessingConstraint("1a. 2b.", seed=0),
        HtmlFormatConstraint(),
        YamlFormatConstraint(),
    ]
    for constraint in constraints:
        if constraint.group.startswith("Ifbench"):
            instruction = constraint.instructions(train_or_test="test")
        else:
            instruction = constraint.instructions()
        assert "ドキュメント" not in instruction
        assert "ドキュメント" not in constraint.rewrite_instructions()
