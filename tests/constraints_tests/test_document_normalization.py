from typing import Any
from typing import cast
from typing import Protocol

from jfbench.constraints.content.excluded import AbstractExcludedContentConstraint
from jfbench.constraints.content.included import AbstractIncludedContentConstraint
from jfbench.constraints.content.intrinsic import IntrinsicContentConstraint
from jfbench.constraints.content.irrevant import IrrevantContentConstraint
from jfbench.constraints.content.reason import ReasonContentConstraint
from jfbench.constraints.ifbench_ratio.overlap import OverlapRatioIfbenchConstraint
from jfbench.constraints.ifbench_repeat.repeat import ChangeRepeatIfbenchConstraint
from jfbench.constraints.processing.concat import ConcatProcessingConstraint
from jfbench.constraints.processing.extraction import ExtractionProcessingConstraint
from jfbench.constraints.processing.extraction import PrefixExtractionProcessingConstraint
from jfbench.constraints.processing.split import SplitProcessingConstraint
from jfbench.constraints.processing.statistics import StatisticsProcessingConstraint


class _DocumentConstraint(Protocol):
    document: str


def test_document_normalization_for_basic_constraints() -> None:
    document = "doc\r\n"
    constraints = [
        ChangeRepeatIfbenchConstraint(document=document, seed=0),
        OverlapRatioIfbenchConstraint(document=document, target_ratio_percent=50, seed=0),
        ConcatProcessingConstraint(document=document, times=2, seed=0),
        SplitProcessingConstraint(document=document, parts=2, seed=0),
        PrefixExtractionProcessingConstraint(document=document, length=2, seed=0),
    ]
    for constraint in constraints:
        constraint_with_doc = cast("_DocumentConstraint", constraint)
        assert constraint_with_doc.document == "doc"


def test_document_normalization_for_llm_constraints(true_client: Any) -> None:
    document = "doc\r\n"
    constraints = [
        IntrinsicContentConstraint(true_client, document=document, seed=0),
        ReasonContentConstraint(true_client, document=document, seed=0),
        IrrevantContentConstraint(true_client, document=document, seed=0),
        AbstractIncludedContentConstraint(true_client, document=document, content="alpha", seed=0),
        AbstractExcludedContentConstraint(true_client, document=document, content="beta", seed=0),
        StatisticsProcessingConstraint(
            true_client, document=document, statistic="average", seed=0
        ),
        ExtractionProcessingConstraint(
            true_client, document=document, condition="contains", seed=0
        ),
    ]
    for constraint in constraints:
        constraint_with_doc = cast("_DocumentConstraint", constraint)
        assert constraint_with_doc.document == "doc"
