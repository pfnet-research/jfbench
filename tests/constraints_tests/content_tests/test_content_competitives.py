import itertools

from jfbench.constraints._competitives import COMPETITIVE_CONSTRAINTS


def test_content_constraints_are_competitive_with_each_other() -> None:
    content_constraints = {
        "AbstractExcludedContentConstraint",
        "AbstractIncludedContentConstraint",
        "IntrinsicContentConstraint",
        "IrrevantContentConstraint",
        "KeywordExcludedContentConstraint",
        "KeywordIncludedContentConstraint",
        "NoReasonContentConstraint",
        "ReasonContentConstraint",
    }
    for left, right in itertools.combinations(sorted(content_constraints), 2):
        assert right in COMPETITIVE_CONSTRAINTS[left]
        assert left in COMPETITIVE_CONSTRAINTS[right]
