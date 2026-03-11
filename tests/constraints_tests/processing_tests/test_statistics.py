import asyncio

from conftest import StubLLMClient

from jfbench.constraints.processing import StatisticsProcessingConstraint


def test_statistics_constraint_accepts_confirmed_output(
    true_client: StubLLMClient,
) -> None:
    document = "values: 1, 2, 3"
    constraint = StatisticsProcessingConstraint(true_client, document, "average")

    assert asyncio.run(constraint.evaluate("The average is 2"))[0]
    assert true_client.captured_prompt is not None
    assert "average" in true_client.captured_prompt


def test_statistics_constraint_rejects_when_llm_disagrees(
    false_client: StubLLMClient,
) -> None:
    document = "values: 1, 2, 3"
    constraint = StatisticsProcessingConstraint(false_client, document, "average")

    assert not asyncio.run(constraint.evaluate("The average is 99"))[0]


def test_statistics_constraint_accepts_prefixed_true_response(
    true_client: StubLLMClient,
) -> None:
    true_client.reply = "True: statistics verified"
    constraint = StatisticsProcessingConstraint(true_client, "values: 2, 4", "average")
    assert asyncio.run(constraint.evaluate("The average is 3"))[0]


def test_statistics_constraint_rejects_prefixed_false_response(
    false_client: StubLLMClient,
) -> None:
    false_client.reply = "False - statistic incorrect"
    constraint = StatisticsProcessingConstraint(false_client, "values: 2, 4", "average")
    assert not asyncio.run(constraint.evaluate("The average is 99"))[0]
