import asyncio

from conftest import StubLLMClient

from jfbench.constraints.style import NoTypoStyleConstraint


PROMPT_HINT = "ユーザーが与えた文章を除き、アシスタントが出力した部分のみを対象としてください。"


def test_no_typo_constraint_accepts_clean_text(
    true_client: StubLLMClient,
) -> None:
    constraint = NoTypoStyleConstraint(true_client)
    assert asyncio.run(constraint.evaluate("Thank you for your assistance."))[0]
    assert true_client.captured_prompt is not None
    assert "Thank you" in true_client.captured_prompt


def test_no_typo_constraint_rejects_when_llm_detects_issue(
    false_client: StubLLMClient,
) -> None:
    constraint = NoTypoStyleConstraint(false_client)
    assert not asyncio.run(constraint.evaluate("Thak you for your asistnce."))[0]


def test_no_typo_constraint_accepts_prefix_true_response(
    true_client: StubLLMClient,
) -> None:
    true_client.reply = "True: no issues found"
    constraint = NoTypoStyleConstraint(true_client)
    assert asyncio.run(constraint.evaluate("All clear."))[0]


def test_no_typo_constraint_rejects_prefix_false_response(
    false_client: StubLLMClient,
) -> None:
    false_client.reply = "False - typo detected"
    constraint = NoTypoStyleConstraint(false_client)
    assert not asyncio.run(constraint.evaluate("Oops."))[0]


def test_no_typo_prompt_focuses_on_assistant_output_only(
    true_client: StubLLMClient,
) -> None:
    true_client.reply = "True"
    constraint = NoTypoStyleConstraint(true_client)
    asyncio.run(constraint.evaluate("dummy text"))
    assert true_client.captured_prompt is not None
    assert PROMPT_HINT in true_client.captured_prompt
