import asyncio
import logging

from _pytest.logging import LogCaptureFixture
from conftest import StubLLMClient

from jfbench.constraints.style import PoliteStyleConstraint


PROMPT_HINT = "ユーザーが与えた文章を除き、アシスタントが出力した部分のみを対象としてください。"


def test_polite_style_constraint_accepts_when_llm_confirms(
    true_client: StubLLMClient,
) -> None:
    constraint = PoliteStyleConstraint(true_client)
    assert asyncio.run(constraint.evaluate("本日はお時間をいただきありがとうございます。"))[0]
    assert true_client.captured_prompt is not None
    assert "本日はお時間" in true_client.captured_prompt


def test_polite_style_constraint_rejects_when_llm_denies(
    false_client: StubLLMClient,
) -> None:
    constraint = PoliteStyleConstraint(false_client)
    assert not asyncio.run(constraint.evaluate("これはカジュアルな文です。"))[0]


def test_polite_style_constraint_accepts_prefixed_true_response(
    true_client: StubLLMClient,
) -> None:
    true_client.reply = "True: polite tone detected"
    constraint = PoliteStyleConstraint(true_client)
    assert asyncio.run(constraint.evaluate("何卒よろしくお願い申し上げます。"))[0]


def test_polite_style_constraint_rejects_prefixed_false_response(
    false_client: StubLLMClient,
) -> None:
    false_client.reply = "False - impolite language"
    constraint = PoliteStyleConstraint(false_client)
    assert not asyncio.run(constraint.evaluate("よろしく！"))[0]


def test_polite_style_constraint_logs_llm_response_on_failure(
    false_client: StubLLMClient,
    caplog: LogCaptureFixture,
) -> None:
    false_client.reply = "False - still too casual"
    constraint = PoliteStyleConstraint(false_client)
    with caplog.at_level(logging.INFO):
        assert not asyncio.run(constraint.evaluate("じゃあね。"))[0]
    assert "False - still too casual" in caplog.text


def test_polite_style_prompt_focuses_on_assistant_output_only(
    true_client: StubLLMClient,
) -> None:
    true_client.reply = "True"
    constraint = PoliteStyleConstraint(true_client)
    asyncio.run(constraint.evaluate("dummy text"))
    assert true_client.captured_prompt is not None
    assert PROMPT_HINT in true_client.captured_prompt
