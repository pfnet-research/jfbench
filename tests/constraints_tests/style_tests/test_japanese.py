import asyncio
import logging

from _pytest.logging import LogCaptureFixture
from conftest import StubLLMClient

from jfbench.constraints.style import JapaneseStyleConstraint
from jfbench.constraints.style import NoJapaneseStyleConstraint


PROMPT_HINT = "ユーザーが与えた文章を除き、アシスタントが出力した部分のみを対象としてください。"


def test_japanese_style_constraint_accepts_japanese_text(true_client: StubLLMClient) -> None:
    true_client.reply = "True: Japanese detected"
    constraint = JapaneseStyleConstraint(true_client)
    assert asyncio.run(constraint.evaluate("これは日本語の文章です。"))[0]


def test_japanese_style_constraint_rejects_non_japanese_text(
    false_client: StubLLMClient, caplog: LogCaptureFixture
) -> None:
    false_client.reply = "False - English words present"
    constraint = JapaneseStyleConstraint(false_client)
    with caplog.at_level(logging.INFO):
        result = asyncio.run(constraint.evaluate("This sentence is English."))
    assert result[0] is False
    assert "not recognized as Japanese" in str(result[1])
    assert "English words present" in caplog.text


def test_japanese_style_constraint_allows_digits_and_symbols(true_client: StubLLMClient) -> None:
    true_client.reply = "True"
    constraint = JapaneseStyleConstraint(true_client)
    text = "2024年の売上は前年比+10%でした。"
    assert asyncio.run(constraint.evaluate(text))[0]


def test_japanese_style_constraint_rejects_empty_input(
    true_client: StubLLMClient,
) -> None:
    constraint = JapaneseStyleConstraint(true_client)
    result = asyncio.run(constraint.evaluate(""))
    assert result[0] is False
    assert result[1] == "[Japanese Style] Empty value provided."


def test_japanese_style_prompt_focuses_on_assistant_output_only(
    true_client: StubLLMClient,
) -> None:
    true_client.reply = "True"
    constraint = JapaneseStyleConstraint(true_client)
    asyncio.run(constraint.evaluate("dummy text"))
    assert true_client.captured_prompt is not None
    assert PROMPT_HINT in true_client.captured_prompt


def test_no_japanese_style_constraint_rejects_when_llm_reports_false(
    false_client: StubLLMClient, caplog: LogCaptureFixture
) -> None:
    false_client.reply = "False - Japanese detected"
    constraint = NoJapaneseStyleConstraint(false_client)
    with caplog.at_level(logging.INFO):
        result = asyncio.run(constraint.evaluate("これは混在しています。"))
    assert result[0] is False
    assert "contains Japanese elements" in str(result[1])
    assert "Japanese detected" in caplog.text


def test_no_japanese_style_constraint_accepts_true_response(true_client: StubLLMClient) -> None:
    true_client.reply = "True"
    constraint = NoJapaneseStyleConstraint(true_client)
    assert asyncio.run(constraint.evaluate("This text stays non-Japanese."))[0]


def test_no_japanese_prompt_focuses_on_assistant_output_only(
    true_client: StubLLMClient,
) -> None:
    true_client.reply = "True"
    constraint = NoJapaneseStyleConstraint(true_client)
    asyncio.run(constraint.evaluate("dummy text"))
    assert true_client.captured_prompt is not None
    assert PROMPT_HINT in true_client.captured_prompt
