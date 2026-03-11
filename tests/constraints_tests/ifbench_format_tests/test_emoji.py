import pytest

from jfbench.constraints.ifbench_format import emoji as emoji_module
from jfbench.constraints.ifbench_format.emoji import EmojiFormatIfbenchConstraint


def test_emoji_format_ifbench_constraint_calls_sentence_splitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, str] = {}

    def fake_split_sentences(text: str) -> list[str]:
        captured["text"] = text
        return ["うれしい😊", "たのしい🎉"]

    monkeypatch.setattr(emoji_module, "split_sentences", fake_split_sentences)
    constraint = EmojiFormatIfbenchConstraint()
    success, reason = constraint.evaluate("dummy")

    assert success is True
    assert reason is None
    assert captured["text"] == "dummy"
