import pytest

from jfbench.constraints.ifbench_format.newline import NewlineFormatIfbenchConstraint


def test_newline_format_ifbench_constraint_rejects_multiple_tokens() -> None:
    constraint = NewlineFormatIfbenchConstraint()
    value = "今日は\n晴れです"

    success, reason = constraint.evaluate(value)

    assert success is False
    assert "one word" in (reason or "").lower()


def test_newline_format_ifbench_constraint_uses_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constraint = NewlineFormatIfbenchConstraint()
    calls: list[str] = []

    def dummy_split_words(text: str) -> list[str]:
        calls.append(text)
        return ["token"]

    monkeypatch.setattr(
        "jfbench.constraints.ifbench_format.newline.split_words", dummy_split_words
    )

    success, reason = constraint.evaluate("テスト\n成功")

    assert success is True
    assert reason is None
    assert calls == ["テスト", "成功"]
