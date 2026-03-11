import pytest

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints.ifbench_format import EmojiFormatIfbenchConstraint
from jfbench.constraints.ifbench_format import OutputTemplateFormatIfbenchConstraint


def test_emoji_format_ifbench_constraint_accepts_sentences_with_emojis() -> None:
    value = "楽しいね😊 最高だね🎉"
    constraint = EmojiFormatIfbenchConstraint()
    assert constraint.evaluate(value)[0] is True


def test_emoji_format_ifbench_constraint_rejects_missing_emojis() -> None:
    value = "楽しいね。 とても嬉しい！"
    constraint = EmojiFormatIfbenchConstraint()
    success, reason = constraint.evaluate(value)
    assert success is False
    assert "emoji" in (reason or "").lower()


def test_output_template_ifbench_constraint_requires_all_markers() -> None:
    constraint = OutputTemplateFormatIfbenchConstraint()
    ok_value = "私の回答：A 私の結論：B 今後の展望：C"
    bad_value = "私の回答：A 私の結論：B"
    assert constraint.evaluate(ok_value)[0] is True
    assert constraint.evaluate(bad_value)[0] is False


def _capture_options(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    captured: dict[str, list[str]] = {}

    def fake_random_instruction(self: ConstraintGroupMixin, options: list[str]) -> str:
        captured["options"] = list(options)
        return options[0]

    monkeypatch.setattr(ConstraintGroupMixin, "_random_instruction", fake_random_instruction)
    return captured


@pytest.mark.parametrize("expected_marker", ("私の回答", "私の結論", "今後の展望"))
def test_output_template_ifbench_instructions_templates(
    monkeypatch: pytest.MonkeyPatch, expected_marker: str
) -> None:
    captured = _capture_options(monkeypatch)
    constraint = OutputTemplateFormatIfbenchConstraint(seed=0)
    instruction = constraint.instructions(train_or_test="test")
    options = captured["options"]

    assert instruction == options[0]
    assert len(options) == 5
    assert all(expected_marker in option for option in options)


def test_ifbench_format_instructions_reject_train_mode() -> None:
    constraint = OutputTemplateFormatIfbenchConstraint(seed=0)
    with pytest.raises(ValueError, match="ifbench"):
        constraint.instructions(train_or_test="train")
