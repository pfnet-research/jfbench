from jfbench.constraints.ifbench_count import PunctuationCountIfbenchConstraint


def test_punctuation_requires_japanese_marks() -> None:
    constraint = PunctuationCountIfbenchConstraint()
    value = (
        "確認します。要素をまとめ、段階を整理；次に：問いは？驚き！"
        "括弧（サンプル）も入れて、最後に驚きと疑問を？！併記します。"
    )

    success, reason = constraint.evaluate(value)

    assert success is True
    assert reason is None


def test_punctuation_requires_fullwidth_parentheses() -> None:
    constraint = PunctuationCountIfbenchConstraint()
    value = "句読点。テスト、例；確認：質問？驚き！組み合わせ？！"

    success, reason = constraint.evaluate(value)

    assert success is False
    assert reason is not None
    assert "Parentheses are required" in reason
