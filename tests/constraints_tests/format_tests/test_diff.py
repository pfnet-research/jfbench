from jfbench.constraints.format.diff import DiffFormatConstraint


def test_diff_evaluate_valid_patch() -> None:
    diff_text = "--- a/file\n+++ b/file\n@@ -1,2 +1,2 @@\n-line\n+line\n"
    constraint = DiffFormatConstraint()
    assert constraint.evaluate(diff_text)[0] is True


def test_diff_evaluate_invalid_without_markers() -> None:
    constraint = DiffFormatConstraint()
    assert constraint.evaluate("plain text")[0] is False


def test_diff_evaluate_strips_code_fence() -> None:
    diff_text = "```diff\n--- a/file\n+++ b/file\n@@ -1,1 +1,1 @@\n-old\n+new\n```"
    constraint = DiffFormatConstraint()
    assert constraint.evaluate(diff_text)[0] is True


def test_diff_instructions() -> None:
    assert DiffFormatConstraint().instructions()
