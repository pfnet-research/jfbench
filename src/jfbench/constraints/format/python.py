import ast
import logging
import re

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class PythonFormatConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        return _validate_python_code(value)

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "Pythonの有効なコードのみを出力し、構文エラーがない状態にしてください。",
            "回答はPythonソースコードの形式で書き、ast.parseで正しく解析できる必要があります。",
            "Pythonインタプリタがそのまま実行できるコードを生成してください。",
            "無効な構文を含めず、完全なPythonコードとして出力を提供してください。",
            "Pythonの構文規則に従い、コードのみを提示して実行可能にしてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "構文エラーのないPythonコードとして記述してください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "無効な構文を修正し、Pythonとして構文解析できるコードに整えてください。"


class NoCodeFencePythonFormatConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        code, _ = _extract_leading_code_fence(value)
        if code is not None:
            reason = "[No Code Fence Python] Code fences must not be used."
            logger.info(reason)
            return False, reason

        ok, parse_reason = _parse_python(value)
        if not ok:
            return False, parse_reason

        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "コードフェンスなしでPythonコードを記述してください。",
            "``` を使わずにPythonコードのみを書いてください。",
            "フェンスを付けずにPythonのコードを示してください。",
            "バッククォートなしでPythonコードを提示してください。",
            "コードブロックを用いずPythonコードを書いてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "コードフェンスを使わず、Pythonコードだけを記述してください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "Pythonコードをコードフェンスなしでそのまま書いてください。"


class WithCodeFencePythonFormatConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        code, language = _extract_leading_code_fence(value)
        if code is None:
            reason = "[With Code Fence Python] A leading code fence is required at the start of the output."
            logger.info(reason)
            return False, reason

        language_hint = (language or "").lower()
        if not language_hint or ("python" not in language_hint and language_hint != "py"):
            reason = "[With Code Fence Python] Language hint for Python is missing at the leading code fence."
            logger.info(reason)
            return False, reason

        ok, parse_reason = _parse_python(code)
        if not ok:
            return False, parse_reason

        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "Pythonコードを```python のコードフェンス付きで記述してください。",
            "```で囲み、言語指定にpythonを付けてコードを書いてください。",
            "フェンス付きのPythonコードブロックで回答してください。",
            "```python 形式でコードを提示してください。",
            "バッククォート三つで囲んだPythonコードを出力してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "```python で始まり ``` で閉じる形でPythonコードのみを記述してください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "```python ... ``` 形式でコードを囲ってください。"


def _unwrap_markdown_code_fence(s: str) -> str:
    # Prefer a fenced block explicitly labeled as python
    m = re.search(r"```(?:python)?\s*(.*?)\s*```", s, flags=re.S | re.I)
    if m:
        return m.group(1)
    m = re.search(r"~~~(?:python)?\s*(.*?)\s*~~~", s, flags=re.S | re.I)
    if m:
        return m.group(1)
    return s


def _extract_leading_code_fence(s: str) -> tuple[str | None, str | None]:
    stripped = s.lstrip()
    m = re.match(
        r"(?P<fence>```|~~~)(?P<lang>[a-zA-Z0-9_+-]*)\s*\n?(?P<code>.*?)(?:\n)?(?P=fence)",
        stripped,
        flags=re.S,
    )
    if not m:
        return None, None
    return m.group("code"), m.group("lang")


def _validate_python_code(value: str) -> ConstraintEvaluation:
    return _parse_python(_unwrap_markdown_code_fence(value))


def _parse_python(code: str) -> ConstraintEvaluation:
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        reason = f"[Python] Syntax error: {e}"
        logger.info(reason)
        return False, reason
