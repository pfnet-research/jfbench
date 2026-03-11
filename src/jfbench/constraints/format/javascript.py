import logging
import re

import esprima

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class JavascriptFormatConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        return _evaluate_js_syntax(value, "[JavaScript]")

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "有効なJavaScriptコードのみを出力し、構文エラーがないようにしてください。",
            "回答はJavaScriptソースとしてそのまま実行できる形式で記述してください。",
            "esprimaで解析して問題ない完全なJavaScriptコードを生成してください。",
            "JavaScriptの構文規則に従ったコードのみを返し、無効な文は含めないでください。",
            "実行可能なJavaScriptプログラムを出力し、構文チェックで失敗しないようにしてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "構文エラーのないJavaScriptコードとして記述してください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "構文エラーを取り除き、JavaScriptとしてパース可能なコードに修正してください。"


class NoCodeFenceJavascriptFormatConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        code, _language = _extract_leading_code_fence(value)
        if code is not None:
            reason = "[No Code Fence JS] Code fences must not be used."
            logger.info(reason)
            return False, reason
        return _evaluate_js_syntax(value, "[No Code Fence JS]")

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "コードフェンスを使わずにJavaScriptコードを出力してください。",
            "``` を使わずにJSコードのみを書いてください。",
            "フェンスなしでJavaScriptのコードを示してください。",
            "バッククォートで囲まずにJSコードを書いてください。",
            "コードブロックを使わずJavaScriptを提示してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "コードフェンスを使わずにJavaScriptコードのみを書いてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "JavaScriptコードをコードフェンスなしでそのまま記述してください。"


class WithCodeFenceJavascriptFormatConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        code, language = _extract_leading_code_fence(value)
        if code is None:
            reason = (
                "[With Code Fence JS] Code fences are required at the beginning of the output."
            )
            logger.info(reason)
            return False, reason
        language_hint = (language or "").lower()
        if language_hint not in {"javascript", "js"}:
            reason = "[With Code Fence JS] Language hint for JavaScript is missing."
            logger.info(reason)
            return False, reason
        return _evaluate_js_syntax(code, "[With Code Fence JS]")

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "JavaScriptコードをコードフェンス付きで記述してください（```javascript）。",
            "JSコードを```で囲み、言語指定を入れてください。",
            "コードフェンスを用いてJavaScriptのコードを提示してください。",
            "```javascript 形式でコードを出力してください。",
            "フェンス付きのJavaScriptコードブロックで回答してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return (
                "```javascript で始まるコードフェンスで囲ってJavaScriptのコードで答えてください。"
            )
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "```javascript ... ``` の形式でコードを囲ってください。"


def _unwrap_markdown_code_fence(s: str) -> str:
    code, _language = _extract_leading_code_fence(s)
    return code if code is not None else s


def _extract_leading_code_fence(s: str) -> tuple[str | None, str | None]:
    stripped = s.lstrip()
    m = re.match(
        r"(?P<fence>```|~~~)(?P<lang>[a-zA-Z0-9_+-]*)\s*\n?(?P<body>.*?)(?:\n)?(?P=fence)",
        stripped,
        flags=re.S,
    )
    if not m:
        return None, None
    return m.group("body"), m.group("lang")


def _evaluate_js_syntax(src: str, reason_prefix: str) -> ConstraintEvaluation:
    is_valid, error_msg = is_valid_js(src)
    if not is_valid:
        reason = f"{reason_prefix} Syntax error: {error_msg}"
        logger.info(reason)
        return False, reason
    return True, None


def is_valid_js(src: str) -> tuple[bool, str | None]:
    src = _unwrap_markdown_code_fence(src)
    src = src.lstrip("\ufeff")
    if src.startswith("#!"):
        lines = src.splitlines()
        src = "\n".join(lines[1:])

    try:
        esprima.parseModule(src, tolerant=False)
        return True, None
    except esprima.Error as e_mod:
        try:
            esprima.parseScript(src, tolerant=False)
            return True, None
        except esprima.Error as e_scr:
            return False, f"Module error: {e_mod}; Script error: {e_scr}"
    except Exception as e:
        return False, f"Unexpected error during parsing: {e}"


__all__ = [
    "JavascriptFormatConstraint",
    "NoCodeFenceJavascriptFormatConstraint",
    "WithCodeFenceJavascriptFormatConstraint",
    "esprima",
]
