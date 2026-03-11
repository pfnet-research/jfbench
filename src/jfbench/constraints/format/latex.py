import logging
import re
from typing import Any

from pylatexenc.latexwalker import get_default_latex_context_db
from pylatexenc.latexwalker import LatexEnvironmentNode
from pylatexenc.latexwalker import LatexGroupNode
from pylatexenc.latexwalker import LatexMacroNode
from pylatexenc.latexwalker import LatexMathNode
from pylatexenc.latexwalker import LatexWalker
from pylatexenc.latexwalker import LatexWalkerParseError
from pylatexenc.macrospec import EnvironmentSpec
from pylatexenc.macrospec import LatexContextDb
from pylatexenc.macrospec import MacroSpec

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


def _build_context() -> LatexContextDb:
    """
    Return a LaTeX context extended with common AMS macros/environments.
    """
    ctx = get_default_latex_context_db()

    add_envs = [
        EnvironmentSpec(n)
        for n in [
            "align",
            "align*",
            "gather",
            "gather*",
            "multline",
            "multline*",
            "equation*",
            "split",
        ]
    ]

    add_macros = [
        MacroSpec("bm", "{"),
        MacroSpec("DeclareMathOperator", "*{{"),
        MacroSpec("operatorname", "*{"),
        MacroSpec("mathbb", "{"),
        MacroSpec("mathcal", "{"),
        MacroSpec("mathrm", "{"),
        MacroSpec("mathbf", "{"),
        MacroSpec("boldsymbol", "{"),
        MacroSpec("href", "{{"),
    ]

    ctx.add_context_category(
        "jfbench-amsmath",
        prepend=True,
        macros=add_macros,
        environments=add_envs,
    )
    return ctx


class LatexFormatConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        cleaned = self._strip_code_fence(value)
        ok, errors = self._pylatexenc_syntax_check(cleaned)
        if ok:
            return True, None
        if errors:
            reason = "\n".join(f"[LaTeX] syntax error: {error}" for error in errors)
        else:
            reason = "[LaTeX] syntax error."
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "出力は正しいLaTeXコードとして解釈できる形式にしてください。",
            "LaTeXソースとしてそのままコンパイル可能な形で回答を記述してください。",
            "LaTeXの構文に従い、整合性のある完全なソースを出力してください。",
            "回答全体を有効なLaTeX文書として書き、構文エラーを含まないようにしてください。",
            "LaTeXパーサーが問題なく読み取れるよう、正規のLaTeX記法のみで出力してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "LaTeX形式で出力してください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "対応していない環境・括弧を修正し、LaTeXとして構文解析できるように整えてください。"

    @classmethod
    def _pylatexenc_syntax_check(cls, value: str) -> tuple[bool, list[str]]:
        """
        Parse LaTeX and return (ok, errors).

        ok == False if:
        - parsing fails (unbalanced braces, mismatched begin/end, math delimiter issues), or
        - an \\item appears outside of a list environment.
        """
        ctx = _build_context()
        walker = LatexWalker(value, latex_context=ctx)

        try:
            nodes, _, _ = walker.get_latex_nodes()
        except LatexWalkerParseError as e:
            return False, [f"Parse error: {e}"]

        errors: list[str] = []
        env_stack: list[str] = []
        visited_ids: set[int] = set()

        def visit(node: Any) -> None:
            nonlocal env_stack
            nid = id(node)
            if nid in visited_ids:
                return
            visited_ids.add(nid)

            if isinstance(node, LatexMacroNode):
                # orphan \item -> error
                if node.macroname == "item":
                    allowed = {"itemize", "enumerate", "description"}
                    if not any(e in allowed for e in env_stack):
                        errors.append(r"\item used outside of a list environment")

            elif isinstance(node, LatexEnvironmentNode):
                env_stack.append(node.environmentname)
                for ch in node.nodelist or []:
                    visit(ch)
                env_stack.pop()
                return  # avoid double traversal

            elif isinstance(node, LatexGroupNode):
                for ch in node.nodelist or []:
                    visit(ch)
                return

            elif isinstance(node, LatexMathNode):
                pass  # math delimiters are balanced enough to parse

            # traverse macro arguments
            nodeargd = getattr(node, "nodeargd", None)
            if nodeargd is not None:
                for arg in nodeargd.argnlist or []:
                    if arg is not None:
                        visit(arg)

            # traverse generic children
            for ch in getattr(node, "nodelist", []) or []:
                visit(ch)

        for n in nodes:
            visit(n)

        ok = len(errors) == 0
        return ok, errors

    @staticmethod
    def _strip_code_fence(value: str) -> str:
        stripped = value.strip()
        match = re.match(r"\A```[^\n]*\n(.*)\n```\s*\Z", stripped, re.DOTALL)
        if match is None:
            return value
        return match.group(1)
