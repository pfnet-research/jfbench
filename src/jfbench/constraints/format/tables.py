import logging
import re

from bs4 import BeautifulSoup
from bs4 import Comment
from bs4 import NavigableString

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class MarkdownTableFormatConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        lines = value.strip().split("\n")
        if len(lines) < 2:
            reason = "[Markdown Table] Not enough lines for a table."
            logger.info(reason)
            return False, reason

        header = lines[0].split("|")
        separator = lines[1].split("|")

        if len(header) < 2 or len(separator) < 2:
            reason = "[Markdown Table] Header or separator row is malformed."
            logger.info(reason)
            return False, reason

        for sep in separator[1:-1]:
            if not (sep.strip().startswith(("-", ":")) and sep.strip().endswith(("-", ":"))):
                reason = "[Markdown Table] Separator row is malformed."
                logger.info(reason)
                return False, reason

        for line in lines[2:]:
            columns = line.split("|")
            if len(columns) != len(header):
                reason = "[Markdown Table] Row does not match header column count."
                logger.info(reason)
                return False, reason

        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "Markdownの表を一つだけ出力し、ヘッダー行と区切り行を必ず含めてください。",
            "表以外の文章を挟まず、ヘッダーと区切りを備えた一つのMarkdown表のみを提示してください。",
            "Markdown形式のテーブルを一つだけ生成し、説明文やコードを混在させないでください。",
            "ヘッダー行および---区切りを備えた一つのMarkdown表を出力し、それ以外の内容は含めないでください。",
            "行や列数を揃えた一つのMarkdown表だけを返し、表の前後に余計な文を追加しないでください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "Markdown表形式で出力してください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "Markdown表以外の記述を削除し、ヘッダ行・区切り行・各行の列数を揃えた一つの表に書き直してください。"


class HtmlTableFormatConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        _table_fragment_re = re.compile(r"^\s*<table\b[^>]*>.*</table>\s*$", re.I | re.S)
        if not _table_fragment_re.match(value):
            reason = "[HTML Table] Not a standalone <table> fragment."
            logger.info(reason)
            return False, reason

        soup = BeautifulSoup(value, "html.parser")
        body = soup.body or soup

        significant = []
        for node in getattr(body, "contents", []):
            if isinstance(node, Comment):
                continue
            if isinstance(node, NavigableString):
                if node.strip():
                    significant.append(node)
            else:
                significant.append(node)

        if len(significant) != 1 or getattr(significant[0], "name", None) != "table":
            reason = "[HTML Table] The fragment must contain exactly one <table> and nothing else."
            logger.info(reason)
            return False, reason

        table = significant[0]

        rows = table.find_all("tr")
        if not rows:
            reason = "[HTML Table] No <tr> found inside <table>."
            logger.info(reason)
            return False, reason

        for i, r in enumerate(rows, start=1):
            if not r.find(["td", "th"], recursive=False):
                reason = f"[HTML Table] Row #{i} has no direct <td>/<th>."
                logger.info(reason)
                return False, reason

            parent = r.parent
            if parent and parent.name not in {"thead", "tbody", "tfoot", "table"}:
                reason = "[HTML Table] <tr> not inside <thead>/<tbody>/<tfoot>/<table>."
                logger.info(reason)
                return False, reason

        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        body = (
            "\n- 出力全体を単一の<table>要素にする。\n"
            "- <thead>/<tbody>/<tfoot>の下に<tr>を配置する。\n"
            "- 各<tr>直下に<td>または<th>を最低1つ置く。"
        )
        templates = [
            "単独のHTML<table>だけで構成された表を出力してください:" + body,
            "HTML表の要件は次の通りです。満たす形で出力してください:" + body,
            "余計な要素を含めず、条件を満たす<table>断片のみを返してください:" + body,
            "以下の制約を守ったHTML表を生成してください:" + body,
            "表全体が<table>一つで完結するよう、次の条件を満たしてください:" + body,
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "HTML表形式で出力してください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "余分な要素を削除し、単一の<table>要素だけで構成された正しいHTML表になるように修正してください。"


class LatexTableFormatConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        _comment_re = re.compile(r"(?<!\\)%[^\n]*")
        _token_re = re.compile(r"\\(begin|end)\{(tabular\*?)\}")
        _begin_header_re = re.compile(
            r"""
            \\begin\{(tabular\*?)\}          # begin{tabular or tabular*}
            \s*(?:\[[^\]]*\]\s*)?            # optional [..] (e.g., [t], [b], [p{..}])
            \{(?:[^{}]|\{[^{}]*\})*\}        # mandatory column spec with shallow nesting
        """,
            re.DOTALL | re.VERBOSE,
        )
        _math_wrap_re = re.compile(r"^\s*\\\[(.*)\\\]\s*$", re.DOTALL)

        # 1) Strip comments and trim outer whitespace
        s = _comment_re.sub("", value).strip()
        math_wrap = _math_wrap_re.match(s)
        if math_wrap:
            s = math_wrap.group(1).strip()
        if not s:
            reason = "[LaTeX Table] Empty string."
            logger.info(reason)
            return False, reason

        # 2) Scan begin/end tokens to detect multiplicity, nesting, or mismatches
        tokens = list(_token_re.finditer(s))
        if not tokens:
            reason = "[LaTeX Table] No \\begin{tabular} / \\end{tabular} found."
            logger.info(reason)
            return False, reason

        stack: list[str] = []
        blocks = 0
        first_begin: re.Match[str] | None = None
        last_end: re.Match[str] | None = None

        for m in tokens:
            typ, env = m.group(1), m.group(2)
            if typ == "begin":
                if stack:
                    reason = "[LaTeX Table] Nested tabular detected."
                    logger.info(reason)
                    return False, reason  # enforce exactly one non-nested table
                stack.append(env)
                if first_begin is None:
                    first_begin = m
            else:  # 'end'
                if not stack or stack[-1] != env:
                    reason = "[LaTeX Table] Mismatched \\begin and \\end."
                    logger.info(reason)
                    return False, reason
                stack.pop()
                blocks += 1
                last_end = m

        if stack or blocks != 1 or first_begin is None or last_end is None:
            reason = "[LaTeX Table] Tabular block count is not 1."
            logger.info(reason)
            return False, reason

        # 3) Ensure only whitespace exists outside the single tabular block
        if s[: first_begin.start()].strip() or s[last_end.end() :].strip():
            reason = "[LaTeX Table] Non-whitespace text outside the tabular block."
            logger.info(reason)
            return False, reason

        # 4) Validate header right after \begin{tabular[*]}
        if not _begin_header_re.match(s, first_begin.start()):
            reason = "[LaTeX Table] Missing or invalid column spec after \\begin{tabular}."
            logger.info(reason)
            return False, reason

        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "LaTeXの\\begin{tabular}...\\end{tabular}のみで構成された表を出力してください。",
            "単一のtabular環境を使ったLaTeX表として回答してください。",
            "tabular（またはtabular*）環境を1つだけ含むLaTeX表にしてください。",
            "余計なテキストを入れず、tabular環境だけで表を作成してください。",
            "LaTeXの表はtabular系環境を1ブロックだけ用いて記述してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "LaTeXの表形式で出力してください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "単一の\\begin{tabular}...\\end{tabular}ブロックになるように整形し、列定義や行記法の不備を直してください。"


class MediawikiTableFormatConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        _open_re = re.compile(r"^\s*\{\|")  # e.g., "{| class='wikitable'"
        _close_re = re.compile(
            r"^\s*\|\}\s*$"
        )  # exactly a closing line with optional surrounding ws
        _has_open_anywhere = re.compile(r"\{\|")  # to catch nested openings
        _has_close_anywhere = re.compile(r"\|\}")
        _html_comment_re = re.compile(r"<!--.*?-->", flags=re.DOTALL)

        # 0) Trim outer whitespace and strip HTML comments
        s = value.strip()
        s = _html_comment_re.sub("", s)

        if not s:
            reason = "[MediaWiki Table] Empty string after trimming/comments."
            logger.info(reason)
            return False, reason

        lines = s.splitlines()

        if not lines:
            reason = "[MediaWiki Table] No content after trimming lines."
            logger.info(reason)
            return False, reason

        # 1) Opening and closing lines
        if not _open_re.match(lines[0]):
            reason = "[MediaWiki Table] Missing or invalid opening '{|' line."
            logger.info(reason)
            return False, reason

        if not _close_re.match(lines[-1]):
            reason = (
                "[MediaWiki Table] Missing or invalid closing '|}' line (must be on its own line)."
            )
            logger.info(reason)
            return False, reason

        # 2) Ensure there is no nested opening or premature closing inside the body
        for idx, raw in enumerate(lines[1:-1], start=2):  # 1-based-ish for messaging
            if _has_open_anywhere.search(raw):
                reason = f"[MediaWiki Table] Nested '{{|' found at line {idx}."
                logger.info(reason)
                return False, reason
            if _has_close_anywhere.search(raw):
                reason = f"[MediaWiki Table] Premature '|}}' found at line {idx}."
                logger.info(reason)
                return False, reason

        # 3) Validate inner lines are table constructs only
        has_header = False
        has_data = False
        seen_data = False

        for i, raw in enumerate(lines[1:-1], start=2):
            line = raw.strip()
            if not line:
                continue  # allow blank lines inside the table

            if line.startswith("|+"):
                continue

            if line.startswith("|-"):
                continue

            if line.startswith("!"):
                if seen_data is True and has_header is False:
                    reason = f"[MediaWiki Table] Header must precede data (line {i})."
                    logger.info(reason)
                    return False, reason
                has_header = True
                continue

            if line.startswith("|"):
                has_data = True
                seen_data = True
                continue

            preview = (line[:40] + "...") if len(line) > 43 else line
            reason = f"[MediaWiki Table] Invalid line at {i}: {preview!r}"
            logger.info(reason)
            return False, reason

        if not has_header:
            reason = "[MediaWiki Table] No header line ('! ...') found."
            logger.info(reason)
            return False, reason

        if not has_data:
            reason = "[MediaWiki Table] No data row ('| ...') found."
            logger.info(reason)
            return False, reason

        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "冒頭を'{|'、末尾を'|}'とするMediaWiki表のみを出力し、ヘッダー行を必ず含めてください。",
            "MediaWiki記法の表だけを返し、ヘッダ行やデータ行以外の文章を入れないでください。",
            "MediaWikiテーブル構文に従い、ヘッダを持つ表のみを生成してください。",
            "MediaWiki形式の表を1つだけ出力し、説明文やその他のテキストを混在させないでください。",
            "ヘッダー必須のMediaWiki表を作成し、表以外の内容は含めないでください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "MediaWiki表形式で出力してください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "冒頭を「{|」終端を「|}」とするMediaWiki表記に合わせ、ヘッダやデータ行のみで構成される表に書き直してください。"
