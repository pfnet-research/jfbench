from collections.abc import Sequence
from itertools import pairwise
import logging
import re
from typing import Optional
from typing import Protocol
import unicodedata
from urllib.parse import urlparse

from markdown_it import MarkdownIt
from markdown_it.token import Token

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class _MarkdownCheck(Protocol):
    def evaluate(self, value: str) -> ConstraintEvaluation: ...


class MarkdownFormatConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        checks: list[tuple[_MarkdownCheck, str]] = [
            (MarkdownParseableConstraint(), "[Markdown] Parseable constraint failed"),
            (MarkdownClosedFencesConstraint(), "[Markdown] Closed fences constraint failed"),
            (MarkdownHeadingJumpsConstraint(), "[Markdown] Heading jumps constraint failed"),
            (MarkdownLinksAndImagesConstraint(), "[Markdown] Links and images constraint failed"),
            (
                MarkdownHeadingsStructureConstraint(),
                "[Markdown] Headings structure constraint failed",
            ),
            (
                MarkdownReferenceLinksConstraint(),
                "[Markdown] Reference links constraint failed",
            ),
            (MarkdownListStructureConstraint(), "[Markdown] List structure constraint failed"),
            (MarkdownTableStructureConstraint(), "[Markdown] Table structure constraint failed"),
            (
                MarkdownUnconsumedEmphasisMarkersConstraint(),
                "[Markdown] Unconsumed emphasis markers constraint failed",
            ),
        ]
        failures: list[str] = []
        for constraint, failure_message in checks:
            passed, reason = constraint.evaluate(value)
            if not passed:
                message = reason or failure_message
                failures.append(message)
        if failures:
            combined = "\n".join(failures)
            logger.info(combined)
            return False, combined
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        detail_items = [
            MarkdownParseableConstraint().instructions(),
            MarkdownClosedFencesConstraint().instructions(),
            MarkdownHeadingJumpsConstraint().instructions(),
            MarkdownLinksAndImagesConstraint().instructions(),
            MarkdownHeadingsStructureConstraint().instructions(),
            MarkdownReferenceLinksConstraint().instructions(),
            MarkdownListStructureConstraint().instructions(),
            MarkdownTableStructureConstraint().instructions(),
            MarkdownUnconsumedEmphasisMarkersConstraint().instructions(),
        ]
        details = "\n".join(self._format_detail_instructions(text) for text in detail_items)
        templates = [
            "Markdown形式で出力し、次の条件をすべて満たしてください:\n" + details,
            "以下の検証をすべて通過できる完全なMarkdownを生成してください:\n" + details,
            "Markdown全体が一貫しており、次の各制約を満たす形にしてください:\n" + details,
            "示すルールを順守したMarkdown出力のみを受け付けます:\n" + details,
            "Markdownの品質チェックを全て満たす必要があります。要件は次の通りです:\n" + details,
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return f"Markdown形式の文字列として、次の条件を満たすようにしてください:\n{details}"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        components = [
            MarkdownParseableConstraint().rewrite_instructions(),
            MarkdownClosedFencesConstraint().rewrite_instructions(),
            MarkdownHeadingJumpsConstraint().rewrite_instructions(),
            MarkdownLinksAndImagesConstraint().rewrite_instructions(),
            MarkdownHeadingsStructureConstraint().rewrite_instructions(),
            MarkdownReferenceLinksConstraint().rewrite_instructions(),
            MarkdownListStructureConstraint().rewrite_instructions(),
            MarkdownTableStructureConstraint().rewrite_instructions(),
            MarkdownUnconsumedEmphasisMarkersConstraint().rewrite_instructions(),
        ]
        bullets = "\n".join(f"- {text}" for text in components)
        return f"Markdown出力全体を修正し、次の条件を満たしてください:\n{bullets}"

    @staticmethod
    def _format_detail_instructions(instruction: str) -> str:
        lines = instruction.splitlines() or [""]
        first_line, *rest = lines
        formatted_lines = [f"- {first_line}"]
        for line in rest:
            formatted_lines.append(f"  {line}")
        return "\n".join(formatted_lines)


class MarkdownParseableConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        md = MarkdownIt("commonmark")
        try:
            md.parse(value)
            return True, None
        except Exception as e:
            reason = f"[Markdown] Parsing failed: {e}"
            logger.info(reason)
            return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "Markdownパーサーで問題なく解析できる形式で出力してください。",
            "構文エラーのない正規のMarkdown文書を生成してください。",
            "Markdownとして解釈可能なテキストのみを返し、不正な記法を避けてください。",
            "パーサーが失敗しない正しいMarkdown構造で文章を組み立ててください。",
            "Markdown仕様に沿った書き方で、解析に失敗する構造を含めないでください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "構文や閉じタグ、エスケープを調整し、Markdownパーサーでエラーが出ないようにしてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "Markdownパーサーでエラーにならないように構文や閉じタグ、エスケープを修正してください。"


class MarkdownClosedFencesConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        issues = self._check_unclosed_fences(value)
        if issues:
            reasons = [f"[Markdown] Unclosed code fence: {issue}" for issue in issues]
            message = "\n".join(reasons)
            logger.info(message)
            return False, message
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "開いたコードフェンスには必ず対応する閉じフェンスを入れたMarkdownを出力してください。",
            "コードブロックは```や~~~が対になっている状態で記述してください。Markdownで書いてください。",
            "途中で終わるフェンスが無いように、全コードフェンスを閉じたMarkdownを書いてください。",
            "フェンスを開いたら同じ記号で閉じる完全なコードブロック構造を守ってMarkdownで書いてください。",
            "未クローズのコードフェンスが存在しないよう、Markdownを整形してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "開いたコードフェンスに対応する閉じフェンスを追加し、記号と長さを揃えてください。Markdown形式で記述してください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "開いたコードフェンスごとに対応する閉じフェンスを追加し、フェンスの長さと種類を揃えてください。"

    @classmethod
    def _check_unclosed_fences(cls, text: str) -> list[str]:
        issues: list[str] = []
        stack: list[tuple[str, int, int]] = []  # (char, length, start_line)
        fence_re = re.compile(r"^\s{0,3}([`~]{3,})([^`~\n]*)$")
        for i, line in enumerate(text.splitlines(), start=1):
            m = fence_re.match(line)
            if not m:
                continue
            fence = m.group(1)
            ch, ln = fence[0], len(fence)
            if stack and stack[-1][0] == ch and ln >= stack[-1][1]:
                stack.pop()
            else:
                stack.append((ch, ln, i))
        for ch, ln, start in stack:
            issues.append(f"Unclosed code fence {ch * ln!s} (opened at line {start})")
        return issues


class MarkdownHeadingJumpsConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        md = MarkdownIt("commonmark")
        tokens = md.parse(value)
        issues = self._check_heading_jumps(tokens)
        if issues:
            reasons = [f"[Markdown] Heading level jump: {issue}" for issue in issues]
            message = "\n".join(reasons)
            logger.info(message)
            return False, message
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "少なくとも1つの見出しを含め、見出しレベルが飛び級しないMarkdownにしてください。",
            "Markdown文書としてH1→H2→H3のように1段ずつ上がる階層で見出しを構成し、最低1つの見出しを入れてください。",
            "見出しは連続したレベルで並べ、H2の次に突然H4へ飛ばすといった構成を避けてください。Markdownで書いてください。",
            "Markdown形式で書いてください。段階を飛ばさずに見出しを配置し、少なくとも1つは見出しを用意してください。",
            "見出しレベルに穴が開かない構造（例: H2の前にH1）を守りながらMarkdownを書いてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "Markdown形式で書いてください。見出しレベルが飛ばないように整え、不足している階層の見出しを補ってください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "見出しレベルが1段階ずつ上がるようにH階層を並べ替えるか不足しているレベルの見出しを追加してください。"

    @classmethod
    def _check_heading_jumps(cls, tokens: Sequence[Token]) -> list[str]:
        issues: list[str] = []
        levels = [int(t.tag[1]) for t in tokens if t.type == "heading_open"]
        if not levels:
            issues.append("No headings found")
            return issues
        prev = None
        for idx, lv in enumerate(levels, start=1):
            if prev is not None and lv > prev + 1:
                issues.append(f"Heading level jump: H{prev} -> H{lv} (at heading #{idx})")
            prev = lv
        return issues


class MarkdownLinksAndImagesConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        md = MarkdownIt("commonmark")
        tokens = md.parse(value)
        issues = self._check_links_and_images(tokens)
        if issues:
            unique = sorted(set(issues))
            reasons = [f"[Markdown] Links and images issue: {issue}" for issue in unique]
            message = "\n".join(reasons)
            logger.info(message)
            return False, message
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        body = (
            "\n少なくとも1つの有効なリンクと1つのalt付き画像を含めてください。\n"
            "- リンクは http/https/mailto/tel/ftp のいずれかのスキームのみ使用可能です。\n"
            "- すべての画像に意味のある代替テキストを設定してください。"
        )
        templates = [
            "リンクと画像の要件を満たすMarkdownにしてください。" + body,
            "Markdown形式で許可されたスキームのリンクとalt必須の画像を最低1つずつ配置してください。\n"
            + body,
            "リンクのスキーム・画像のalt条件を守り、両方を含むMarkdownを作成してください。\n"
            + body,
            "Markdownで書いてください。リンク/画像の検査に通るよう、下記ルールを満たしてください:\n"
            + body,
            "Markdown文書としてリンクと画像の品質基準を満たす必要があります:\n" + body,
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "Markdown形式で書いてください。有効なリンクと代替テキスト付き画像を最低1つずつ入れ、http/https/mailto/tel/ftpだけを使ってください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "少なくとも1つの有効なリンクと代替テキスト付き画像を追加し、許可されたスキームのみを使うように修正してください。"

    @classmethod
    def _check_links_and_images(cls, tokens: Sequence[Token]) -> list[str]:
        ALLOWED_SCHEMES = {"http", "https", "mailto", "tel", "ftp"}
        issues: list[str] = []
        link_count = 0
        image_count = 0

        def walk(ts: Sequence[Token]) -> None:
            nonlocal link_count, image_count
            for t in ts:
                if t.type == "link_open":
                    link_count += 1
                    href = dict(t.attrs or {}).get("href", "")
                    if not href.strip():
                        issues.append("Link without href")
                    else:
                        p = urlparse(href)
                        scheme = (p.scheme or "").lower()
                        if scheme and scheme not in ALLOWED_SCHEMES:
                            issues.append(f"Suspicious URL scheme: {href}")
                elif t.type == "image":
                    image_count += 1
                    src = (t.attrGet("src") or "").strip()
                    if not src:
                        issues.append("Image without src")
                    alt = (t.attrGet("alt") or t.content or "").strip()
                    if not alt:
                        issues.append("Image without alt text")
                if t.children:
                    walk(t.children)

        walk(tokens)

        if link_count == 0:
            issues.append("No links found")
        if image_count == 0:
            issues.append("No images found")
        return issues


class MarkdownHeadingsStructureConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        md = MarkdownIt("commonmark")
        tokens = md.parse(value)
        issues = self._check_headings_structure(tokens)
        if issues:
            reasons = [f"[Markdown] Heading structure issue: {issue}" for issue in issues]
            message = "\n".join(reasons)
            logger.info(message)
            return False, message
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        body = (
            "\n- H1見出しは1つだけにする。\n"
            "- 各見出しのタイトルを空にしない。\n"
            "- スラグ（アンカー）が重複しないようにする。"
        )
        templates = [
            "見出し構造のルールを順守したMarkdownにしてください。" + body,
            "Markdownで書いてください。見出しの数・内容・スラグに関する下記条件を満たしてください:"
            + body,
            "見出しは一意で空でないタイトルを持ち、H1は1つという規則を守ってください。Markdownでお願いします。"
            + body,
            "以下のヘッダー要件を満たすMarkdownのみ有効です:" + body,
            "Markdown形式としてH1の重複禁止や空見出し禁止などの構造規則に従ってください:" + body,
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "Markdownで書いてください。空の見出しをなくし、H1を1つにし、同じタイトルが重複しないように整えてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "空の見出しを埋め、H1は1つだけに調整し、同じタイトルは重ならないように書き換えてください。"

    @classmethod
    def _check_headings_structure(cls, tokens: Sequence[Token]) -> list[str]:
        issues: list[str] = []
        h1_count = 0
        slugs: set[str] = set()
        for i in range(len(tokens) - 1):
            if tokens[i].type == "heading_open" and tokens[i + 1].type == "inline":
                level = int(tokens[i].tag[1])
                title = tokens[i + 1].content.strip()
                if not title:
                    issues.append("Empty heading title")
                if level == 1:
                    h1_count += 1
                slug = cls._gh_slugify(title)
                if slug in slugs:
                    issues.append(f"Duplicate heading slug: {slug}")
                slugs.add(slug)
        if h1_count == 0:
            issues.append("Missing H1")
        if h1_count > 1:
            issues.append("Multiple H1 headings")
        return issues

    @classmethod
    def _gh_slugify(cls, text: str) -> str:
        s = text.strip().lower()
        s = "".join(ch for ch in s if ch.isalnum() or ch in (" ", "-"))
        s = re.sub(r"\s+", "-", s).strip("-")
        return s


class MarkdownReferenceLinksConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        md = MarkdownIt("commonmark")
        tokens = md.parse(value)
        heading_slugs = {
            self._gh_slugify(t.content.strip())
            for i, t in enumerate(tokens)
            if t.type == "inline" and i > 0 and tokens[i - 1].type == "heading_open"
        }
        issues = self._check_reference_links(value, heading_slugs)
        if issues:
            reasons = [f"[Markdown] Reference links issue: {issue}" for issue in issues]
            message = "\n".join(reasons)
            logger.info(message)
            return False, message
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        body = (
            "\n- 未定義の参照リンクを使わない。\n"
            "- 同じ参照名は同一URLを指す。\n"
            "- 見出しアンカーが存在する場所だけを参照する。"
        )
        templates = [
            "参照リンクが破綻しないMarkdownを作成してください。" + body,
            "Markdownとしてリンク参照の定義と利用が一致するよう、以下の条件を満たしてください:"
            + body,
            "参照リンクを用いる場合は、未定義や矛盾がない状態のMarkdown文書にしてください:" + body,
            "Markdownで書いてください。内部アンカーを含めた参照リンクの整合性を保ち、次のルールを守ってください:"
            + body,
            "Markdown形式で書いてください。参照リンクの定義/使用ルールは以下の通りです。必ず従ってください:"
            + body,
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "参照リンクの定義と使用を整合させ、未定義や壊れた内部アンカーがない状態で記述してください。Markdown形式で書いてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "参照リンクの定義と使用を照合し、未定義・重複・壊れた内部アンカーを解消して整えてください。"

    @classmethod
    def _check_reference_links(cls, text: str, heading_slugs: set[str]) -> list[str]:
        ref_def_re = re.compile(r"^\s*\[([^\]]+)\]:\s*(\S+)", re.MULTILINE)
        ref_use_re = re.compile(r"\[([^\]]+)\]\[([^\]]+)\]|\[([^\]]+)\]\[\]")
        issues: list[str] = []
        defs: dict[str, str] = {}
        for m in ref_def_re.finditer(text):
            key, url = m.group(1).strip().lower(), m.group(2).strip()
            defs.setdefault(key, url)
        for m in ref_use_re.finditer(text):
            key = (m.group(2) or m.group(3) or "").strip().lower()
            if key and key not in defs:
                issues.append(f"Undefined link reference: [{key}]")
        seen: dict[str, str] = {}
        for m in ref_def_re.finditer(text):
            key, url = m.group(1).strip().lower(), m.group(2).strip()
            if key in seen and seen[key] != url:
                issues.append(f"Conflicting reference definition: [{key}] -> {seen[key]} vs {url}")
            seen[key] = url
        anchor_re = re.compile(r"\[[^\]]*\]\(#([^)#\s]+)\)")
        for m in anchor_re.finditer(text):
            anchor = m.group(1).strip().lower()
            if anchor not in heading_slugs:
                issues.append(f"Broken internal anchor: #{anchor}")
        return issues

    @classmethod
    def _gh_slugify(cls, text: str) -> str:
        s = text.strip().lower()
        s = "".join(ch for ch in s if ch.isalnum() or ch in (" ", "-"))
        s = re.sub(r"\s+", "-", s).strip("-")
        return s


class MarkdownListStructureConstraint(ConstraintGroupMixin):
    OL_RE = re.compile(r"^(\s*)(\d{1,9})([.)])([ \t]+)(.*)$")  # cap length to avoid absurd numbers
    BLANK_RE = re.compile(r"^\s*$")

    def evaluate(self, value: str) -> ConstraintEvaluation:
        md = MarkdownIt("commonmark")
        tokens = md.parse(value)
        issues = self._check_no_empty_list_items(tokens)
        issues += self._check_no_mixed_list_types_per_block(tokens)
        issues += self._check_accidental_lists(value, tokens)
        if issues:
            reasons = [f"[Markdown] List structure issue: {issue}" for issue in issues]
            message = "\n".join(reasons)
            logger.info(message)
            return False, message
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        body = (
            "\n- 空のリスト項目を作らない。\n"
            "- 同一ブロックで記号付きリストと番号付きリストを混在させない。"
        )
        templates = [
            "リスト構造の次の制約を満たすMarkdownにしてください:" + body,
            "Markdown形式としてリスト項目に内容を入れ、リスト種別を統一するという条件を守ってください:"
            + body,
            "Markdownで書いてください。空のリストや混在リストを避けるよう、以下のルールに従ってください:"
            + body,
            "Markdownとしてリストの品質基準を満たしてください。条件は次の通りです:" + body,
            "出力はMarkdown形式で書く必要があります。箇条書きの整合性を保つために、下記要件に従ってください:"
            + body,
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "空のリスト項目を埋め、同じ階層では一種類のリストに揃えてください。出力はMarkdown形式で記述してください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "空のリスト項目を埋め、同じ階層のリストでは一種類の箇条書きのみになるよう並び替えてください。"

    @classmethod
    def _list_item_range(cls, tokens: Sequence[Token], i_open: int) -> int:
        """
        Given index of 'list_item_open', return index of matching 'list_item_close'.
        """
        depth = 0
        for j in range(i_open, len(tokens)):
            if tokens[j].type == "list_item_open":
                depth += 1
            elif tokens[j].type == "list_item_close":
                depth -= 1
                if depth == 0:
                    return j
        return i_open  # fallback

    @classmethod
    def _list_block_range(
        cls,
        tokens: Sequence[Token],
        i_open: int,
        open_type: str,
        close_type: str,
    ) -> int:
        """
        Given index of 'bullet_list_open' or 'ordered_list_open', find matching close index.
        """
        depth = 0
        for j in range(i_open, len(tokens)):
            if tokens[j].type == open_type:
                depth += 1
            elif tokens[j].type == close_type:
                depth -= 1
                if depth == 0:
                    return j
        return i_open

    @classmethod
    def _check_no_empty_list_items(cls, tokens: Sequence[Token]) -> list[str]:
        """
        A list item must contain some non-list content (paragraph, code, quote, table, heading, HTML).
        An item that only nests another list is considered empty by this policy.
        """
        issues: list[str] = []

        i = 0
        while i < len(tokens):
            if tokens[i].type != "list_item_open":
                i += 1
                continue
            j = cls._list_item_range(tokens, i)
            has_content = False
            k = i + 1
            while k < j:
                t = tokens[k]
                if t.type in (
                    "paragraph_open",
                    "fence",
                    "code_block",
                    "blockquote_open",
                    "table_open",
                    "heading_open",
                    "html_block",
                    "html_inline",
                ):
                    has_content = True
                    break
                # A bare 'inline' may appear (rare) without paragraph_open; accept it
                if t.type == "inline" and (t.content or "").strip():
                    has_content = True
                    break
                k += 1
            if not has_content:
                line = tokens[i].map[0] + 1 if tokens[i].map else "?"
                issues.append(f"Empty list item at line {line}. Add text before any nested list.")
            i = j + 1
        return issues

    @classmethod
    def _check_no_mixed_list_types_per_block(cls, tokens: Sequence[Token]) -> list[str]:
        issues: list[str] = []
        seen_type: dict[int, str] = {}
        depth = 0
        for t in tokens:
            if t.type in ("bullet_list_open", "ordered_list_open"):
                kind = "ul" if t.type == "bullet_list_open" else "ol"
                if depth in seen_type and seen_type[depth] != kind:
                    line = (t.map[0] + 1) if t.map else "?"
                    issues.append(
                        f"Mixed list types at depth {depth}: prev {seen_type[depth]!r}, now {kind!r} (line {line})."
                    )
                else:
                    seen_type.setdefault(depth, kind)
                depth += 1
            elif t.type in ("bullet_list_close", "ordered_list_close"):
                depth = max(0, depth - 1)
                seen_type.pop(depth, None)
        return issues

    @classmethod
    def _check_accidental_lists(
        cls,
        text: str,
        tokens: Sequence[Token],
    ) -> list[str]:
        """
        Heuristics:
        - Root-level single-item list with no blank line before (and/or after).
        - Ordered list starting with a suspiciously large number (e.g., 1900/2000… likely a year).
        """
        year_threshold = 1900
        issues: list[str] = []
        lines = text.splitlines()

        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t.type not in ("bullet_list_open", "ordered_list_open"):
                i += 1
                continue
            list_close = cls._list_block_range(
                tokens,
                i,
                "bullet_list_open" if t.type == "bullet_list_open" else "ordered_list_open",
                "bullet_list_close" if t.type == "bullet_list_open" else "ordered_list_close",
            )
            # Count items
            item_count = 0
            first_item_idx = None
            for k in range(i + 1, list_close):
                if tokens[k].type == "list_item_open":
                    item_count += 1
                    if first_item_idx is None:
                        first_item_idx = k

            # Determine list start line from first item (more reliable than list token)
            start_line = (
                tokens[first_item_idx].map[0]
                if first_item_idx is not None and tokens[first_item_idx].map
                else None
            )
            end_line = (
                tokens[list_close].map[0] if tokens[list_close].map else None
            )  # first line after list

            # Heuristic A: single-item list with no blank line before
            if item_count == 1 and start_line is not None:
                prev_line_idx = start_line - 1
                prev_blank = (prev_line_idx < 0) or cls.BLANK_RE.match(lines[prev_line_idx])
                if not prev_blank:
                    issues.append(
                        f"Possible accidental list at line {start_line + 1}: single-item list without a blank line before."
                    )
                # Also check next line blankness (optional)
                if end_line is not None and end_line < len(lines):
                    next_blank = cls.BLANK_RE.match(lines[end_line])
                    if not next_blank:
                        issues.append(
                            f"Possible accidental list around line {start_line + 1}: no blank line after the list."
                        )

            # Heuristic B: ordered list starting with a suspiciously large number (likely a year)
            if (
                t.type == "ordered_list_open"
                and first_item_idx is not None
                and start_line is not None
            ):
                item_numbers = cls._collect_ordered_list_numbers(tokens, i + 1, list_close, lines)
                line = lines[start_line] if start_line < len(lines) else ""
                m = cls.OL_RE.match(line)
                if m:
                    num = int(m.group(2))
                    if num >= year_threshold and not cls._looks_like_year_list(
                        item_numbers, year_threshold
                    ):
                        example = f"\\{num}."
                        issues.append(
                            f"Ordered list begins with a large number ({num}) at line {start_line + 1}. "
                            f"If this is a year or numbering, consider escaping (e.g., {example!r}) or adding a blank line."
                        )

            i = list_close + 1

        return issues

    @classmethod
    def _collect_ordered_list_numbers(
        cls, tokens: Sequence[Token], start: int, end: int, lines: Sequence[str]
    ) -> list[int]:
        numbers: list[int] = []
        for idx in range(start, end):
            if tokens[idx].type != "list_item_open":
                continue
            line_idx = tokens[idx].map[0] if tokens[idx].map else None
            if line_idx is None or line_idx >= len(lines):
                continue
            m = cls.OL_RE.match(lines[line_idx])
            if m:
                numbers.append(int(m.group(2)))
        return numbers

    @classmethod
    def _looks_like_year_list(cls, numbers: Sequence[int], threshold: int) -> bool:
        if len(numbers) < 2:
            return False
        upper_bound = 2100
        return all(threshold <= n <= upper_bound for n in numbers) and all(
            earlier <= later for earlier, later in pairwise(numbers)
        )


class MarkdownTableStructureConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        md = MarkdownIt("commonmark").enable("table")
        tokens = md.parse(value)
        issues = self.check_tables(tokens)
        if issues:
            unique = sorted(set(issues))
            reasons = [f"[Markdown] Table structure issue: {issue}" for issue in unique]
            message = "\n".join(reasons)
            logger.info(message)
            return False, message
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        body = "\n- 各行のセル数を揃える。\n- ヘッダーセルを空欄にしない。"
        templates = [
            "Markdown表の構造要件を満たすよう、次の条件を守ってください:" + body,
            "出力はMarkdown形式で、テーブルは列数を統一し、ヘッダーを必ず埋めてください:" + body,
            "Markdownで書いてください。表組みは下記ルール準拠で記述してください:" + body,
            "Markdownの表の品質を保つため、行ごとのセル数とヘッダーの空欄禁止を守ってください:"
            + body,
            "Markdown形式で書いてください。テーブル構造チェックで失敗しないよう、以下を徹底してください:"
            + body,
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "各Markdown表の列数を揃え、ヘッダーセルが空にならないよう補ってください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "各Markdown表で列数を揃え、ヘッダーセルを空にしないように値を補ってください。"

    @classmethod
    def check_tables(cls, tokens: Sequence[Token]) -> list[str]:
        issues: list[str] = []
        i = 0
        n = len(tokens)

        while i < n:
            t = tokens[i]
            if t.type != "table_open":
                i += 1
                continue

            rows: list[list[str]] = []
            j = i + 1
            while j < n and tokens[j].type != "table_close":
                if tokens[j].type == "tr_open":
                    cells: list[str] = []
                    k = j + 1
                    while k < n and tokens[k].type != "tr_close":
                        if tokens[k].type in ("th_open", "td_open"):
                            content_parts: list[str] = []
                            m = k + 1
                            while m < n and tokens[m].type not in ("th_close", "td_close"):
                                if tokens[m].type == "inline":
                                    content_parts.append(tokens[m].content)
                                m += 1
                            cells.append("".join(content_parts))
                            k = m
                        k += 1
                    rows.append(cells)
                    j = k
                j += 1

            if rows:
                expected = len(rows[0])
                for r_ix, r in enumerate(rows, start=1):
                    if len(r) != expected:
                        issues.append(
                            f"Table row {r_ix} has {len(r)} cells but expected {expected}"
                        )
                if any((c.strip() == "") for c in rows[0]):
                    issues.append("Empty header cell in table")

            i = max(i + 1, j + 1)

        return issues


class MarkdownUnconsumedEmphasisMarkersConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        md = MarkdownIt("commonmark")
        tokens = md.parse(value)
        issues = self._check_unconsumed_emphasis_markers(tokens)
        if issues:
            unique = sorted(set(issues))
            reasons = [
                f"[Markdown] Unconsumed emphasis markers issue: {issue}" for issue in unique
            ]
            message = "\n".join(reasons)
            logger.info(message)
            return False, message
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        body = (
            "\n- 少なくとも1つは強調マーカー(*, _)を正しく使用する。\n"
            "- 余った強調マーカーやエスケープされない裸の記号を残さない。"
        )
        templates = [
            "強調記号の整合性を守ったMarkdownを出力してください:" + body,
            "Markdownの強調マーカーが余らないよう、以下の条件を満たしてください:" + body,
            "強調表現に関する制約を守るMarkdownにしてください:" + body,
            "Markdownで書いてください。強調マーカー(*, _)を適切にペアにし、次のルールを守ってください:"
            + body,
            "Markdown形式として強調記号の取り扱いで失敗しないよう、以下の要件に従ってください:"
            + body,
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "強調マーカー(*, _)が余らないよう、ペアを追加するか不要分を削除してください。少なくとも1つは正しく使ってください。出力はMarkdown形式で記述してください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "強調マーカー(*, _)が余らないように対応するペアを追加するか不要なマーカーを削除してください。"

    @classmethod
    def _check_unconsumed_emphasis_markers(cls, tokens: Sequence[Token]) -> list[str]:
        issues: list[str] = []
        has_emphasis = False

        for t in tokens:
            if t.type != "inline" or not t.children:
                continue

            if any(c.type in ("em_open", "strong_open") for c in t.children):
                has_emphasis = True

            children: list[Token] = list(t.children)
            for ci, child in enumerate(children):
                if child.type != "text" or not child.content:
                    continue
                s = child.content
                i = 0
                while i < len(s):
                    ch = s[i]
                    if ch not in ("*", "_"):
                        i += 1
                        continue

                    j = i
                    while j < len(s) and s[j] == ch:
                        j += 1
                    run_len = j - i

                    if cls._count_backslashes_before(s, i) % 2 == 1:
                        i = j
                        continue

                    prev_ch, next_ch = cls._peek_prev_next(children, ci, s, i, j)

                    left_flanking = (not cls._is_ws(next_ch)) and (
                        not cls._is_punct_or_symbol(next_ch)
                        or cls._is_ws(prev_ch)
                        or cls._is_punct_or_symbol(prev_ch)
                    )
                    right_flanking = (not cls._is_ws(prev_ch)) and (
                        not cls._is_punct_or_symbol(prev_ch)
                        or cls._is_ws(next_ch)
                        or cls._is_punct_or_symbol(next_ch)
                    )

                    if ch == "_" and cls._is_alnum(prev_ch) and cls._is_alnum(next_ch):
                        left_flanking = False
                        right_flanking = False

                    if not left_flanking and not right_flanking:
                        context = s[max(0, i - 8) : min(len(s), j + 8)]
                        issues.append(
                            f"Unconsumed marker run {ch * run_len!r} near: …{context}…. "
                            f"Escape it (\\{ch}) or adjust punctuation/spacing."
                        )

                    i = j

        if not has_emphasis:
            issues.append("No emphasis (em/strong) found")

        return issues

    @classmethod
    def _is_ws(cls, ch: Optional[str]) -> bool:
        return ch is None or ch.isspace()

    @classmethod
    def _is_punct_or_symbol(cls, ch: Optional[str]) -> bool:
        if not ch:
            return False
        cat = unicodedata.category(ch)  # P* = punctuation, S* = symbols
        return cat.startswith("P") or cat.startswith("S")

    @classmethod
    def _is_alnum(cls, ch: Optional[str]) -> bool:
        if not ch:
            return False
        cat = unicodedata.category(ch)  # L* = letters, N* = numbers
        return cat[0] in ("L", "N")

    @classmethod
    def _count_backslashes_before(cls, s: str, idx: int) -> int:
        k, j = 0, idx - 1
        while j >= 0 and s[j] == "\\":
            k += 1
            j -= 1
        return k

    @classmethod
    def _peek_prev_next(
        cls,
        children: Sequence[Token],
        child_idx: int,
        s: str,
        run_start: int,
        run_end: int,
    ) -> tuple[Optional[str], Optional[str]]:
        # previous char
        prev_ch: Optional[str] = s[run_start - 1] if run_start > 0 else None
        if prev_ch is None:
            for j in range(child_idx - 1, -1, -1):
                n = children[j]
                if n.type == "text" and n.content:
                    prev_ch = n.content[-1]
                    break
        # next char
        next_ch: Optional[str] = s[run_end] if run_end < len(s) else None
        if next_ch is None:
            for j in range(child_idx + 1, len(children)):
                n = children[j]
                if n.type == "text" and n.content:
                    next_ch = n.content[0]
                    break
        return prev_ch, next_ch


__all__ = [
    "MarkdownClosedFencesConstraint",
    "MarkdownFormatConstraint",
    "MarkdownHeadingJumpsConstraint",
    "MarkdownHeadingsStructureConstraint",
    "MarkdownLinksAndImagesConstraint",
    "MarkdownListStructureConstraint",
    "MarkdownParseableConstraint",
    "MarkdownReferenceLinksConstraint",
    "MarkdownTableStructureConstraint",
    "MarkdownUnconsumedEmphasisMarkersConstraint",
]
