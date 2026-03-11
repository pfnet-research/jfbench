import logging
import re

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class CitationFormatConstraint(ConstraintGroupMixin):
    REFERENCES_KEYWORDS = (
        "references",
        "bibliography",
        "works cited",
        "sources",
        "参考文献",
        "引用文献",
        "出典",
        "文献",
    )

    PAREN_CITE_RE = re.compile(r"\([^)]+?(?:19|20)\d{2}[a-z]?[^)]*\)")
    NUMERIC_CITE_RE = re.compile(r"(?<!\^)\[(\d+)\]")
    FOOTNOTE_CITE_RE = re.compile(r"\[\^(\d+)\]")
    YEAR_RE = re.compile(
        r"(?:\b(?:10|11|12|13|14|15|16|17|18|19|20)\d{2}[a-z]?(?:(?=\b)|(?=[^\w])|(?=年))|n\.d\.)",
        re.IGNORECASE,
    )

    def evaluate(self, value: str) -> ConstraintEvaluation:
        text = value.strip()
        if not text:
            reason = "[Citation] No content provided."
            logger.info(reason)
            return False, reason

        # Strip code blocks to avoid noise
        cleaned = re.sub(r"```.*?```", "", text, flags=re.S)
        cleaned = re.sub(r"~~~.*?~~~", "", cleaned, flags=re.S)

        lines = cleaned.splitlines()

        entry_start_re = re.compile(r"^\s*(?:[\-\*\+]\s+|\[\^?\d+\]\s*|\d+\.\s+|•\s+)")
        header_re = re.compile(r"^\s{0,3}#{1,6}\s+\S")  # stop if a next heading appears

        def looks_like_reference_line(line: str) -> bool:
            return bool(entry_start_re.match(line)) or bool(self.YEAR_RE.search(line))

        def is_reference_header(line: str) -> bool:
            stripped = line.strip()
            if not stripped:
                return False

            lowered = stripped.lower()
            no_hash = re.sub(r"^#{1,6}\s*", "", stripped).strip().lower()

            candidates = (lowered, no_hash)
            for kw in self.REFERENCES_KEYWORDS:
                kw_lower = kw.lower()
                for s in candidates:
                    if s == kw_lower:
                        return True
                    if s.rstrip("：:") == kw_lower:
                        return True
            return False

        # Find the start of the references section (header if present, otherwise trailing block)
        body_lines: list[str] | None = None
        ref_lines: list[str] | None = None

        candidate_indices: list[int] = []
        for idx, line in enumerate(lines):
            if is_reference_header(line):
                candidate_indices.append(idx)

        selected_idx: int | None = None
        for idx in reversed(candidate_indices):
            window = lines[idx + 1 : idx + 6]
            if any(looks_like_reference_line(line) for line in window):
                selected_idx = idx
                break

        if selected_idx is not None:
            body_lines = lines[:selected_idx]
            ref_lines = lines[selected_idx + 1 :]
        else:
            ref_anchor = None
            ref_start = None
            for idx in range(len(lines) - 1, -1, -1):
                line = lines[idx]
                if not line.strip():
                    if ref_anchor is not None:
                        ref_start = idx + 1
                        break
                    continue
                if looks_like_reference_line(line):
                    ref_anchor = idx
                    continue
                if ref_anchor is not None:
                    ref_anchor = None
                    break

            if (
                ref_start is None
                and ref_anchor is not None
                and entry_start_re.match(lines[ref_anchor])
            ):
                ref_start = ref_anchor

            if ref_start is not None:
                body_lines = lines[:ref_start]
                ref_lines = lines[ref_start:]
            else:
                reason = "[Citation] References section not found."
                logger.info(reason)
                return False, reason

        if body_lines is None or ref_lines is None:
            reason = "[Citation] References section not found."
            logger.info(reason)
            return False, reason

        # Detect inline citations only from the body (exclude references area)
        body_text = "\n".join(body_lines)
        parenthetical_citations = self.PAREN_CITE_RE.findall(body_text)
        numeric_citations = self.NUMERIC_CITE_RE.findall(body_text)
        footnote_citations = self.FOOTNOTE_CITE_RE.findall(body_text)

        if not (parenthetical_citations or numeric_citations or footnote_citations):
            reason = "[Citation] No inline citations detected in body."
            logger.info(reason)
            return False, reason

        # Group reference entries by paragraph/list item
        entries: list[str] = []
        buf: list[str] = []
        seen_entry = False

        def flush() -> None:
            nonlocal seen_entry
            if buf:
                entry = " ".join(s.strip() for s in buf if s.strip())
                if entry:
                    entries.append(entry)
                    seen_entry = True
            buf.clear()

        for line in ref_lines:
            if header_re.match(line):
                flush()
                break
            if not line.strip():
                flush()
                if seen_entry:
                    break
                continue
            if entry_start_re.match(line) and buf:
                flush()
            buf.append(line)
        flush()

        if not entries:
            reason = "[Citation] References section is empty."
            logger.info(reason)
            return False, reason

        # Validate reference entries (lenient)
        ref_numbers = set()
        any_numbered = False
        for entry in entries:
            # Optional leading number like [1] or [^1]
            m = re.match(r"^\s*\[(\^)?(\d+)\]\s*", entry)
            if m:
                any_numbered = True
                ref_numbers.add(m.group(2))
                entry_body = entry[m.end() :]
            else:
                entry_body = entry

            # Require a year like 2020 / 2020a or n.d.
            if not self.YEAR_RE.search(entry_body):
                reason = f"[Citation] Reference entry missing year/n.d.: {entry}"
                logger.info(reason)
                return False, reason

            # Require at least two segments (e.g., author; title.)
            segments = [seg.strip() for seg in re.split(r"[.;　。.]", entry_body) if seg.strip()]
            if len(segments) < 2:
                reason = f"[Citation] Reference entry too short: {entry}"
                logger.info(reason)
                return False, reason

        expected_numbers = set(numeric_citations) | set(footnote_citations)

        # If references are numbered, enforce number coverage; otherwise check count only
        if expected_numbers and any_numbered:
            missing = expected_numbers - ref_numbers
            if missing:
                missing_list = ", ".join(sorted(missing))
                reason = f"[Citation] Missing reference entries for citations: {missing_list}"
                logger.info(reason)
                return False, reason
        elif expected_numbers and not any_numbered:
            if len(entries) < len(set(expected_numbers)):
                distinct = len(set(expected_numbers))
                reason = (
                    "[Citation] Fewer reference entries "
                    f"({len(entries)}) than distinct numeric/footnote citations ({distinct})."
                )
                logger.info(reason)
                return False, reason

        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "本文中で(Author, Year)もしくは[1]/[^1]といった引用記号を挿入し、最後に参考文献リストを載せてください。",
            "出力内では適切な引用表記（例: (Author, Year) や [1]）を用い、対応する出典一覧を末尾にまとめてください。",
            "本文に引用表記を付け、締めくくりにその番号や著者に紐づく参考文献一覧を必ず記載してください。",
            "[n] / [^n] または著者年方式で本文中に出典を明示し、文末に全出典の一覧を揃えてください。",
            "引用するときは(Author, Year)などの形式を使い、最後に参照元リストを整然と列挙してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "適切に引用を付与してください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return (
            "本文に適切な引用記号を挿入し、対応する参考文献リストを末尾に追加して整えてください。"
        )
