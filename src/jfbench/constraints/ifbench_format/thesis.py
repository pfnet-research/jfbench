import logging
import re

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class ThesisFormatIfbenchConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        # Handle empty or whitespace-only responses explicitly
        if not value or not value.strip():
            reason = "[Thesis Format] Empty response."
            logger.info(reason)
            return False, reason

        # Normalize line endings to handle Windows/macOS/Linux consistently
        normalized = value.replace("\r\n", "\n").replace("\r", "\n")

        # Split sections by blank lines (lines that may contain only whitespace)
        sections = [block.strip() for block in re.split(r"\n\s*\n", normalized) if block.strip()]

        if not sections:
            reason = (
                "[Thesis Format] No sections detected. "
                "Separate sections with two consecutive newlines."
            )
            logger.info(reason)
            return False, reason

        # Thesis line must be the first non-empty line of each section
        # and must be fully wrapped in <i>...</i>
        thesis_pattern = re.compile(r"^\s*<i>.+?</i>\s*$")

        for idx, section in enumerate(sections, start=1):
            # Extract non-empty lines within the section
            lines = [line for line in section.splitlines() if line.strip()]

            if not lines:
                reason = f"[Thesis Format] Section {idx} is empty."
                logger.info(reason)
                return False, reason

            first_line = lines[0]

            if not thesis_pattern.match(first_line):
                reason = (
                    f"[Thesis Format] Section {idx} must start with an italicized "
                    "thesis line fully wrapped in <i>...</i>."
                )
                logger.info("%s first_line=%r", reason, first_line)
                return False, reason

        # All sections start with a valid thesis line
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "各セクションはHTMLの<i>タグを用いたイタリック体の論旨文で始めてください。",
            "セクション冒頭に<i>…</i>で示す斜体の主張文を配置してください。",
            "それぞれのセクションを、HTMLのイタリック表記で始めてください。",
            "各節の先頭行を<i>タグで囲った論旨文にしてください。",
            "セクション開始時にイタリック体（<i>…</i>）の主張を入れてください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates) + "セクションは改行2つで区切ってください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "各セクションの先頭に<i>…</i>で括った論旨文を追加してください。セクションは改行2つで区切ってください。"
