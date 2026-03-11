import logging
import re

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class QuoteUnquoteFormatIfbenchConstraint(ConstraintGroupMixin):
    # Pattern for "..." or '...' followed by non-quoted explanation text
    QUOTE_EXPLANATION_PATTERN = re.compile(
        r"""
        (["'])          # opening quote
        (.+?)           # quoted content (non-empty)
        \1              # matching closing quote
        \s+             # at least some whitespace
        ([^"'「」\s].+) # explanation starting with non-quote, non-space
        """,
        re.DOTALL | re.VERBOSE,
    )

    # Pattern for Japanese corner quotes: 「...」 explanation
    JP_QUOTE_EXPLANATION_PATTERN = re.compile(
        r"""
        「              # Japanese opening quote
        (.+?)           # quoted content (non-empty)
        」              # Japanese closing quote
        \s*             # some whitespace
        ([^「」\s].+)   # explanation starting with non-quote, non-space
        """,
        re.DOTALL | re.VERBOSE,
    )

    def evaluate(self, value: str) -> ConstraintEvaluation:
        # Handle empty or whitespace-only responses explicitly
        if not value or not value.strip():
            reason = "[Quote Unquote Format] Empty response."
            logger.info(reason)
            return False, reason

        # Fast path: try to find any quoted phrase followed by explanation
        if self.QUOTE_EXPLANATION_PATTERN.search(
            value
        ) or self.JP_QUOTE_EXPLANATION_PATTERN.search(value):
            return True, None

        # Differentiate error messages for better debugging
        has_ascii_quotes = ('"' in value) or ("'" in value)
        has_jp_quotes = ("「" in value) and ("」" in value)

        if not (has_ascii_quotes or has_jp_quotes):
            reason = "[Quote Unquote Format] No quoted phrase found."
        else:
            reason = (
                "[Quote Unquote Format] Could not find quoted phrase followed by "
                "non-quoted explanation."
            )

        logger.info("%s value_preview=%r", reason, value[:80])
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "引用したフレーズの後に必ず非引用の説明を続けてください。",
            "引用符で囲んだ言葉の直後に、その説明を素のテキストで補ってください。",
            '「」や""などで囲んだフレーズの後に説明を続けてください。',
            "引用部分の後に引用外の説明文を必ず添えてください。",
            "引用語句の直後に非引用の解説を追加してください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "引用したフレーズの後に、引用なしの説明文を追記してください。"


class QuotesFormatIfbenchConstraint(ConstraintGroupMixin):
    _REQUIRED_DEPTH = 3

    def _max_alternating_quote_depth(self, value: str) -> tuple[int, bool]:
        """
        Compute a simple heuristic of nested alternating quote depth.

        We treat a quote character as:
        - opening a new level when it is different from the current top of the stack
        - closing the current level when it is the same as the top of the stack

        This is a heuristic for natural language style nested quotes like:
            "outer 'middle \"inner\"' text"
        """
        stack: list[str] = []
        max_depth = 0

        for ch in value:
            if ch not in ('"', "'"):
                continue

            if stack and stack[-1] == ch:
                # Same quote as top of stack -> treat as closing
                stack.pop()
            else:
                # Different quote or stack empty -> treat as opening
                stack.append(ch)
                max_depth = max(max_depth, len(stack))

        # Return both the maximum depth and whether all quotes were balanced
        return max_depth, len(stack) == 0

    def evaluate(self, value: str) -> ConstraintEvaluation:
        # Handle empty or whitespace-only responses explicitly
        if not value or not value.strip():
            reason = "[Quotes Format] Empty response."
            logger.info(reason)
            return False, reason

        # Quick pre-check: we need at least two of each quote type to even try
        if value.count('"') < 2 or value.count("'") < 2:
            reason = "[Quotes Format] Nested alternating quotes are missing."
            logger.info(reason)
            return False, reason

        max_depth, balanced = self._max_alternating_quote_depth(value)

        # Require at least the configured nested depth and balanced quotes
        if balanced and max_depth >= self._REQUIRED_DEPTH:
            return True, None

        reason = (
            "[Quotes Format] Could not detect alternating nested quotes of "
            f"required depth (found depth={max_depth}, balanced={balanced})."
        )
        logger.info("%s value_preview=%r", reason, value[:80])
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "引用符の中にさらに引用符を入れ、ダブルとシングルを交互に使って少なくとも3段にしてください。",
            "入れ子の引用を3層以上作り、\" と ' を交互に配置してください。",
            "ダブルクォートとシングルクォートを交互に用いた三重の引用構造を作ってください。",
            "引用を重ねて3段以上にし、\"と'を交互に挟んでください。",
            "三層以上の入れ子引用を作り、ダブルとシングルを交互にしてください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "ダブルクォートとシングルクォートを交互に重ねた3段以上の引用にしてください。"
