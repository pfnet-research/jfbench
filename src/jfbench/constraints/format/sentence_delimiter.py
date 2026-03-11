import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints._utils import split_sentences
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


def _split_by_delimiter(text: str, delimiter: str) -> list[str]:
    # Keep the original behavior: drop empty parts.
    return [part.strip() for part in text.split(delimiter) if part.strip()]


def _has_empty_segment(text: str, delimiter: str) -> bool:
    """
    Returns True if splitting by the delimiter yields an empty (or whitespace-only)
    segment in the *leading or middle* positions.

    Examples:
      - "A||B|" -> True
      - "|A|"   -> True
      - "A|B|"  -> False

    Note:
      Splitting "A|B|" produces a trailing empty string after the last delimiter.
      We intentionally ignore the last part here and validate "must end with delimiter"
      separately.
    """
    raw_parts = text.split(delimiter)
    if len(raw_parts) <= 1:
        return False

    # Ignore the last part (often empty when the text ends with the delimiter).
    middle = raw_parts[:-1]
    return any(part.strip() == "" for part in middle)


class SentenceDelimiterFormatConstraint(ConstraintGroupMixin):
    def __init__(self, delimiter: str, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.delimiter = delimiter

    def evaluate(self, value: str) -> ConstraintEvaluation:
        text = value.strip()
        d = self.delimiter

        # 1) No delimiter at all.
        if d not in text:
            reason = f"[Sentence Delimiter Format] Delimiter '{d}' not found in output."
            logger.info(reason)
            return False, reason

        # 2) If the instruction requires every sentence to end with the delimiter,
        #    enforce that the whole output ends with the delimiter as well.
        if not text.rstrip().endswith(d):
            reason = f"[Sentence Delimiter Format] Output must end with delimiter '{d}'."
            logger.info(reason)
            return False, reason

        # 3) Reject consecutive/leading delimiters that create empty "sentences".
        if _has_empty_segment(text, d):
            reason = (
                "[Sentence Delimiter Format] Empty sentence detected "
                "(e.g., consecutive/leading delimiter)."
            )
            logger.info(reason)
            return False, reason

        delimiter_segments = _split_by_delimiter(text, d)
        if not delimiter_segments:
            reason = (
                "[Sentence Delimiter Format] No sentences found after splitting by the delimiter."
            )
            logger.info(reason)
            return False, reason

        # 4) Compare sentence counts estimated by the sentence splitter (pysbd wrapper)
        #    with the delimiter-based segmentation.
        normalized_text = text.replace(d, "。")
        pysbd_segments = split_sentences(normalized_text)

        # Enforce a strict match to prevent:
        #   - missing delimiters (pysbd > delimiter)
        #   - delimiter misuse inside a sentence (pysbd < delimiter)
        if len(pysbd_segments) != len(delimiter_segments):
            if len(pysbd_segments) > len(delimiter_segments):
                reason = (
                    "[Sentence Delimiter Format] Detected more sentence boundaries than delimiters. "
                    "Some sentences may be missing the configured delimiter."
                )
            else:
                reason = (
                    "[Sentence Delimiter Format] Detected more delimiters than sentence boundaries. "
                    "The delimiter may be used inside a sentence."
                )
            logger.info(reason)
            return False, reason

        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"各文の区切りとして必ず「{self.delimiter}」を用いてください。",
            f"文の終わりには毎回「{self.delimiter}」を付けて区切ってください。",
            f"回答の文はそれぞれ「{self.delimiter}」で終わらせて連結してください。",
            f"文を繋ぐときは必ず「{self.delimiter}」を境界として入れてください。",
            f"すべての文間に「{self.delimiter}」を挿入して出力してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return f"文末の区切りをすべて「{self.delimiter}」に揃えてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return f"文の区切り記号をすべて「{self.delimiter}」に統一してください。"
