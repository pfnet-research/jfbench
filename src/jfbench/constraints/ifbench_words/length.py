import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints._utils import split_words
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class PrimeLengthsWordsIfbenchConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        """Check that all tokenized words have prime-length surfaces.

        Assumptions:
        - `value` is a Japanese sentence.
        - `split_words` performs Japanese-aware tokenization and returns
          surface strings (kanji/kana/ASCII etc.).
        - "Length" here means the number of characters in each token's surface.
        """
        words = split_words(value)

        if not words:
            reason = "[Prime Length Words] No words to evaluate."
            logger.info(reason)
            return False, reason

        for idx, word in enumerate(words, start=1):
            length = len(word)
            if not self._is_prime(length):
                reason = (
                    "[Prime Length Words] All words must have prime lengths. "
                    f"Failing word at index {idx}: {word!r} (length={length})"
                )
                logger.info(reason)
                return False, reason

        logger.debug(
            "[Prime Length Words] Passed. words=%r, lengths=%s",
            words,
            [len(w) for w in words],
        )
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        # Keep wording in terms of "character count" so it matches len(word).
        templates = [
            "素数の文字数を持つ単語のみを使用して回答してください。",
            "各単語の文字数がすべて素数になるようにしてください。",
            "文字数が素数となる単語だけで文章を構成してください。",
            "各単語の長さ（文字数）が素数となる語を選んでください。",
            "素数の文字数を持たない単語は使わないでください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return (
            "各単語の文字数が素数になるように単語を置き換え、"
            "素数長の単語だけで文章を再構成してください。"
        )

    @staticmethod
    def _is_prime(value: int) -> bool:
        if value < 2:
            return False
        if value == 2:
            return True
        if value % 2 == 0:
            return False
        factor = 3
        while factor * factor <= value:
            if value % factor == 0:
                return False
            factor += 2
        return True
