import logging
import unicodedata

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints._utils import split_sentences
from jfbench.constraints._utils import split_words
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class KeywordsSpecificPositionWordsIfbenchConstraint(ConstraintGroupMixin):
    def __init__(
        self,
        sentence_index: int,
        word_index: int,
        keyword: str,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        if sentence_index < 1 or word_index < 1:
            raise ValueError("sentence_index and word_index must be 1-based positive integers.")
        self.sentence_index = sentence_index
        self.word_index = word_index
        self.keyword = keyword

    def evaluate(self, value: str) -> ConstraintEvaluation:
        sentences = [s for s in split_sentences(value) if s.strip()]

        if len(sentences) < self.sentence_index:
            reason = (
                "[Keywords Specific Position Words] Not enough sentences to check the keyword. "
                f"Required sentence_index={self.sentence_index}, found {len(sentences)} sentences."
            )
            logger.info(reason)
            return False, reason

        target_sentence = sentences[self.sentence_index - 1]
        words = split_words(target_sentence)

        if len(words) < self.word_index:
            reason = (
                "[Keywords Specific Position Words] Not enough words in the target sentence. "
                f"Required at least {self.word_index} words, found {len(words)}. "
                f"sentence={target_sentence!r}, words={words!r}"
            )
            logger.info(reason)
            return False, reason

        actual = words[self.word_index - 1]
        if actual != self.keyword:
            reason = (
                "[Keywords Specific Position Words] The target sentence does not place "
                f"{self.keyword!r} at position {self.word_index}. "
                f"Found {actual!r} instead. "
                f"sentence_index={self.sentence_index}, word_index={self.word_index}. "
                f"sentence={target_sentence!r}, words={words!r}"
            )
            logger.info(reason)
            return False, reason

        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"{self.sentence_index}番目の文の{self.word_index}番目の単語に「{self.keyword}」を配置してください。",
            f"{self.sentence_index}文目で{self.word_index}番目の単語として「{self.keyword}」を必ず入れてください。",
            f"{self.sentence_index}番目の文では{self.word_index}番目に「{self.keyword}」という単語を置いてください。",
            f"{self.sentence_index}番目の文の指定位置（{self.word_index}番目）に「{self.keyword}」という単語を挿入してください。",
            f"「{self.keyword}」を{self.sentence_index}文目の{self.word_index}番目の単語として含めてください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return (
            f"{self.sentence_index}番目の文の{self.word_index}番目の単語が「{self.keyword}」になるように、"
            "文や単語の配置を調整してください。"
        )


class WordsPositionWordsIfbenchConstraint(ConstraintGroupMixin):
    def __init__(
        self,
        word_index: int,
        from_end_index: int,
        word: str,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        if word_index < 1 or from_end_index < 1:
            raise ValueError("word_index and from_end_index must be 1-based positive integers.")
        self.word_index = word_index
        self.from_end_index = from_end_index
        # Normalize keyword once for comparison (e.g., NFKC + lowercase)
        self.word = self._normalize_word(word)

    def evaluate(self, value: str) -> ConstraintEvaluation:
        words = split_words(value)

        if not words:
            reason = "[Words Position] No words to evaluate."
            logger.info(reason)
            return False, reason

        required_len = max(self.word_index, self.from_end_index)
        if len(words) < required_len:
            reason = (
                "[Words Position] Not enough words to satisfy the positional requirement. "
                f"Required at least {required_len} words, found {len(words)}."
            )
            logger.info(reason)
            return False, reason

        first_word = self._normalize_word(words[self.word_index - 1])
        last_word = self._normalize_word(words[-self.from_end_index])

        if first_word != self.word or last_word != self.word:
            reason = (
                f"[Words Position] Word {self.word!r} not found at required positions "
                f"{self.word_index} and -{self.from_end_index}. "
                f"Found first={first_word!r}, last={last_word!r}."
            )
            logger.info(reason)
            return False, reason

        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"回答の{self.word_index}番目の単語と最後から{self.from_end_index}番目の単語はどちらも「{self.word}」にしてください。",
            f"「{self.word}」を{self.word_index}番目と末尾から{self.from_end_index}番目に配置してください。",
            f"文章の{self.word_index}語目と終端から{self.from_end_index}語目を「{self.word}」に固定してください。",
            f"{self.word_index}番目および末尾から{self.from_end_index}番目の単語が「{self.word}」になるようにしてください。",
            f"「{self.word}」という単語を指定の2箇所（{self.word_index}番目と終わりから{self.from_end_index}番目）に置いてください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return (
            f"{self.word_index}番目と末尾から{self.from_end_index}番目の単語が"
            f"いずれも「{self.word}」になるように単語を入れ替えてください。"
        )

    @staticmethod
    def _normalize_word(word: str) -> str:
        """Normalize a word for comparison.

        - Apply NFKC normalization to unify full-width/half-width forms.
        - Lowercase ASCII letters so that case differences do not matter.
        """
        if not word:
            return word
        normalized = unicodedata.normalize("NFKC", word)
        return normalized.lower()
