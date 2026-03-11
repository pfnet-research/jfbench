from collections import Counter
import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints._utils import split_sentences
from jfbench.constraints._utils import split_words
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class SentenceWordsRatioIfbenchConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        sentences = [s for s in split_sentences(value) if s.strip()]

        if len(sentences) != 3:
            reason = (
                "[Sentence Words Ratio] Exactly three sentences are required "
                f"(got {len(sentences)})."
            )
            logger.info(reason)
            return False, reason

        word_lists = [self._split_sentence_words(sentence) for sentence in sentences]

        if any(not words for words in word_lists):
            reason = (
                "[Sentence Words Ratio] Each sentence must contain at least one word. "
                f"word_counts={[len(ws) for ws in word_lists]}"
            )
            logger.info(reason)
            return False, reason

        lengths = [len(words) for words in word_lists]
        if len(set(lengths)) != 1:
            reason = (
                "[Sentence Words Ratio] All three sentences must have the same word count. "
                f"word_counts={lengths}"
            )
            logger.info(reason)
            return False, reason

        flat_words = [w for words in word_lists for w in words]
        counts = Counter(flat_words)
        duplicates = {word: cnt for word, cnt in counts.items() if cnt > 1}
        if duplicates:
            reason = (
                "[Sentence Words Ratio] Sentences must use distinct words. "
                f"Found duplicates: {duplicates}"
            )
            logger.info(reason)
            return False, reason

        logger.debug(
            "[Sentence Words Ratio] Passed. sentences=%d, word_count_per_sentence=%d",
            len(sentences),
            lengths[0],
        )
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "同じ単語数で、すべて異なる単語を使った3つの文で回答してください。",
            "3文で構成し、各文の単語数をそろえつつ単語は重複させないでください。",
            "単語数が同じ3文を作り、それぞれ異なる単語のみで構成してください。",
            "3つの文を同一の単語数にし、使用単語が重ならないようにしてください。",
            "同じ単語数の3文を用い、各文で異なる単語を使用してください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "3文とも単語数を同じにし、使う単語が重ならないように調整してください。"

    @staticmethod
    def _split_sentence_words(sentence: str) -> list[str]:
        stripped = sentence.strip()
        if " " in stripped:
            return [word for word in stripped.split() if word]
        return split_words(stripped)


STOP_WORDS: set[str] = {
    # ---- particles ----
    "の",
    "に",
    "は",
    "が",
    "を",
    "と",
    "も",
    "で",
    "へ",
    "から",
    "まで",
    "より",
    "や",
    "など",
    "って",
    "とか",
    "だけ",
    "しか",
    "ほど",
    "くらい",
    # ---- conjunctions ----
    "そして",
    "しかし",
    "でも",
    "けれど",
    "けれども",
    "また",
    "それで",
    "だから",
    "なので",
    "ところが",
    # ---- politeness / auxiliaries ----
    "だ",
    "です",
    "ます",
    "でした",
    "では",
    "じゃ",
    "じゃない",
    "じゃなかった",
    "ない",
    "なかった",
    "なく",
    "なくて",
    "たい",
    "たく",
    "たくて",
    "らしい",
    "ようだ",
    "ようです",
    "みたい",
    "そうだ",
    "そうです",
    # ---- light verbs ----
    "する",
    "した",
    "して",
    "します",
    "しない",
    "できる",
    "できた",
    "できない",
    "ある",
    "あった",
    "ない",
    "いる",
    "いた",
    "いない",
    "なる",
    "なった",
}


class StopWordsRatioIfbenchConstraint(ConstraintGroupMixin):
    def __init__(self, max_ratio_percent: int, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.max_ratio_percent = max_ratio_percent

    def evaluate(self, value: str) -> ConstraintEvaluation:
        words = split_words(value)
        if not words:
            reason = "[Stop Words Ratio] No words to evaluate."
            logger.info(reason)
            return False, reason

        total_words = len(words)
        stop_count = sum(1 for w in words if w in STOP_WORDS)
        ratio = (stop_count / total_words) * 100.0

        if ratio > self.max_ratio_percent:
            reason = (
                "[Stop Words Ratio] Stop word ratio "
                f"{ratio:.2f}% exceeds maximum {self.max_ratio_percent}%. "
                f"(stop_words={stop_count}, total_words={total_words})"
            )
            logger.info(reason)
            return False, reason

        logger.debug(
            "[Stop Words Ratio] Passed. ratio=%.2f%%, stop_words=%d, total_words=%d, max=%.2f%%",
            ratio,
            stop_count,
            total_words,
            self.max_ratio_percent,
        )
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"ストップワードの割合を{self.max_ratio_percent}%以下に抑えてください。",
            f"全体の単語数に対してストップワードが{self.max_ratio_percent}%を超えないようにしてください。",
            f"ストップワード比率を{self.max_ratio_percent}%以内にコントロールしてください。",
            f"ストップワードが{self.max_ratio_percent}%を超えない文章にしてください。",
            f"ストップワードの割合を{self.max_ratio_percent}%以下に制限してください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return (
            f"ストップワードの割合が{self.max_ratio_percent}%以下になるように書き換えてください。"
        )
