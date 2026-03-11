from collections import Counter
from functools import lru_cache
from itertools import pairwise
import logging
import re
import unicodedata

from janome.tokenizer import Tokenizer

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints._utils import split_sentences
from jfbench.constraints._utils import split_words
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_tokenizer() -> Tokenizer:
    """Load a shared Janome tokenizer instance."""
    return Tokenizer()


class NoConsecutiveWordsIfbenchConstraint(ConstraintGroupMixin):
    """Constraint that forbids consecutive words starting with the same character.

    Assumptions:
    - `value` is a Japanese sentence or paragraph (possibly mixed with ASCII).
    - `split_words` performs Japanese-aware tokenization.
    - "First character" means the first non-whitespace character of each token,
      normalized with NFKC and lowercased for ASCII letters.
    """

    def evaluate(self, value: str) -> ConstraintEvaluation:
        words = split_words(value)

        if len(words) < 2:
            reason = (
                "[No Consecutive Words] Not enough words to check consecutive starting characters."
            )
            logger.info(reason)
            return False, reason

        # Iterate through consecutive word pairs
        for idx, (first, second) in enumerate(pairwise(words), start=1):
            key1 = self._initial_key(first)
            key2 = self._initial_key(second)

            # If either word has no meaningful initial character, skip the pair
            if key1 is None or key2 is None:
                continue

            if key1 == key2:
                reason = (
                    "[No Consecutive Words] Found consecutive words starting with the "
                    f"same character at positions {idx} and {idx + 1}. "
                    f"words={first!r}, {second!r}, initials={key1!r}, {key2!r}"
                )
                logger.info(reason)
                return False, reason

        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "連続する2つの単語の先頭の1文字が同じにならないようにしてください。",
            "隣り合う単語が同じ文字で始まらないよう配慮してください。",
            "続けて登場する単語の先頭文字が被らないようにしてください。",
            "同じ頭文字の単語が連続しない文章にしてください。",
            "隣接する単語の開始文字が毎回変わるように語を選んでください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "連続する単語の先頭文字が重ならないように、語順や語彙を調整してください。"

    @staticmethod
    def _initial_key(word: str) -> str | None:
        """Return a normalized initial character used for comparison.

        - Apply NFKC normalization to unify full-width/half-width forms.
        - Skip whitespace characters at the beginning.
        - For ASCII letters, use lowercase.
        - For Japanese (kanji/kana) and other letters, return the first non-space
          character as-is.
        - Return None if no suitable character is found (e.g., only punctuation).
        """
        if not word:
            return None

        normalized = unicodedata.normalize("NFKC", word)

        for ch in normalized:
            if ch.isspace():
                continue

            # ASCII letters: compare in lowercase
            if "A" <= ch <= "Z" or "a" <= ch <= "z":
                return ch.lower()

            # For Japanese characters and other letters, use as-is
            return ch

        return None


class LastFirstWordsIfbenchConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        sentences = split_sentences(value)
        words_per_sentence = [split_words(sentence.lower()) for sentence in sentences]
        for previous, current in pairwise(words_per_sentence):
            if not previous or not current:
                reason = "[Last First Words] Each sentence must contain at least one word."
                logger.info(reason)
                return False, reason
            if previous[-1] != current[0]:
                reason = (
                    "[Last First Words] The last word of one sentence must match "
                    "the first word of the next sentence. "
                    f"Found last={previous[-1]!r}, first={current[0]!r}."
                )
                logger.info(reason)
                return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "各文の最後の単語を次の文の最初の単語として使ってください。",
            "文末の単語を次の文頭で繰り返すチェーン構造にしてください。",
            "前の文のラストワードを次の文の先頭に配置してください。",
            "各文の終わりの単語を、次の文の開始単語として引き継いでください。",
            "文章をつなげるように、文末語を次の文頭で再利用してください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "各文の最後の単語が次の文の最初に来るように並べ直してください。"


class ParagraphLastFirstWordsIfbenchConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        paragraphs = [block.strip() for block in value.split("\n\n") if block.strip()]
        for paragraph in paragraphs:
            words = split_words(paragraph.lower())
            if not words:
                continue
            if words[0] != words[-1]:
                reason = (
                    "[Paragraph Last First] Paragraphs must end with the same word they start. "
                    f"Found first={words[0]!r}, last={words[-1]!r}."
                )
                logger.info(reason)
                return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "各段落の冒頭と末尾を同じ単語にしてください（段落は改行で区切ってください）。",
            "段落ごとに最初と最後の単語が一致するように構成してください。",
            "パラグラフの始まりと終わりを同一の単語でそろえてください。",
            "段落の先頭語と末尾語を同一にし、改行で段落を分けてください。",
            "各段落を同じ単語で始めて同じ単語で締めてください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "各段落の最初と最後の単語が一致するように段落を調整してください。"


class RepeatsWordsIfbenchConstraint(ConstraintGroupMixin):
    def __init__(self, max_repeats: int, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.max_repeats = max_repeats

    def evaluate(self, value: str) -> ConstraintEvaluation:
        counts = Counter(word.lower() for word in split_words(value))
        if not counts:
            return True, None
        if counts.most_common(1)[0][1] >= self.max_repeats:
            reason = (
                "[Repeats Words] A word appears too many times. "
                f"Maximum allowed is {self.max_repeats - 1}."
            )
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"同じ単語を{self.max_repeats}回以上含めないでください。",
            f"同じ単語の出現回数が{self.max_repeats}回未満に収まるようにしてください。",
            f"語の重複を避け、{self.max_repeats}回以上の繰り返しがないようにしてください。",
            f"どの単語も{self.max_repeats}回以上は使わないでください。",
            f"単語の再利用は{self.max_repeats - 1}回までに抑えてください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return f"どの単語も{self.max_repeats}回以上現れないように書き直してください。"


class StartVerbWordsIfbenchConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        words = split_words(value)

        if not words:
            reason = "[Start Verb Words] No words found to check the first verb."
            logger.info(reason)
            return False, reason

        if not self._looks_like_japanese_verb(words[:5]):
            reason = (
                "[Start Verb Words] The response must start with a verb-like word "
                f"in Japanese. Found first several words: {words[:5]!r}"
            )
            logger.info(reason)
            return False, reason

        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        # Keep it simple but clearly refer to "a verb-like word" in Japanese.
        templates = [
            "回答は必ず動詞で始めてください。",
            "最初の単語を日本語の動詞（命令形や「〜してください」などを含む）にしてください。",
            "出だしの語が動詞になるようにしてください。",
            "文章は動詞から書き始めてください。",
            "先頭の単語として動詞・動詞相当の語（〜して・〜してください など）を置いてください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return (
            "先頭の単語が日本語の動詞（または動詞相当の表現）になるように、"
            "語句を差し替えてください。"
        )

    @staticmethod
    def _looks_like_japanese_verb(words: list[str]) -> bool:
        """Heuristically check if a surface form looks like a Japanese verb.

        Priority:
        1. Use Janome POS: treat tokens whose major POS is '動詞' or '助動詞'
           as verb-like.
        2. As a fallback, use surface-based suffix heuristics similar to the
           original implementation.
        """
        word = words[0]
        if not word:
            return False

        tokenizer = _load_tokenizer()

        # 1) POS-based check (preferred)
        for token in tokenizer.tokenize(word):
            pos_major = token.part_of_speech.split(",")[0]
            if pos_major in ("動詞", "助動詞"):
                return True

        # 2) Fallback: suffix-based heuristics on the surface form
        verb_endings = (
            "する",
            "した",
            "して",
            "します",
            "しろ",
            "せよ",
            "させる",
            "される",
            "なる",
            "なれ",
            "なった",
            "なって",
            "なろう",
            "ください",
            "下さい",
            "くれ",
            "いく",
            "いこう",
            "よう",
            "ように",
            "ろ",
            "よ",
        )
        if word.endswith(verb_endings):
            return True

        # Plain-form-looking endings (u-row verbs etc.)
        return bool(re.search(r"[うくぐすずつづぬむるぶぷふゆよ]$", word))
