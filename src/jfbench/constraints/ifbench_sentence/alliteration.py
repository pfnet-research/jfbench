from functools import lru_cache
import logging
import unicodedata

from janome.tokenizer import Tokenizer

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints._utils import split_sentences
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_tokenizer() -> Tokenizer:
    """Load a shared Janome tokenizer instance."""
    return Tokenizer()


class AlliterationIncrementSentenceIfbenchConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        # Remove empty sentences just in case
        sentences = [s for s in split_sentences(value) if s.strip()]
        if not sentences:
            reason = "[Alliteration Increment Sentence] No sentences to evaluate."
            logger.info(reason)
            return False, reason

        tokenizer = _load_tokenizer()
        runs: list[int] = []
        previous_run = 0

        for idx, sentence in enumerate(sentences, start=1):
            longest_run = 0
            run = 0
            previous_key: str | None = None

            for token in tokenizer.tokenize(sentence):
                surface = token.surface.strip()
                if not surface:
                    continue

                # Janome: part_of_speech = "品詞,品詞細分類1,品詞細分類2,品詞細分類3,活用型,活用形,原形,読み,発音"
                pos0 = token.part_of_speech.split(",")[0]
                if pos0 in ("助詞", "助動詞", "記号"):
                    # Do not break the run; just ignore these tokens
                    continue

                # Prefer reading (katakana). If unknown, Janome usually returns "*".
                reading = token.reading
                key = self._alliteration_key(surface, reading)
                if key is None:
                    continue

                if key == previous_key:
                    run += 1
                else:
                    run = 1

                previous_key = key
                if run > longest_run:
                    longest_run = run

            if longest_run == 0:
                reason = (
                    "[Alliteration Increment Sentence] "
                    f"Sentence {idx} has no valid words for alliteration."
                )
                logger.info(reason)
                return False, reason

            runs.append(longest_run)

            # Each sentence must have a strictly larger run than the previous one
            if longest_run <= previous_run:
                reason = (
                    "[Alliteration Increment Sentence] "
                    f"Alliteration run did not increase at sentence {idx}. "
                    f"runs={runs}"
                )
                logger.info(reason)
                return False, reason

            previous_run = longest_run

        logger.debug("[Alliteration Increment Sentence] Passed. runs=%s", runs)
        return True, None

    @staticmethod
    def _alliteration_key(surface: str, reading: str | None) -> str | None:
        if reading and reading != "*":
            base = reading
        elif surface:
            base = surface
        else:
            return None

        normalized = unicodedata.normalize("NFKC", base)

        SMALL_HIRA_MAP = {
            "ぁ": "あ",
            "ぃ": "い",
            "ぅ": "う",
            "ぇ": "え",
            "ぉ": "お",
            "ゃ": "や",
            "ゅ": "ゆ",
            "ょ": "よ",
            "っ": "つ",
        }

        DAKUTEN_BASE_MAP = {
            "が": "か",
            "ぎ": "き",
            "ぐ": "く",
            "げ": "け",
            "ご": "こ",
            "ざ": "さ",
            "じ": "し",
            "ず": "す",
            "ぜ": "せ",
            "ぞ": "そ",
            "だ": "た",
            "ぢ": "ち",
            "づ": "つ",
            "で": "て",
            "ど": "と",
            "ば": "は",
            "び": "ひ",
            "ぶ": "ふ",
            "べ": "へ",
            "ぼ": "ほ",
            "ぱ": "は",
            "ぴ": "ひ",
            "ぷ": "ふ",
            "ぺ": "へ",
            "ぽ": "ほ",
        }

        def to_hiragana(ch: str) -> str:
            code = ord(ch)
            if 0x30A1 <= code <= 0x30FA:
                return chr(code - 0x60)
            return ch

        for ch in normalized:
            if "A" <= ch <= "Z" or "a" <= ch <= "z":
                return ch.lower()

            if ch.isspace():
                continue

            hira = to_hiragana(ch)

            if "ぁ" <= hira <= "ん":
                base_char = SMALL_HIRA_MAP.get(hira, hira)
                base_char = DAKUTEN_BASE_MAP.get(base_char, base_char)
                return base_char

            if unicodedata.category(hira).startswith("L"):
                return hira

        return None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "各文は前の文よりも長い連続した頭韻の並びを含めてください。",
            "文ごとに連続する頭韻の長さが伸びるようにしてください。",
            "連続した同音開始語の数を文ごとに増やしてください。",
            "各文で頭韻の連続が前文よりも長くなるように構成してください。",
            "前の文より長い頭韻の連続を次の文で作ってください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "文を書き直し、頭韻の連続が文を追うごとに長くなるようにしてください。"
