from functools import lru_cache
import logging
import re
import unicodedata

from janome.tokenizer import Tokenizer

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints._utils import split_words
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


_EMPHASIS_KANA_PATTERN = re.compile(r"[っッー]|([ぁ-ゖァ-ヺー])\1")
_KANA_PATTERN = re.compile(r"[ぁ-ゖァ-ヺー]+")
_ASCII_VOWEL_PATTERN = re.compile(r"[aeiouyAEIOUY]+")


@lru_cache(maxsize=1)
def _load_tokenizer() -> Tokenizer:
    """Load a shared Janome tokenizer instance."""
    return Tokenizer()


class ConsonantsWordsIfbenchConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        words = split_words(value)

        if not words:
            reason = "[Consonants Words] No words to evaluate."
            logger.info(reason)
            return False, reason

        for idx, word in enumerate(words, start=1):
            if not self._has_emphasis_kana(word):
                reason = (
                    "[Consonants Words] Each word must include doubled kana, sokuon, or a "
                    f"long sound mark (in its reading). Failing word at index {idx}: {word!r}"
                )
                logger.info(reason)
                return False, reason

        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "各単語の読みの中に、促音（っ／ッ）、長音（ー）、または同じ仮名の連続を含めてください。",
            "「っ」や「ー」など音を強調する仮名、もしくは同じ仮名が続く読みを持つ単語のみを使ってください。",
            "すべての単語について、読みのどこかに促音・長音・連続仮名のいずれかが入るようにしてください。",
            "促音や長音、同じ仮名の連続を読みの中に含む単語だけで文章を構成してください。",
            "音を伸ばす・詰まらせる表記（っ・ッ・ー）や仮名の重なりが読みの中に現れる単語のみを使ってください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return (
            "すべての単語について、その読みの中に促音（っ／ッ）、長音（ー）、"
            "または同じ仮名の連続が1か所以上含まれるように書き換えてください。"
        )

    @staticmethod
    def _has_emphasis_kana(word: str) -> bool:
        """Return True if the word's reading contains sokuon, long vowel mark, or repeated kana.

        The check is performed on a reading string derived from Janome:
        - Prefer token.reading (katakana) when available.
        - Fall back to token.surface when reading is "*" or empty.
        - Normalize the concatenated reading with NFKC before applying the pattern.
        """
        tokenizer = _load_tokenizer()
        readings: list[str] = []

        for token in tokenizer.tokenize(word):
            # Janome usually returns reading in katakana; "*" means unknown.
            r = token.reading
            if r and r != "*":
                readings.append(r)
            else:
                readings.append(token.surface)

        if not readings:
            return False

        reading_str = "".join(readings)
        normalized = unicodedata.normalize("NFKC", reading_str)

        return bool(_EMPHASIS_KANA_PATTERN.search(normalized))


class OddEvenSyllablesWordsIfbenchConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        """Check that words alternate between odd and even syllable counts.

        Assumptions:
        - `value` is a Japanese sentence.
        - `split_words` returns a Japanese-aware tokenization.
        - Syllable count is approximated from the reading (kana) of each word.
        """
        words = split_words(value)

        if not words:
            reason = "[Odd/Even Syllables] No words to evaluate."
            logger.info(reason)
            return False, reason

        if len(words) < 2:
            reason = (
                "[Odd/Even Syllables] At least two words are required to alternate "
                "odd and even syllable counts."
            )
            logger.info(reason)
            return False, reason

        syllables = [self._syllable_count_from_reading(word) for word in words]

        if any(count <= 0 for count in syllables):
            reason = (
                "[Odd/Even Syllables] Unable to detect syllables in one or more words. "
                f"syllables={syllables}"
            )
            logger.info(reason)
            return False, reason

        parities = [count % 2 for count in syllables]

        # Alternation: every adjacent pair must have different parity (odd/even).
        alternates = all(parities[i] != parities[i + 1] for i in range(len(parities) - 1))

        if not alternates:
            reason = (
                "[Odd/Even Syllables] Words do not alternate between odd and even "
                f"syllables. syllables={syllables}, parities={parities}"
            )
            logger.info(reason)
            return False, reason

        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "各単語の読みの音節数が奇数と偶数で交互になるように並べてください。",
            "単語の読みについて、奇数音節と偶数音節の語が交互に現れる文章にしてください。",
            "読みの音節数が奇数・偶数・奇数・偶数…と交互になるように単語を並べてください。",
            "奇数音節の読みを持つ単語と偶数音節の読みを持つ単語を交互に配置してください。",
            "各単語の読みの音節数の偶奇が、隣り合う単語ごとに必ず切り替わるようにしてください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return (
            "各単語の読みの音節数が奇数・偶数・奇数・偶数…と交互になるように、"
            "単語を入れ替えたり書き換えたりしてください。"
        )

    @staticmethod
    def _syllable_count_from_reading(word: str) -> int:
        """Approximate syllable (mora) count based on the reading of a word.

        - Use Janome's reading (katakana) for each morpheme.
        - If reading is unavailable ("*"), fall back to the surface form.
        - Concatenate readings, normalize with NFKC.
        - For Japanese: count kana characters (including long vowel mark).
        - For ASCII: count vowel runs as syllables.
        - Fallback: use the length of the normalized string.
        """
        if not word:
            return 0

        tokenizer = _load_tokenizer()
        readings: list[str] = []

        for token in tokenizer.tokenize(word):
            reading = token.reading
            if reading and reading != "*":
                readings.append(reading)
            else:
                readings.append(token.surface)

        if not readings:
            return 0

        reading_str = "".join(readings)
        normalized = unicodedata.normalize("NFKC", reading_str)

        # Japanese: mora count approximated by kana characters (including ー).
        kana_chunks = _KANA_PATTERN.findall(normalized)
        kana_count = sum(len(chunk) for chunk in kana_chunks)
        if kana_count:
            return kana_count

        # ASCII: approximate syllables by counting vowel runs.
        ascii_matches = _ASCII_VOWEL_PATTERN.findall(normalized)
        if ascii_matches:
            return len(ascii_matches)

        # Fallback for other scripts: just use character length.
        return len(normalized)


class PalindromeWordsIfbenchConstraint(ConstraintGroupMixin):
    def __init__(self, minimum_palindromes: int = 10, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.minimum_palindromes = minimum_palindromes

    def evaluate(self, value: str) -> ConstraintEvaluation:
        """Check that there are enough palindrome words based on their readings.

        Assumptions:
        - `value` is a Japanese sentence.
        - `split_words` returns Japanese-aware tokens (surfaces).
        - Palindrome is judged on the reading string, not on the surface form:
          - We use Janome's reading (katakana) for each word.
          - The concatenated reading is normalized and then checked for palindromicity.
          - We require the reading length to be at least 5 characters.
        """
        words = split_words(value)

        if not words:
            reason = "[Palindrome Words] No words to evaluate."
            logger.info(reason)
            return False, reason

        tokenizer = _load_tokenizer()
        palindromes: list[tuple[str, str]] = []

        for word in words:
            reading = self._reading_for_palindrome(word, tokenizer)
            if len(reading) >= 5 and reading == reading[::-1]:
                palindromes.append((word, reading))

        if len(palindromes) < self.minimum_palindromes:
            failing_info = {
                "expected": self.minimum_palindromes,
                "found": len(palindromes),
                "palindromes": palindromes,
            }
            reason = (
                "[Palindrome Words] Not enough palindrome words (reading length >= 5). "
                f"Expected at least {self.minimum_palindromes}, found {len(palindromes)}."
            )
            logger.info("%s Details: %r", reason, failing_info)
            return False, reason

        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        # Explicitly mention "reading" so the constraint matches user-facing text.
        templates = [
            f"各単語の読みが5文字以上の回文となる単語を、少なくとも{self.minimum_palindromes}個含めてください。",
            f"読みが5文字以上の回文になる単語を、{self.minimum_palindromes}個以上文中に散りばめてください。",
            f"少なくとも{self.minimum_palindromes}個、読みが5文字以上の回文となる語を入れてください。",
            f"読みが5文字以上の回文語を{self.minimum_palindromes}個以上使用してください。",
            f"単語の読みが5文字以上の回文になるものを最低{self.minimum_palindromes}個盛り込んでください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return (
            f"読みが5文字以上の回文となる単語を"
            f"{self.minimum_palindromes}個以上追加・置き換えてください。"
        )

    @staticmethod
    def _reading_for_palindrome(word: str, tokenizer: Tokenizer) -> str:
        """Return a normalized reading string used for palindrome detection.

        - Use Janome's reading (katakana) for each morpheme.
        - If reading is unavailable ("*"), fall back to the surface form.
        - Concatenate readings and normalize with NFKC.
        - Convert katakana to hiragana, and lowercase ASCII letters,
          so that equivalent readings compare consistently.
        """
        if not word:
            return ""

        readings: list[str] = []

        for token in tokenizer.tokenize(word):
            reading = token.reading
            if reading and reading != "*":
                readings.append(reading)
            else:
                readings.append(token.surface)

        if not readings:
            return ""

        raw = "".join(readings)
        normalized = unicodedata.normalize("NFKC", raw)

        result_chars: list[str] = []
        for ch in normalized:
            # Normalize ASCII letters to lowercase
            if "A" <= ch <= "Z":
                ch = ch.lower()

            # Convert katakana to hiragana (so カ and か compare the same)
            code = ord(ch)
            if 0x30A1 <= code <= 0x30FA:  # katakana block
                ch = chr(code - 0x60)

            result_chars.append(ch)

        return "".join(result_chars)


class VowelWordsIfbenchConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        words = split_words(value)

        if not words:
            reason = "[Vowel Words] No words to evaluate."
            logger.info(reason)
            return False, reason

        vowels_used: set[str] = set()
        for word in words:
            vowels_used.update(self._extract_vowels_from_reading(word))

        if not vowels_used:
            reason = "[Vowel Words] No vowels detected in the response."
            logger.info(reason)
            return False, reason

        if len(vowels_used) > 3:
            reason = (
                "[Vowel Words] More than three vowel types detected in the response. "
                f"vowels={sorted(vowels_used)}"
            )
            logger.info(reason)
            return False, reason

        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "単語の読みで使われる母音の種類が3種類以内になるように文章を作成してください。",
            "各単語の読みで現れる母音の種類を3種類以内に抑えた語だけで段落を書いてください。",
            "読みの母音が3種類以内となる単語のみを使って文章を構築してください。",
            "読みとして現れる母音の種類が多くなりすぎないように、3種類までに限定してください。",
            "3種以内の母音だけを含む読みを持つ単語を組み合わせてください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return (
            "各単語の読みで使われる母音の種類が3種類以内になるように、"
            "語を差し替えて書き直してください。"
        )

    @staticmethod
    def _extract_vowels_from_reading(word: str) -> set[str]:
        """Extract a set of vowel types used in the reading of a word.

        - Use Janome's reading (katakana) for each morpheme.
        - If reading is unavailable ("*"), fall back to the surface form.
        - Concatenate readings and normalize with NFKC.
        - For Japanese kana, infer the vowel from the Unicode name:
          e.g., 'カ' -> 'KATAKANA LETTER KA' -> 'A' -> 'a'.
        - Treat 'ー' (long vowel mark) as repeating the previous vowel.
        - For ASCII letters, treat 'a/e/i/o/u' as vowels.
        - Return a set of 'a', 'i', 'u', 'e', 'o'.
        """
        tokenizer = _load_tokenizer()
        readings: list[str] = []

        if not word:
            return set()

        # Build reading string (katakana or fallback surface)
        for token in tokenizer.tokenize(word):
            reading = token.reading
            if reading and reading != "*":
                readings.append(reading)
            else:
                readings.append(token.surface)

        if not readings:
            return set()

        text = unicodedata.normalize("NFKC", "".join(readings))

        vowels: set[str] = set()
        previous_vowel: str | None = None

        for ch in text:
            # Long vowel mark: reuse previous vowel if any
            if ch == "ー":
                if previous_vowel:
                    vowels.add(previous_vowel)
                continue

            name = unicodedata.name(ch, "")

            # Try to infer vowel from Unicode name for kana/letters
            if "LETTER" in name:
                # e.g. "KATAKANA LETTER KA" -> "KA" -> "A"
                last = name.split()[-1]
                vowel_letter = last[-1] if last else ""
                if vowel_letter in "AIUEO":
                    vowel = vowel_letter.lower()
                    vowels.add(vowel)
                    previous_vowel = vowel
                    continue

            # ASCII vowels as a fallback
            lower = ch.lower()
            if lower in "aeiou":
                vowels.add(lower)
                previous_vowel = lower
                continue

            # Otherwise, reset previous vowel (e.g., consonants, symbols)
            previous_vowel = None

        return vowels
