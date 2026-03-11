import logging
import re
import string
import unicodedata

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)

HALF_WIDTH_ALLOWED_PATTERN = r"\u0020-\u007E\uFF61-\uFF9F\uFFA0-\uFFDC\uFFE8-\uFFEE\u201C\u201D"
KANJI_ALLOWED_PUNCTUATION = {"、", "。", "！", "？"}
SCRIPT_ALLOWED_ASCII_PUNCTUATION = set("(){}-|!@#$%^&*-=_+'\"?/.,<>[]\\;:`~（）・？！。、，．")
HIRAGANA_KATAKANA_ALLOWED_MARKS = {"ー"}


def _remove_whitespace(value: str) -> str:
    return "".join(ch for ch in value if not ch.isspace())


class UppercaseCharacterConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        letters = re.findall(r"[A-Za-z]", value)
        if not letters:
            reason = "[Uppercase Character] No alphabetic characters found."
            logger.info(reason)
            return False, reason
        if any(letter.islower() for letter in letters):
            reason = "[Uppercase Character] All alphabetic characters must be uppercase."
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "少なくとも一つのアルファベットを含め、登場する英字はすべて大文字にしてください。",
            "アルファベットを使用する場合は大文字のみで表記してください。",
            "英字はすべて大文字で記述してください。",
            "大文字のアルファベットを含め、英字に小文字を混ぜないでください。",
            "英字を入れる際は必ず大文字にしてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "英単語を少なくとも一つ含めてください。英字はすべて大文字で記述してください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "含まれる英字をすべて大文字に統一してください。"


class LowercaseCharacterConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        letters = re.findall(r"[A-Za-z]", value)
        if not letters:
            reason = "[Lowercase Character] No alphabetic characters found."
            logger.info(reason)
            return False, reason
        if any(letter.isupper() for letter in letters):
            reason = "[Lowercase Character] All alphabetic characters must be lowercase."
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "少なくとも一つのアルファベットを含め、登場する英字はすべて小文字にしてください。",
            "英字を使う際は小文字のみで統一してください。",
            "アルファベットは小文字だけを使用してください。",
            "英字を含める場合、必ず小文字で書いてください。",
            "大文字を含めず小文字の英字のみで表記してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "英単語を少なくとも一つ含めてください。英字はすべて小文字で記述してください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "含まれる英字をすべて小文字に統一してください。"


class HiraganaCharacterConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        cleaned_value = _remove_whitespace(value)
        if not cleaned_value:
            reason = "[Hiragana Character] Empty response."
            logger.info(reason)
            return False, reason
        for ch in cleaned_value:
            if "\u3040" <= ch <= "\u309f":
                continue
            if ch in KANJI_ALLOWED_PUNCTUATION:
                continue
            if ch in HIRAGANA_KATAKANA_ALLOWED_MARKS:
                continue
            if ch in SCRIPT_ALLOWED_ASCII_PUNCTUATION:
                continue
            if unicodedata.category(ch).startswith("S"):
                continue
            if ch in string.ascii_letters:
                continue
            if ch in string.digits:
                continue
            reason = (
                "[Hiragana Character] Only hiragana, punctuation, and symbols are allowed. "
                f"Invalid character: {ch}"
            )
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "すべてひらがな（数字・句読点・記号は可）で記述してください。",
            "文章をひらがなのみで書いてください（数字・句読点・記号は使用可）。",
            "漢字やカタカナを使わず、ひらがなだけで書いてください（数字・記号は可）。",
            "ひらがな表記のみを用いてください（数字・句読点・記号は可）。",
            "ひらがな以外の文字を含めないでください（数字・記号は許可）。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "ひらがなのみで記述してください。漢字・カタカナは使用不可です（数字・記号可）。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "ひらがなと数字、記号のみになるように文字を置き換えてください。"


class KatakanaCharacterConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        cleaned_value = _remove_whitespace(value)
        if not cleaned_value:
            reason = "[Katakana Character] Empty response."
            logger.info(reason)
            return False, reason
        for ch in cleaned_value:
            if "\u30a0" <= ch <= "\u30ff":
                continue
            if ch in KANJI_ALLOWED_PUNCTUATION:
                continue
            if ch in HIRAGANA_KATAKANA_ALLOWED_MARKS:
                continue
            if ch in SCRIPT_ALLOWED_ASCII_PUNCTUATION:
                continue
            if unicodedata.category(ch).startswith("S"):
                continue
            if ch in string.ascii_letters:
                continue
            if ch in string.digits:
                continue
            reason = (
                "[Katakana Character] Only katakana, punctuation, and symbols are allowed. "
                f"Invalid character: {ch}"
            )
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "全てカタカナ（数字・句読点・記号は可）で記述してください。",
            "カタカナのみを使って文章を書いてください（数字・句読点・記号は使用可）。",
            "ひらがなや漢字を避け、カタカナだけで表記してください（数字・記号は可）。",
            "カタカナ表記のみで回答してください（数字・記号可）。",
            "カタカナ以外の文字を含めないようにしてください（数字・記号は許可）。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "カタカナのみで記述してください。ひらがな・漢字は使用不可です（数字・記号可）。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "カタカナと数字、記号のみになるように書き換えてください。"


class KanjiCharacterConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        cleaned_value = _remove_whitespace(value)
        if not cleaned_value:
            reason = "[Kanji Character] Empty response."
            logger.info(reason)
            return False, reason
        for ch in cleaned_value:
            if "\u4e00" <= ch <= "\u9fff":
                continue
            if ch in KANJI_ALLOWED_PUNCTUATION:
                continue
            if ch in SCRIPT_ALLOWED_ASCII_PUNCTUATION:
                continue
            if unicodedata.category(ch).startswith("S"):
                continue
            if ch in string.ascii_letters:
                continue
            reason = (
                "[Kanji Character] Only kanji characters, punctuation, and symbols are allowed. "
                f"Invalid character: {ch}"
            )
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "全て漢字で記述してください（句読点と記号は可）。",
            "漢字のみを用いて文章を書いてください（記号は使用可）。",
            "ひらがなやカタカナを排除し、漢字だけで表記してください（記号は可）。",
            "漢字表記のみで回答してください（記号可）。",
            "漢字以外の文字が入らないようにしてください（記号は許可）。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "漢字のみで記述してください。ひらがな・カタカナは使用不可です（記号可）。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "漢字と記号のみになるように他の文字を置き換えてください。"


# Romaji syllables (Hebon-style, with some common extensions)
BASE_SYLLABLES = [
    # palatalized syllables
    "kya",
    "kyu",
    "kyo",
    "gya",
    "gyu",
    "gyo",
    "sha",
    "shu",
    "sho",
    "ja",
    "ju",
    "jo",
    "cha",
    "chu",
    "cho",
    "nya",
    "nyu",
    "nyo",
    "hya",
    "hyu",
    "hyo",
    "mya",
    "myu",
    "myo",
    "rya",
    "ryu",
    "ryo",
    "pya",
    "pyu",
    "pyo",
    "bya",
    "byu",
    "byo",
    # extra for foreign words
    "fa",
    "fi",
    "fu",
    "fe",
    "fo",
    # basic syllables (Hebon-style)
    "ka",
    "ki",
    "ku",
    "ke",
    "ko",
    "ga",
    "gi",
    "gu",
    "ge",
    "go",
    "sa",
    "shi",
    "su",
    "se",
    "so",
    "za",
    "ji",
    "zu",
    "ze",
    "zo",
    "ta",
    "chi",
    "tsu",
    "te",
    "to",
    "da",
    "de",
    "do",
    "na",
    "ni",
    "nu",
    "ne",
    "no",
    "ha",
    "hi",
    "fu",
    "he",
    "ho",
    "ba",
    "bi",
    "bu",
    "be",
    "bo",
    "pa",
    "pi",
    "pu",
    "pe",
    "po",
    "ma",
    "mi",
    "mu",
    "me",
    "mo",
    "ya",
    "yu",
    "yo",
    "ra",
    "ri",
    "ru",
    "re",
    "ro",
    "wa",
    "wo",
    # vowels
    "a",
    "i",
    "u",
    "e",
    "o",
]

SYLLABLES = sorted(set(BASE_SYLLABLES), key=len, reverse=True)

JP_SCRIPT_RE = re.compile(r"[\u3040-\u30FF\u4E00-\u9FFF]")

# Keep this strict (ASCII printable + whitespace + curly apostrophe) AFTER normalization.
ALLOWED_CHAR_PATTERN = r"[ -~\s’]"
ALLOWED_CHARS_RE = re.compile(ALLOWED_CHAR_PATTERN + "*")
ALLOWED_CHAR_RE_SINGLE = re.compile(ALLOWED_CHAR_PATTERN)

ASCII_LETTERS = set(string.ascii_letters)
VOWELS = set("aeiou")

# --- NEW: Unicode normalization for romaji-friendly punctuation/diacritics ---
_MACRON_MAP = str.maketrans(
    {
        "ā": "a",
        "Ā": "A",
        "ī": "i",
        "Ī": "I",
        "ū": "u",
        "Ū": "U",
        "ē": "e",
        "Ē": "E",
        "ō": "o",
        "Ō": "O",
    }
)

_HYPHENS = [
    "\u2010",  # hyphen
    "\u2011",  # non-breaking hyphen
    "\u2012",  # figure dash
    "\u2013",  # en dash
    "\u2014",  # em dash
    "\u2212",  # minus sign
]


def _normalize_romaji_text(text: str) -> str:
    # NFKC helps with some “compatibility” characters while generally keeping length 1:1 for most punctuation.
    t = unicodedata.normalize("NFKC", text)

    # Replace macron vowels with plain vowels (length info is ignored for validation)
    t = t.translate(_MACRON_MAP)

    # Normalize common unicode hyphens/dashes to ASCII hyphen
    for h in _HYPHENS:
        t = t.replace(h, "-")

    # Normalize curly quotes to ASCII quotes (optional but helps your sample)
    t = t.replace("“", '"').replace("”", '"')
    t = t.replace("‘", "'").replace("’", "'")  # keep ASCII apostrophe; it's allowed in [ -~]

    return t


def check_romaji(text: str) -> ConstraintEvaluation:
    """
    Return (True, None) if `text` is judged as valid Japanese written in strict romaji.
    Return (False, reason) if not, with a human-readable explanation.

    Notes:
      - This function now normalizes common Unicode punctuation and macron vowels
        (ō/ū/… -> o/u/…) before validation so that typical Hepburn-with-macrons text passes.
    """

    # --- NEW: normalize first ---
    norm = _normalize_romaji_text(text)

    # 1) Japanese scripts check
    if JP_SCRIPT_RE.search(norm):
        return False, "contains Japanese script characters (hiragana/katakana/kanji)"

    # 2) Allowed characters check (after normalization)
    if not ALLOWED_CHARS_RE.fullmatch(norm):
        invalid = sorted({ch for ch in norm if not ALLOWED_CHAR_RE_SINGLE.fullmatch(ch)})
        invalid_str = "".join(invalid)
        return False, f"contains disallowed characters: {invalid_str!r}"

    # 3) Extract letters only (A-Z, a-z) and index map (over normalized text)
    letters: list[str] = []
    index_map: list[int] = []
    for idx, ch in enumerate(norm):
        if ch in ASCII_LETTERS:
            letters.append(ch)
            index_map.append(idx)

    if not letters:
        return False, "does not contain any alphabetic characters"

    s = "".join(letters).lower()
    n = len(s)
    i = 0

    while i < n:
        ch = s[i]

        # --- Handle standalone 'n' as ん ---
        if ch == "n":
            if i == n - 1:
                i += 1
                continue
            if s[i + 1] not in VOWELS and s[i + 1] != "y":
                i += 1
                continue
            # else: could be part of "na/ni/nyu..." -> fall through

        # --- Handle geminate consonant (small っ) ---
        if i + 1 < n and s[i] == s[i + 1] and s[i] not in VOWELS and s[i] != "n":
            i += 1
            continue

        matched = False
        for syl in SYLLABLES:
            if s.startswith(syl, i):
                i += len(syl)
                matched = True
                break

        if not matched:
            orig_pos = index_map[i]
            context_start = max(0, orig_pos - 3)
            context_end = min(len(norm), orig_pos + 4)
            context = norm[context_start:context_end]
            offending = s[i : i + 3]
            return False, (
                "cannot parse as valid romaji syllable: "
                f"starts at '{offending}' (letters-only index {i}, "
                f"around normalized text '{context}' at position {orig_pos})"
            )

    return True, None


class FullWidthCharacterConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        cleaned_value = _remove_whitespace(value)
        match = re.search(r"[!-~]", cleaned_value)
        if match:
            offending = match.group(0)
            reason = (
                "[Full Width Character] Half-width ASCII characters detected. "
                f"Invalid character: {offending}"
            )
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "全て全角文字で記述してください。",
            "半角を使わず全角で表記してください。",
            "ASCIIの半角文字を含めないように全角で書いてください。",
            "文字はすべて全角にしてください。",
            "半角ではなく全角表記を徹底してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "文字をすべて全角で記述し、半角文字を混ぜないでください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "半角文字を全角に置き換えてください。"


class RomajiCharacterConstraint(ConstraintGroupMixin):
    """
    「ローマ字のみで書かれているか」をチェックする制約。

    - 日本語の文字(ひらがな / カタカナ / 漢字)は不可
    - 英字部分は Hebon-style のローマ字として分解可能であること
    - 記号や数字は許可するが、ローマ字判定の対象にはしない
    """

    def evaluate(self, value: str) -> ConstraintEvaluation:
        is_valid, reason = check_romaji(value)
        if not is_valid:
            reason_msg = f"[Romaji Character] Invalid romaji: {reason}"
            logger.info(reason_msg)
            return False, reason_msg
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "全てローマ字で記述してください。",
            "ローマ字表記のみを使って文章を書いてください。",
            "英字だけで日本語をローマ字表記してください。",
            "ローマ字以外の文字を含めないでください。",
            "ローマ字だけで文を作成してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "日本語の文字や英語を使わず、ヘボン式のローマ字のみで表記してください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "ローマ字以外の文字を取り除いてください。"


class HalfWidthCharacterConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        cleaned_value = _remove_whitespace(value)
        match = re.search(rf"[^{HALF_WIDTH_ALLOWED_PATTERN}]", cleaned_value)
        if match:
            offending = match.group(0)
            reason = (
                "[Half Width Character] Non half-width characters detected. "
                f"Invalid character: {offending}"
            )
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "全て半角文字で記述してください。",
            "全角を使わず半角のみで表記してください。",
            "非ASCII文字を含めず半角で書いてください。",
            "文字はすべて半角に統一してください。",
            "全角表記を避け、半角のみで記述してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "文字をすべて半角で記述し、全角文字を混ぜないでください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "全角文字を半角に置き換えてください。"
