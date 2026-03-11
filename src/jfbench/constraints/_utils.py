from functools import lru_cache
import re

from janome.tokenizer import Tokenizer
import pysbd


_WORD_PATTERN = re.compile(r"[A-Za-z]+|[ぁ-ゔァ-ヴー々〆〤一-鿐]+")


@lru_cache(maxsize=1)
def _load_segmenter() -> pysbd.Segmenter:
    return pysbd.Segmenter(language="ja", clean=False)


@lru_cache(maxsize=1)
def _load_tokenizer() -> Tokenizer:
    return Tokenizer()


def split_sentences(text: str) -> list[str]:
    segmenter = _load_segmenter()
    sentences = [sentence.strip() for sentence in segmenter.segment(text) if sentence.strip()]
    naive_sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[。．\.！？!?])\s+", text)
        if sentence.strip()
    ]
    if len(naive_sentences) > len(sentences):
        return naive_sentences
    return sentences


def split_words(text: str) -> list[str]:
    tokenizer = _load_tokenizer()
    words: list[str] = []
    for token in tokenizer.tokenize(text):
        surface = token.surface.strip().lower()
        if not surface:
            continue
        if _WORD_PATTERN.fullmatch(surface):
            words.append(surface)
    if words:
        return words
    return _WORD_PATTERN.findall(text.lower())
