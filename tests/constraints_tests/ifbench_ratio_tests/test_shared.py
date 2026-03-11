from jfbench.constraints._utils import split_sentences
from jfbench.constraints._utils import split_words


def test_split_sentences_handles_basic_punctuation() -> None:
    text = "Hello world. Goodbye moon!"
    assert split_sentences(text) == ["Hello world.", "Goodbye moon!"]


def test_split_words_uses_janome_and_filters_symbols() -> None:
    text = "自然言語処理を学ぶ Hello, WORLD!"
    assert split_words(text) == ["自然", "言語", "処理", "を", "学ぶ", "hello", "world"]
