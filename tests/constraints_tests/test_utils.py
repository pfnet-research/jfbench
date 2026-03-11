from pytest import MonkeyPatch

import jfbench.constraints._utils as utils


def test_split_sentences_prefers_naive_when_more(monkeypatch: MonkeyPatch) -> None:
    class DummySegmenter:
        def segment(self, text: str) -> list[str]:
            return ["Only one segment"]

    monkeypatch.setattr(utils, "_load_segmenter", lambda: DummySegmenter())

    text = "First sentence. Second sentence."
    assert utils.split_sentences(text) == ["First sentence.", "Second sentence."]


def test_split_words_uses_tokenizer_and_filters(monkeypatch: MonkeyPatch) -> None:
    class DummyToken:
        def __init__(self, surface: str) -> None:
            self.surface = surface

    class DummyTokenizer:
        def tokenize(self, text: str) -> list[DummyToken]:
            return [DummyToken("Hello"), DummyToken(" "), DummyToken("123"), DummyToken("テスト")]

    monkeypatch.setattr(utils, "_load_tokenizer", lambda: DummyTokenizer())

    assert utils.split_words("ignored") == ["hello", "テスト"]


def test_split_words_falls_back_to_regex(monkeypatch: MonkeyPatch) -> None:
    class DummyTokenizer:
        def tokenize(self, text: str) -> list:
            return []

    monkeypatch.setattr(utils, "_load_tokenizer", lambda: DummyTokenizer())

    assert utils.split_words("Hello WORLD 123") == ["hello", "world"]
