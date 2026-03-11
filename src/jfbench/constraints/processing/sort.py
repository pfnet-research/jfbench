import logging
import re

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints._utils import split_sentences
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


def _format_block(sentences: list[str]) -> str:
    return "[\n" + ",\n".join(sentences) + "\n]"


class DictionarySortProcessingConstraint(ConstraintGroupMixin):
    def __init__(self, document: str, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.document = document.rstrip("\r\n")

    def evaluate(self, value: str) -> ConstraintEvaluation:
        sentences = split_sentences(self.document)
        expected_block = _format_block(sorted(sentences))
        if expected_block not in value:
            reason = "[Dictionary Sort] Output does not include dictionary-sorted sentences."
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "与えられた指示文の文を辞書順にソートし、[\n<文1>,\n<文2>,\n…\n<文N>] 形式のリストを回答の中に含めてください。",
            "指示文の文章を辞書順に並べ替え、[\n<文1>,\n<文2>,\n…\n<文N>] の形のリストを出力に含めてください。",
            "指示文の各文を辞書順にソートしたリストを[\n<文1>,\n<文2>,\n…\n<文N>] の書式で回答中に含めて示してください。",
            "指示文の文を辞書順で整列させた[\n<文1>,\n<文2>,\n…\n<文N>] のリストを必ず含めてください。",
            "辞書順に並んだ指示文の文のリストを[\n<文1>,\n<文2>,\n…\n<文N>] 形式で本文に挿入してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "指示文の文を辞書順に並べたリストを[\n<文1>,\n<文2>,\n…\n<文N>] 形式で含めてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return (
            "指示文の各文を辞書順に並べ直したリスト([\n<文1>,...])を回答に含めてください。\n"
            "### 指示文\n"
            f"{self.document}"
        )


class LengthSortProcessingConstraint(ConstraintGroupMixin):
    def __init__(self, document: str, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.document = document.rstrip("\r\n")

    def evaluate(self, value: str) -> ConstraintEvaluation:
        sentences = split_sentences(self.document)
        expected_block = _format_block(sorted(sentences, key=len))
        if expected_block not in value:
            reason = "[Length Sort] Output does not include length-sorted sentences."
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "与えられた指示文の文を長さ順にソートし、[\n<文1>,\n<文2>,\n…\n<文N>] 形式のリストを回答に含めてください。",
            "指示文の各文を短い順に並べ替えた[\n<文1>,\n<文2>,\n…\n<文N>] のリストを出力に含めてください。",
            "長さ順に整列した指示文の文のリスト([\n<文1>,\n<文2>,\n…\n<文N>])を回答中に含めて示してください。",
            "指示文の文を文字数で昇順ソートした[\n<文1>,\n<文2>,\n…\n<文N>] を必ず含めてください。",
            "指示文の各文を短いものから長いものへ順番にした[\n<文1>,\n<文2>,\n…\n<文N>] リストを本文に挿入してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "指示文の文を長さ順に並べたリストを[\n<文1>,\n<文2>,\n…\n<文N>] 形式で含めてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return (
            "指示文の各文を長さ順に並べたリスト([\n<文1>,...])を回答に含めてください。\n"
            "### 指示文\n"
            f"{self.document}"
        )


class NumberSortProcessingConstraint(ConstraintGroupMixin):
    def __init__(self, document: str, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.document = document.rstrip("\r\n")

    def evaluate(self, value: str) -> ConstraintEvaluation:
        sentences = split_sentences(self.document)

        def _sentence_sum(sentence: str) -> int:
            numbers = re.findall(r"-?\d+", sentence)
            return sum(int(num) for num in numbers)

        expected_block = _format_block(sorted(sentences, key=_sentence_sum))
        if expected_block not in value:
            reason = "[Number Sort] Output does not include number-sorted sentences."
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "与えられた指示文の文を含まれる数字の和で昇順ソートし、[\n<文1>,\n<文2>,\n…\n<文N>] のリストを回答に含めてください。",
            "指示文の各文に含まれる数値の合計で並べ替えた[\n<文1>,\n<文2>,\n…\n<文N>] を出力に含めてください。",
            "数字の総和が小さい順に指示文の文を整列したリスト([\n<文1>,\n<文2>,\n…\n<文N>])を回答中に含めて示してください。",
            "指示文の文中の数字の和で昇順ソートした[\n<文1>,\n<文2>,\n…\n<文N>] の書式を必ず含めてください。",
            "数値合計を基準に指示文の文を並び替え、[\n<文1>,\n<文2>,\n…\n<文N>] 形式のリストを本文に挿入してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "指示文の文を含まれる数字の和で昇順に並べたリストを[\n<文1>,\n<文2>,\n…\n<文N>] 形式で含めてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return (
            "指示文の各文に含まれる数字の和でソートしたリスト([\n<文1>,...])を回答に含めてください。\n"
            "### 指示文\n"
            f"{self.document}"
        )
