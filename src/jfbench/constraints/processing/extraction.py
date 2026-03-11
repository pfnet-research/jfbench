import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.llm import LLMClient
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class PrefixExtractionProcessingConstraint(ConstraintGroupMixin):
    def __init__(self, document: str, length: int, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.document = document.rstrip("\r\n")
        self.length = length

    def evaluate(self, value: str) -> ConstraintEvaluation:
        prefix = self.document[: self.length]
        if prefix not in value:
            reason = "[Prefix Extraction] Required prefix fragment is missing from the output."
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"与えられた指示文の先頭{self.length}文字を抽出して回答に含めてください。",
            f"指示文の冒頭{self.length}文字を取り出し、出力に載せてください。",
            f"回答にはテキストの最初の{self.length}文字を必ず入れてください。",
            f"指定指示文の先頭から{self.length}文字を抜粋して示してください。",
            f"最初の{self.length}文字を切り出してレスポンスに含めてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return f"指示文の先頭{self.length}文字を入れてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return f"指示文先頭{self.length}文字を引用するように修正してください。"


class SuffixExtractionProcessingConstraint(ConstraintGroupMixin):
    def __init__(self, document: str, length: int, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.document = document.rstrip("\r\n")
        self.length = length

    def evaluate(self, value: str) -> ConstraintEvaluation:
        suffix = self.document[-self.length :]
        if suffix not in value:
            reason = "[Suffix Extraction] Required suffix fragment is missing from the output."
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"与えられた指示文の末尾{self.length}文字を抽出して回答に含めてください。",
            f"指示文の最後の{self.length}文字を切り出して出力してください。",
            f"回答にはテキスト末尾{self.length}文字を必ず含めてください。",
            f"末尾から{self.length}文字分を抜き出し、レスポンスに載せてください。",
            f"与えられた文章の終わり{self.length}文字を取り込み回答してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return f"指示文の末尾{self.length}文字を入れてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return f"指示文末尾{self.length}文字を含めるようにしてください。"


class RangeExtractionProcessingConstraint(ConstraintGroupMixin):
    def __init__(self, document: str, start: int, end: int, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.document = document.rstrip("\r\n")
        self.start = start
        self.end = end

    def evaluate(self, value: str) -> ConstraintEvaluation:
        fragment = self.document[self.start : self.end + 1]
        if fragment not in value:
            reason = "[Range Extraction] Required range fragment is missing from the output."
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"指示文の{self.start}文字目から{self.end}文字目までを抽出して回答に含めてください。",
            f"{self.start}文字目〜{self.end}文字目の範囲を切り出して出力してください。",
            f"指定区間（{self.start}〜{self.end}文字目）の文字列を必ず含めてください。",
            f"テキストの{self.start}文字目から{self.end}文字目までを抜粋し回答に載せてください。",
            f"対象範囲{self.start}〜{self.end}文字目をそのまま取り入れてください。",
        ]
        if train_or_test == "train":
            return (
                self._random_instruction(templates)
                + "ただし、最初の文字を0文字目と数えてください。"
            )
        if train_or_test == "test":
            return f"指示文の{self.start}文字目から{self.end}文字目を含めてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return f"指定範囲（{self.start}文字目から{self.end}文字目）を含めてください。"


class ExtractionProcessingConstraint(ConstraintGroupMixin):
    def __init__(
        self,
        client: LLMClient,
        document: str,
        condition: str,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.document = document.rstrip("\r\n")
        self.condition = condition
        self.client = client

    async def evaluate(self, value: str) -> ConstraintEvaluation:
        prompt = self._prompt(value)
        responses, _ = await self.client.async_ask([prompt])
        response = responses[0].strip()
        if response.startswith("False"):
            reason = (
                f"[Extraction Processing] Extracted value {value!r} does not satisfy condition {self.condition!r}. "
                f"LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        if response.startswith("True"):
            return True, None
        reason = (
            f"[Extraction Processing] Unexpected LLM response {response!r} for value "
            f"{value!r} under condition {self.condition!r}."
        )
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"指示文から条件「{self.condition}」を満たす部分を抽出し、そのテキストを出力に含めてください。",
            f"条件{self.condition}に合致する指示文内の記述を取り出して提示してください。",
            f"資料中で「{self.condition}」を満たすテキストを抜き出し、回答に盛り込んでください。",
            f"{self.condition}の条件にかなう箇所を抽出し、それを出力内容にしてください。",
            f"指示文の中から条件「{self.condition}」に一致するテキストのみを結果に含めてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return f"指示文から条件「{self.condition}」を満たす部分を抜き出して含めてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return f"指示文から条件「{self.condition}」に合致する部分を抜き出して出力を置き換えてください。"

    def _prompt(self, text: str) -> str:
        return (
            "入力文字列に、指示文から抽出された条件を満たすテキストが含まれているか確認してください。\n"
            "条件を満たす場合は'True'、満たさない場合は'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}\n"
            "\n### 指示文\n"
            f"{self.document}\n"
            "\n### 条件\n"
            f"{self.condition}"
        )
