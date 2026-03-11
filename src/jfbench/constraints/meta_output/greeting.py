import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.llm import LLMClient
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class GreetingOutputConstraint(ConstraintGroupMixin):
    def __init__(self, client: LLMClient, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.client = client

    async def evaluate(self, value: str) -> ConstraintEvaluation:
        prompt = self._prompt(value)
        responses, _ = await self.client.async_ask([prompt])
        response = responses[0].strip()
        if response.startswith("False"):
            reason = (
                f"[Greeting Output] Output {value!r} must start with a greeting. "
                f"LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        if response.startswith("True"):
            return True, None
        reason = f"[Greeting Output] Unexpected LLM response {response!r} for value {value!r}."
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "回答の冒頭に挨拶を入れてください。",
            "最初に一言挨拶を添えてから本文を書いてください。",
            "冒頭で挨拶してから内容を続けてください。",
            "開始時に挨拶をしてから回答を始めてください。",
            "出だしに挨拶を必ず入れてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "冒頭に短い挨拶を一つ置いてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "最初の一文目に挨拶を追加してください。"

    def _prompt(self, text: str) -> str:
        return (
            "与えられたテキストの冒頭が挨拶から始まっているか確認してください:\n"
            "テキストが挨拶で始まっている場合は'True'、挨拶で始まっていない場合は'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )


class NoGreetingOutputConstraint(ConstraintGroupMixin):
    def __init__(self, client: LLMClient, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.client = client

    async def evaluate(self, value: str) -> ConstraintEvaluation:
        prompt = self._prompt(value)
        responses, _ = await self.client.async_ask([prompt])
        response = responses[0].strip()
        if response.startswith("False"):
            reason = (
                f"[No Greeting Output] Output {value!r} contains greeting phrases at the beginning. "
                f"LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        if response.startswith("True"):
            return True, None
        reason = f"[No Greeting Output] Unexpected LLM response {response!r} for value {value!r}."
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "出力の冒頭には挨拶や前置きを入れず、すぐ本題に入ってください。",
            "先頭での挨拶表現は禁止です。いきなり主題から記述を始めてください。",
            "「こんにちは」などの挨拶文を冒頭に置かず、ダイレクトに内容を書き出してください。",
            "冒頭部分は本題のみを述べ、前置きや挨拶語は付けないようにしてください。",
            "開始直後の挨拶・導入を省き、要件から記載を始める形で出力してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "挨拶抜きで本文から開始してください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "冒頭にある挨拶・前置きを削除し、本題の文章から始まるようにしてください。"

    def _prompt(self, text: str) -> str:
        return (
            "与えられたテキストの冒頭に挨拶や前置きが含まれておらずすぐに本題に入っているか確認してください。:\n"
            "テキストの冒頭に挨拶や前置きが含まれておらずすぐに本題に入っている場合は'True'、そうでない場合は'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )
