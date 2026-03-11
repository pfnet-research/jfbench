import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.llm import LLMClient
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class ExplanationConstraint(ConstraintGroupMixin):
    def __init__(self, client: LLMClient, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.client = client

    async def evaluate(self, value: str) -> ConstraintEvaluation:
        prompt = self._prompt(value)
        responses, _ = await self.client.async_ask([prompt])
        response = responses[0].strip()
        if response.startswith("False"):
            reason = (
                f"[Explanation] Output {value!r} is missing a brief explanation at the end. "
                f"LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        if response.startswith("True"):
            return True, None
        reason = f"[Explanation] Unexpected LLM response {response!r} for value {value!r}."
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "回答の末尾に簡潔な説明を付け加えてください。",
            "最後に説明文を添えてください。",
            "回答の終わりに理由や説明を入れてください。",
            "末尾に解説を一文追加してください。",
            "締めくくりとして説明を加筆してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "末尾に短い説明文を付けてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "回答の最後に説明文を追記してください。"

    def _prompt(self, text: str) -> str:
        return (
            "与えられたテキストの末尾に回答の理由や説明が含まれているか確認してください:\n"
            "テキストの末尾に簡潔な説明が含まれている場合は'True'、含まれていない場合は'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )


class NoExplanationConstraint(ConstraintGroupMixin):
    def __init__(self, client: LLMClient, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.client = client

    async def evaluate(self, value: str) -> ConstraintEvaluation:
        prompt = self._prompt(value)
        responses, _ = await self.client.async_ask([prompt])
        response = responses[0].strip()
        if response.startswith("False"):
            reason = (
                f"[No Explanation] Output {value!r} contains explanations or justifications at "
                f"the end. LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        if response.startswith("True"):
            return True, None
        reason = f"[No Explanation] Unexpected LLM response {response!r} for value {value!r}."
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "出力の末尾に説明や言い訳を加えず、本題のみで終わらせてください。",
            "最後の段落には解説文や自己弁護を挿入せずに締めてください。",
            "文章は回答だけで完結させ、余計な説明や正当化を付けないでください。",
            "締め括りに補足説明を足すのは禁止です。指定された回答のみで結びましょう。",
            "回答末尾はシンプルにまとめ、理由や弁明は含めないでください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "最後に説明文を付けずに終えてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "出力の末尾から説明や自己弁護の文を削除し、指示された本体のみを残してください。"

    def _prompt(self, text: str) -> str:
        return (
            "与えられたテキストの最後に説明や正当化が含まれていないか確認してください:\n"
            "テキストの最後に説明や正当化が含まれていない場合は'True'、含まれている場合は'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )
