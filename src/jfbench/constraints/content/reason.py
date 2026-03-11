import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.llm import LLMClient
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class NoReasonContentConstraint(ConstraintGroupMixin):
    def __init__(self, client: LLMClient, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.client = client

    async def evaluate(self, value: str) -> ConstraintEvaluation:
        prompt = self._prompt(value)
        responses, _ = await self.client.async_ask([prompt])
        response = responses[0].strip()
        if response.startswith("False"):
            reason = (
                f"[No Reason Content] Output {value!r} includes reasoning or explanations. "
                f"LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        if response.startswith("True"):
            return True, None
        reason = f"[No Reason Content] Unexpected LLM response {response!r} for value {value!r}."
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "理由づけを含めずに回答してください。",
            "根拠や理由を述べずに結論だけを書いてください。",
            "因果関係の説明を避け、理由なしで答えてください。",
            "説明を加えずに結果だけを提示してください。",
            "理由や背景を書かずにシンプルに答えてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "結論だけ答えてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "理由表現を取り除き、説明なしの回答にしてください。"

    def _prompt(self, text: str) -> str:
        return (
            "与えられたテキストが理由付けや因果関係の説明を含まないか確認してください:\n"
            "テキストが理由付けを含まず結論のみの場合は'True'、そうでない場合は'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )


class ReasonContentConstraint(ConstraintGroupMixin):
    def __init__(
        self,
        client: LLMClient,
        document: str,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.client = client
        self.document = document.rstrip("\r\n")

    async def evaluate(self, value: str) -> ConstraintEvaluation:
        prompt = self._prompt(value)
        responses, _ = await self.client.async_ask([prompt])
        response = responses[0].strip()
        if response.startswith("False"):
            reason = (
                f"[Reason Content] Output {value!r} does not provide reasoning based on the document. "
                f"LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        if response.startswith("True"):
            return True, None
        reason = f"[Reason Content] Unexpected LLM response {response!r} for value {value!r}."
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "出力には指示文に基づく根拠や理由付けを必ず含めてください。",
            "回答には資料の内容を引用した理由説明を盛り込んでください。",
            "与えられた指示文に基づいた説明・根拠を文章中で述べてください。",
            "根拠となる情報を指示文から取り出して提示してください。",
            "指示文内容を参照した理由付けを文章内に明示してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "指示文に基づく根拠を示してください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "指示文の根拠を引用しながら理由付けを追加してください。"

    def _prompt(self, text: str) -> str:
        return (
            "与えられたテキストが次の指示文の内容に基づく理由付けを提供しているか確認してください:\n"
            "テキストが指示文の内容に基づく理由付けを提供している場合は'True'、そうでない場合は'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}\n"
            "\n### 指示文\n"
            f"{self.document}"
        )
