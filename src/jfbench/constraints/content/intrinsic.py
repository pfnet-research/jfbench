import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.llm import LLMClient
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class IntrinsicContentConstraint(ConstraintGroupMixin):
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
                f"[Intrinsic Content] Output {value!r} is not based solely on the provided document. "
                f"LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        if response.startswith("True"):
            return True, None
        reason = f"[Intrinsic Content] Unexpected LLM response {response!r} for value {value!r}."
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "出力は与えられた指示文の情報だけに基づいて記述してください。",
            "資料外の知識を持ち込まず、提示された指示文から得られる内容のみを書いてください。",
            "回答は参照指示文に依拠し、それ以外の情報源を含めないでください。",
            "指示文の内容だけを根拠にし、追加の推測や外部情報を含めないでください。",
            "テキストは資料の記述から逸脱しないよう、情報源を指示文のみに限定してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "指示文外の知識に基づく内容を混ぜずに書いてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "指示文以外の情報を取り除き、記述を資料の内容だけに基づく形へ書き直してください。"

    def _prompt(self, text: str) -> str:
        return (
            "与えられたテキストが次の指示文の内容にのみ基づいているか確認してください:\n"
            "テキストが指示文の内容にのみ基づいている場合は'True'、そうでない場合は'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}\n"
            "\n### 指示文\n"
            f"{self.document}"
        )
