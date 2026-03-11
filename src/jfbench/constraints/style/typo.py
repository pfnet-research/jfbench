import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.llm import LLMClient
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class NoTypoStyleConstraint(ConstraintGroupMixin):
    def __init__(self, client: LLMClient, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.client = client

    async def evaluate(self, value: str) -> ConstraintEvaluation:
        prompt = self._prompt(value)
        responses, _ = await self.client.async_ask([prompt])
        response = responses[0].strip()
        if response.startswith("False"):
            reason = (
                f"[No Typo Style] Output {value!r} contains typographical errors. "
                f"LLM response: {response!r}"
            )
            logger.info(reason)
            return False, reason
        if response.startswith("True"):
            return True, None
        reason = f"[No Typo Style] Unexpected LLM response {response!r} for value {value!r}."
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "誤字脱字のない丁寧な文章を出力してください。",
            "スペルミスやタイプミスがないよう慎重に記述してください。",
            "表記ゆれや誤字を取り除いた清書済みの文章を提出してください。",
            "誤記を避け、正確な文字列のみで構成してください。",
            "脱字・誤字が残らないよう校正した上で出力してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "誤字脱字のない文章にしてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "誤字脱字をすべて修正し、自然な文章になるように書き直してください。"

    def _prompt(self, text: str) -> str:
        return (
            "次のテキストのLLMが出力した部分に誤字脱字が含まれていないか確認してください。"
            "評価の際はユーザーが与えた文章を除き、アシスタントが出力した部分のみを対象としてください。\n"
            "誤字脱字が含まれていなければ'True'、含まれていれば'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )
