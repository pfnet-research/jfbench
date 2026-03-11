import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.llm import LLMClient
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class PlainStyleConstraint(ConstraintGroupMixin):
    def __init__(self, client: LLMClient, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.client = client

    async def evaluate(self, value: str) -> ConstraintEvaluation:
        prompt = self._prompt(value)
        responses, _ = await self.client.async_ask([prompt])
        response = responses[0].strip()
        if response.startswith("True"):
            return True, None
        if response.startswith("False"):
            reason = (
                f"[Plain Style] Output {value!r} is not in plain form. LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        reason = f"[Plain Style] Unexpected LLM response {response!r} for value {value!r}."
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "回答の文章を常体（だ・である調）で書いてください。",
            "です・ます調ではなく常体で記述してください。",
            "終止形を用いて常体の文体にしてください。",
            "丁寧語を避け、だ・である調で書いてください。",
            "常体の文体で統一し、ですます調にしないでください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "常体で書いてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "です・ます調を避け、常体に書き直してください。"

    def _prompt(self, text: str) -> str:
        return (
            "次のテキストが常体（だ・である調）で書かれており、敬体（です・ます調）を含んでいないか確認してください。"
            "評価の際はユーザーが与えた文章を除き、アシスタントが出力した部分のみを対象としてください。\n"
            "常体だけで書かれていれば'True'、敬体表現が混じっていれば'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )
