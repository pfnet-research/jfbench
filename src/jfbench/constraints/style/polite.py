import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.llm import LLMClient
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class PoliteStyleConstraint(ConstraintGroupMixin):
    def __init__(self, client: LLMClient, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.client = client

    async def evaluate(self, value: str) -> ConstraintEvaluation:
        prompt = self._prompt(value)
        responses, _ = await self.client.async_ask([prompt])
        response = responses[0].strip()
        if response.startswith("False"):
            reason = f"[Polite Style] Output {value!r} is not polite. LLM response: {response!r}."
            logger.info(reason)
            return False, reason
        if response.startswith("True"):
            return True, None
        reason = f"[Polite Style] Unexpected LLM response {response!r} for value {value!r}."
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "出力は敬体（です・ます調）の丁寧な言い回しで記述してください。",
            "丁寧語を用い、語尾を「です」「ます」で統一した文章にしてください。",
            "礼儀正しい敬体表現で回答してください。",
            "カジュアルな口調を避け、です・ます調の丁寧語で書いてください。",
            "丁寧語表現（敬体）を使った文章のみを出力してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "です・ます調で書いてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "語尾を「です・ます」調にそろえ、丁寧な表現へ書き直してください。"

    def _prompt(self, text: str) -> str:
        return (
            "次のテキストが丁寧な表現(敬体, です・ます調)であるか確認してください。"
            "評価の際はユーザーが与えた文章を除き、アシスタントが出力した部分のみを対象としてください。\n"
            "丁寧な表現であれば'True'、そうでなければ'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )
