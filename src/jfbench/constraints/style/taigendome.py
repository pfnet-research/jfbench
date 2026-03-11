import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.llm import LLMClient
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class TaigendomeStyleConstraint(ConstraintGroupMixin):
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
                f"[Taigendome Style] Output {value!r} does not end sentences with nouns. "
                f"LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        reason = f"[Taigendome Style] Unexpected LLM response {response!r} for value {value!r}."
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "体言止めで文章を終えるようにしてください。",
            "文末を名詞で止める体言止めの形にしてください。",
            "各文の終わりを名詞で締めるように書いてください。",
            "終止を動詞にせず、名詞止めの文体にしてください。",
            "最後を体言止めで終わらせる形に整えてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "文末は体言止めにしてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "文末を名詞止めに書き換えてください。"

    def _prompt(self, text: str) -> str:
        return (
            "次のテキストの各文が体言止めになっているか確認してください。"
            "文末が動詞や形容詞、助動詞で終わらず名詞や名詞句で終わっていれば'True'、そうでなければ'False'と回答してください。"
            "評価の際はユーザーが与えた文章を除き、アシスタントが出力した部分のみを対象としてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )
