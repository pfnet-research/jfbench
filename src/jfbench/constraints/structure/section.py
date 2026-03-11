import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.llm import LLMClient
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class SectionStructureConstraint(ConstraintGroupMixin):
    def __init__(self, client: LLMClient, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.client = client

    async def evaluate(self, value: str) -> ConstraintEvaluation:
        prompt = self._prompt(value)
        responses, _ = await self.client.async_ask([prompt])
        response = responses[0].strip()
        if response.startswith("False"):
            reason = (
                f"[Section Structure] Output {value!r} does not follow section structure. "
                f"LLM response: {response!r}"
            )
            logger.info(reason)
            return False, reason
        if response.startswith("True"):
            return True, None
        reason = f"[Section Structure] Unexpected LLM response {response!r} for value {value!r}."
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        body = "導入・本文・結論の各セクションを明示し、それぞれを区別してください。"
        templates = [
            "出力は導入→本文→結論の3セクション構造にしてください。" + body,
            "文章を導入/本文/結論の順番で構成し、各セクションを区切ってください。" + body,
            "導入・本文・結論を明示する構造に整えてください。" + body,
            "以下の3部構成を必ず守ってください: 導入・本文・結論。" + body,
            "セクションを3つに分け、それぞれの役割が分かるようにしてください。" + body,
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "導入・本文・結論のセクション構造で書いてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "導入→本文→結論のセクションが明示されるように段落や見出しを整理してください。"

    def _prompt(self, text: str) -> str:
        return (
            "次のテキストが適切なセクション構造（導入、本文、結論）を持っているか確認してください:\n"
            "適切なセクション構造を持っていれば'True'、そうでなければ'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )
