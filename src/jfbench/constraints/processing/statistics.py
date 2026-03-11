import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.llm import LLMClient
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class StatisticsProcessingConstraint(ConstraintGroupMixin):
    def __init__(
        self,
        client: LLMClient,
        document: str,
        statistic: str,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.document = document.rstrip("\r\n")
        self.statistic = statistic
        self.client = client

    async def evaluate(self, value: str) -> ConstraintEvaluation:
        prompt = self._prompt(value)
        responses, _ = await self.client.async_ask([prompt])
        response = responses[0].strip()
        if response.startswith("False"):
            reason = (
                f"[Statistics Processing] Output {value!r} does not contain statistic "
                f"{self.statistic!r}. LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        if response.startswith("True"):
            return True, None
        reason = (
            f"[Statistics Processing] Unexpected LLM response {response!r} for value "
            f"{value!r} under statistic {self.statistic!r}."
        )
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"出力に統計量「{self.statistic}」の値を含め、指示文を元に計算した結果を示してください。",
            f"指示文から得られる統計値「{self.statistic}」を明記してください。",
            f"回答には統計量{self.statistic}に相当する数値を必ず含めてください。",
            f"資料から計算した{self.statistic}の値を出力に盛り込んでください。",
            f"{self.statistic}の統計を算出し、その数値を本文に記載してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return f"指示文における統計量「{self.statistic}」の値を入れてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return f"出力を見直し、統計量「{self.statistic}」が正しく反映された数値を含めてください。"

    def _prompt(self, text: str) -> str:
        return (
            "テキストが指示文に対して計算された統計量を含むか確認してください:\n"
            "統計量を含む場合は'True'、満たさない場合は'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}\n"
            "\n### 指示文\n"
            f"{self.document}\n"
            "\n### 統計量\n"
            f"{self.statistic}"
        )
