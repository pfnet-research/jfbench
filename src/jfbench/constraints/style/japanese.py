import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.llm import LLMClient
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class JapaneseStyleConstraint(ConstraintGroupMixin):
    def __init__(self, client: LLMClient, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.client = client
        self._excuse = "ただし、専門用語や固有名詞など、どうしても日本語に置き換えられない場合は例外とします。また記号や絵文字の使用も許容されます。"

    async def evaluate(self, value: str) -> ConstraintEvaluation:
        if not value:
            reason = "[Japanese Style] Empty value provided."
            logger.info(reason)
            return False, reason
        prompt = self._prompt(value)
        responses, _ = await self.client.async_ask([prompt])
        response = responses[0].strip()
        if response.startswith("True"):
            return True, None
        if response.startswith("False"):
            reason = (
                f"[Japanese Style] Output {value!r} is not recognized as Japanese. "
                f"LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        reason = f"[Japanese Style] Unexpected LLM response {response!r} for value {value!r}."
        logger.info(reason)
        return False, reason

    def _prompt(self, text: str) -> str:
        return (
            "次のテキストが自然な日本語のみで書かれているか判定してください。"
            "評価の際はユーザーが与えた文章を除き、アシスタントが出力した部分のみを対象としてください。"
            f"{self._excuse}"
            "日本語のみで構成されていれば'True'、そうでなければ'False'と回答してください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "出力はすべて自然な日本語で記述してください。",
            "日本語以外の表現を含めず、日本語のみで回答してください。",
            "文章全体を日本語に統一し、他言語は避けてください。",
            "英数字を必要最小限にし、日本語の語彙を中心に書いてください。",
            "日本語だけで構成された文章を返してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates) + self._excuse
        if train_or_test == "test":
            return "日本語だけで書いてください。" + self._excuse
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "英字など他言語が混在している場合は日本語に置き換えてください。"


class NoJapaneseStyleConstraint(ConstraintGroupMixin):
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
                f"[No Japanese Style] Output {value!r} contains Japanese elements. "
                f"LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        reason = f"[No Japanese Style] Unexpected LLM response {response!r} for value {value!r}."
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "回答が日本語にならないようにしてください。",
            "日本語の文章を避け、別の言語で記述してください。",
            "日本語表現を使わずに答えてください。",
            "日本語にならないよう言語を選んでください。",
            "日本語の文字を含めない形で回答してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "日本語を使わずに書いてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "日本語の文字を含めないようにしてください。"

    def _prompt(self, text: str) -> str:
        return (
            "次のテキストが日本語の文字や表現を含んでいないか確認してください。"
            "評価の際はユーザーが与えた文章を除き、アシスタントが出力した部分のみを対象としてください。\n"
            "日本語が含まれていなければ'True'、含まれていれば'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )
