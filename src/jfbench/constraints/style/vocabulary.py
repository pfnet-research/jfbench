import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.llm import LLMClient
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class EasyVocabularyStyleConstraint(ConstraintGroupMixin):
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
                f"[Easy Vocabulary Style] Output {value!r} is not child-friendly. "
                f"LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        reason = (
            f"[Easy Vocabulary Style] Unexpected LLM response {response!r} for value {value!r}."
        )
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "子供向けのやさしい語彙のみを用いて記述してください。",
            "平易な単語だけで分かりやすく書いてください。",
            "難しい言葉を避け、簡単な語彙で説明してください。",
            "子どもでも読めるやさしい言葉で表現してください。",
            "専門的な語を避け、平易な単語に置き換えてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "やさしい語彙で書いてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "難しい単語を避け、簡単な語に置き換えてください。"

    def _prompt(self, text: str) -> str:
        return (
            "次のテキストが子どもでも理解できるやさしい語彙で書かれているか確認してください。"
            "評価の際はユーザーが与えた文章を除き、アシスタントが出力した部分のみを対象としてください。\n"
            "語彙が平易であれば'True'、難解な専門用語や長い単語が多ければ'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )


class DifficultVocacularyStyleConstraint(ConstraintGroupMixin):
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
                f"[Difficult Vocabulary Style] Output {value!r} is too simple. "
                f"LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        reason = f"[Difficult Vocabulary Style] Unexpected LLM response {response!r} for value {value!r}."
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "専門家向けの難しい語彙を用いて記述してください。",
            "高度な専門用語を交えて書いてください。",
            "平易さより専門性を優先し、難解な語彙を使ってください。",
            "専門的で高度な単語を選んで説明してください。",
            "専門家レベルの難易度の語彙を使用してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "難しい語彙で書いてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "より専門的で難易度の高い語彙に置き換えてください。"

    def _prompt(self, text: str) -> str:
        return (
            "次のテキストが専門家向けの難解な語彙を用いているか確認してください。"
            "評価の際はユーザーが与えた文章を除き、アシスタントが出力した部分のみを対象としてください。\n"
            "高度で専門的な単語が使われていれば'True'、平易で日常的な語彙が中心なら'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )
