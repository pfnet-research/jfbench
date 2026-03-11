import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.llm import LLMClient
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class FirstPersonSingularStyleConstraint(ConstraintGroupMixin):
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
                f"[First Person Singular] Output {value!r} lacks first-person singular. "
                f"LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        reason = (
            f"[First Person Singular] Unexpected LLM response {response!r} for value {value!r}."
        )
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "一人称として「私」を用いて記述してください。",
            "「私」を使った一人称で語ってください。",
            "一人称は「私」として文章を書いてください。",
            "自分を指すときは必ず「私」を使ってください。",
            "一人称表現に「私」を用いてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "一人称は私で書いてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "「私」を用いた一人称表現を追加してください。"

    def _prompt(self, text: str) -> str:
        return (
            "次のテキストに一人称単数の代名詞（日本語の「私」や英語のI）が明示的に含まれているか確認してください。"
            "評価の際はユーザーが与えた文章を除き、アシスタントが出力した部分のみを対象としてください。\n"
            "一人称単数が使われていれば'True'、含まれていなければ'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )


class FirstPersonPluralStyleConstraint(ConstraintGroupMixin):
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
                f"[First Person Plural] Output {value!r} lacks first-person plural. "
                f"LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        reason = f"[First Person Plural] Unexpected LLM response {response!r} for value {value!r}."
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "一人称として「われわれ（私たち）」を用いて記述してください。",
            "自分たちを指す際は「われわれ」や「私たち」を使ってください。",
            "一人称複数の代名詞（われわれ/私たち）で述べてください。",
            "「われわれ」「私たち」といった表現を使って書いてください。",
            "一人称複数を示す語を用いて記述してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "一人称複数で書いてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "「われわれ」や「私たち」などの一人称複数を入れてください。"

    def _prompt(self, text: str) -> str:
        return (
            "次のテキストに一人称複数の代名詞（日本語の「われわれ」「私たち」、英語のweなど）が含まれているか確認してください。"
            "評価の際はユーザーが与えた文章を除き、アシスタントが出力した部分のみを対象としてください。\n"
            "一人称複数が使われていれば'True'、含まれていなければ'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )


class ImpersonalStyleConstraint(ConstraintGroupMixin):
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
                f"[Impersonal Style] Output {value!r} contains first-person references. "
                f"LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        reason = f"[Impersonal Style] Unexpected LLM response {response!r} for value {value!r}."
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "一人称表現を避け、中立的に記述してください。",
            "「私」「われわれ」などの一人称を使わずに書いてください。",
            "主観的な一人称を排し、非人称で表現してください。",
            "一人称代名詞を含めず客観的に述べてください。",
            "自分を指す語を使わない中立的な文体にしてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "一人称を避けて書いてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "一人称の語を除き、中立的な文にしてください。"

    def _prompt(self, text: str) -> str:
        return (
            "次のテキストが一人称の代名詞を含まない非人称の文体になっているか確認してください。"
            "日本語の「私」「われわれ」「私たち」や英語のI, weなどが含まれていない場合は'True'、含まれている場合は'False'と答えてください。"
            "評価の際はユーザーが与えた文章を除き、アシスタントが出力した部分のみを対象としてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )
