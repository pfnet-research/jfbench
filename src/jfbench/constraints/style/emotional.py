import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.llm import LLMClient
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class JoyfulEmotionalStyleConstraint(ConstraintGroupMixin):
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
                f"[Joyful Emotional Style] Output {value!r} lacks a joyful tone. "
                f"LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        reason = (
            f"[Joyful Emotional Style] Unexpected LLM response {response!r} for value {value!r}."
        )
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "湧き上がる喜びを抑えられない嬉しそうな口調で答えてください。",
            "心から嬉しい雰囲気が伝わる語調で書いてください。",
            "喜びに満ちたテンションで表現してください。",
            "嬉しさを前面に出したトーンで回答してください。",
            "喜びを感じる言葉遣いで明るく答えてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "喜びが伝わる口調で書いてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "喜びが伝わる語を追加し、嬉しさを前面に出してください。"

    def _prompt(self, text: str) -> str:
        return (
            "次のテキストが喜びや嬉しさを感じさせる口調になっているか確認してください。"
            "評価の際はユーザーが与えた文章を除き、アシスタントが出力した部分のみを対象としてください。\n"
            "喜びが表れていれば'True'、そうでなければ'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )


class AngryEmotionalStyleConstraint(ConstraintGroupMixin):
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
                f"[Angry Emotional Style] Output {value!r} lacks an angry tone. "
                f"LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        reason = (
            f"[Angry Emotional Style] Unexpected LLM response {response!r} for value {value!r}."
        )
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "怒った口調で答えてください。",
            "憤りがにじむ語調で表現してください。",
            "苛立ちを示すトーンで回答してください。",
            "怒りを感じさせる言葉遣いで書いてください。",
            "腹立たしさが伝わるように怒り口調で述べてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "怒り口調で書いてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "怒りを示す語彙を用いて怒った調子にしてください。"

    def _prompt(self, text: str) -> str:
        return (
            "次のテキストが怒りや苛立ちを示す口調になっているか確認してください。"
            "評価の際はユーザーが与えた文章を除き、アシスタントが出力した部分のみを対象としてください。\n"
            "怒っている様子が伝われば'True'、それ以外の場合は'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )


class SadEmotionalStyleConstraint(ConstraintGroupMixin):
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
                f"[Sad Emotional Style] Output {value!r} lacks a sad tone. "
                f"LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        reason = f"[Sad Emotional Style] Unexpected LLM response {response!r} for value {value!r}."
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "悲しみに満ちた口調で答えてください。",
            "哀しさを帯びたトーンで表現してください。",
            "物悲しい語調で回答してください。",
            "悲しみが伝わる言葉遣いで書いてください。",
            "しんみりとした悲哀のこもった口調にしてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "悲しい口調で書いてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "悲しみを表す語を追加し、哀調にしてください。"

    def _prompt(self, text: str) -> str:
        return (
            "次のテキストが悲しみや哀愁を帯びた口調になっているか確認してください。"
            "評価の際はユーザーが与えた文章を除き、アシスタントが出力した部分のみを対象としてください。\n"
            "悲しみが感じられれば'True'、そうでなければ'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )


class HappyEmotionalStyleConstraint(ConstraintGroupMixin):
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
                f"[Happy Emotional Style] Output {value!r} lacks a happy tone. "
                f"LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        reason = (
            f"[Happy Emotional Style] Unexpected LLM response {response!r} for value {value!r}."
        )
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "幸せそうな口調で答えてください。",
            "幸福感に満ちたトーンで表現してください。",
            "嬉しさがにじむ穏やかな口調で書いてください。",
            "満ち足りた幸せを感じる語調で回答してください。",
            "幸せを感じさせる温かな言葉遣いで述べてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "幸せそうな口調で書いてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "幸福感が伝わる表現を増やしてください。"

    def _prompt(self, text: str) -> str:
        return (
            "次のテキストが穏やかな幸福感を示す口調になっているか確認してください。"
            "評価の際はユーザーが与えた文章を除き、アシスタントが出力した部分のみを対象としてください。\n"
            "幸せそうな雰囲気であれば'True'、そうでなければ'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )
