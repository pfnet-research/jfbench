import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.llm import LLMClient
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class PastTenseStyleConstraint(ConstraintGroupMixin):
    def __init__(self, client: LLMClient, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.client = client

    async def evaluate(self, value: str) -> ConstraintEvaluation:
        prompt = self._prompt(value)
        responses, _ = await self.client.async_ask([prompt])
        response = responses[0].strip()
        if response.startswith("False"):
            reason = (
                f"[Past Tense Style] Output {value!r} is not in past tense. "
                f"LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        if response.startswith("True"):
            return True, None
        reason = f"[Past Tense Style] Unexpected LLM response {response!r} for value {value!r}."
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "文章全体を過去形で記述してください。",
            "すべての動詞を過去形にし、過去の出来事として述べてください。",
            "現在形や未来形を用いず、過去形表現のみで回答してください。",
            "記述は完了した事柄として過去形で統一してください。",
            "過去を語る形で文章をまとめ、時制を過去形に固定してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "過去形で書いてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "動詞や時制表現を過去形に統一してください。現在形や未来形の表現があれば過去形に書き換えてください。"

    def _prompt(self, text: str) -> str:
        return (
            "次のテキストに含まれる文が過去形であるか確認してください。"
            "評価の際はユーザーが与えた文章を除き、アシスタントが出力した部分のみを対象としてください。\n"
            "全て過去形の文であれば'True'、そうでなければ'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )


class PresentTenseStyleConstraint(ConstraintGroupMixin):
    def __init__(self, client: LLMClient, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.client = client

    async def evaluate(self, value: str) -> ConstraintEvaluation:
        prompt = self._prompt(value)
        responses, _ = await self.client.async_ask([prompt])
        response = responses[0].strip()
        if response.startswith("False"):
            reason = (
                f"[Present Tense Style] Output {value!r} is not in present tense. "
                f"LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        if response.startswith("True"):
            return True, None
        reason = f"[Present Tense Style] Unexpected LLM response {response!r} for value {value!r}."
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "文章は現在形で記述し、今起きている事柄として語ってください。",
            "すべての動詞を現在形にそろえてください。",
            "現在進行中・恒常的な状況を示す現在形のみを使用してください。",
            "現在形で統一し、過去・未来形の表現は避けてください。",
            "語尾や時制を現在形に合わせた文で回答してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "現在形で書いてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "動詞を現在形に統一し、現在の状況を述べる形へ書き直してください。"

    def _prompt(self, text: str) -> str:
        return (
            "次のテキストに含まれる文が現在形であるか確認してください。"
            "評価の際はユーザーが与えた文章を除き、アシスタントが出力した部分のみを対象としてください。\n"
            "全て現在形の文であれば'True'、そうでなければ'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )
