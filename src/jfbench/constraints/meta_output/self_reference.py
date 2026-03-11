import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.llm import LLMClient
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class SelfReferenceConstraint(ConstraintGroupMixin):
    def __init__(self, client: LLMClient, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.client = client

    async def evaluate(self, value: str) -> ConstraintEvaluation:
        prompt = self._prompt(value)
        responses, _ = await self.client.async_ask([prompt])
        response = responses[0].strip()
        if response.startswith("False"):
            reason = (
                f"[Self Reference] Output {value!r} must include self-referential language. "
                f"LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        if response.startswith("True"):
            return True, None
        reason = f"[Self Reference] Unexpected LLM response {response!r} for value {value!r}."
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "自己言及的な内容を含めてください。",
            "回答の中で自分自身に触れてください。",
            "自分を指す表現を一文入れてください。",
            "自己言及を含める形で記述してください。",
            "「私」などの自己言及表現を入れてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "自己言及の語句を一度は入れてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "自己言及の文を追加してください。"

    def _prompt(self, text: str) -> str:
        return (
            "与えられたテキストに「I」「私」など自己言及する表現が含まれているか確認してください:\n"
            "テキストに自己言及表現が含まれている場合は'True'、含まれていない場合は'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )


class NoSelfReferenceConstraint(ConstraintGroupMixin):
    def __init__(self, client: LLMClient, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.client = client

    async def evaluate(self, value: str) -> ConstraintEvaluation:
        prompt = self._prompt(value)
        responses, _ = await self.client.async_ask([prompt])
        response = responses[0].strip()
        if response.startswith("False"):
            reason = (
                f"[No Self-Reference] Output {value!r} contains self-referential phrases. "
                f"LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        if response.startswith("True"):
            return True, None
        reason = f"[No Self-Reference] Unexpected LLM response {response!r} for value {value!r}."
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "出力に「私は」「この回答では」などの自己言及表現を含めないでください。",
            "文章は客観的に記述し、自分自身への言及やメタ発言は避けてください。",
            "自己紹介や回答者について触れる文を入れず、内容のみを書き表してください。",
            "自身を指す語句を排し、客観的・第三者視点の文章にしてください。",
            "回答中で話者や生成プロセスを語らず、対象の情報だけを述べてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "自己言及は入れずに書いてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "「私は～」「この回答では～」といった自己言及表現を削除し、内容のみに集中した文にしてください。"

    def _prompt(self, text: str) -> str:
        return (
            "与えられたテキストに「私は～」「この回答では～」といった自己言及的な表現が含まれていないか確認してください:\n"
            "テキストに自己言及的な表現が含まれていない場合は'True'、含まれている場合は'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )
