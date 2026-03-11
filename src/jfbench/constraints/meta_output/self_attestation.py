import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.llm import LLMClient
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class NoSelfAttestationConstraint(ConstraintGroupMixin):
    def __init__(self, client: LLMClient, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.client = client

    async def evaluate(self, value: str) -> ConstraintEvaluation:
        prompt = self._prompt(value)
        responses, _ = await self.client.async_ask([prompt])
        response = responses[0].strip()
        if response.startswith("False"):
            reason = (
                f"[No Self Attestation] Output {value!r} contains self-attestation phrases. "
                f"LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        if response.startswith("True"):
            return True, None
        reason = f"[No Self Attestation] Unexpected LLM response {response!r} for value {value!r}."
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "出力には、指定された制約をすべて満たしたと明言する一文を加えないようにしてください。",
            "回答内で「全条件に従った」など、要件を満たした旨の自己宣言を入れないでください。",
            "提示された制約に合致していると自ら確認するフレーズを本文に含めないでください。",
            "最終文などで、この回答が全制約を遵守していると明確に述べないようにしてください。",
            "結果には、上記条件を満たしたと断言する自己証明の文を添えないでください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "制約を満たしたと明言する文は入れないでください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "制約を満たしたと宣言する文言を削除してください。"

    def _prompt(self, text: str) -> str:
        return (
            "与えられたテキストに「全ての制約を満たした」といった自己証明の文言が含まれていないか確認してください:\n"
            "テキストにそのような自己証明が含まれていない場合は'True'、含まれている場合は'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )


class SelfAttestationConstraint(ConstraintGroupMixin):
    def __init__(self, client: LLMClient, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.client = client

    async def evaluate(self, value: str) -> ConstraintEvaluation:
        prompt = self._prompt(value)
        responses, _ = await self.client.async_ask([prompt])
        response = responses[0].strip()
        if response.startswith("False"):
            reason = (
                f"[Self Attestation] Output {value!r} does not include self-attestation "
                f"statements. LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        if response.startswith("True"):
            return True, None
        reason = f"[Self Attestation] Unexpected LLM response {response!r} for value {value!r}."
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "出力のどこかに、指定された制約をすべて満たしたと明言する一文を加えてください。",
            "回答内で「全条件に従った」など、要件を満たした旨の自己宣言を入れてください。",
            "提示された制約に合致していると自ら確認するフレーズを本文に含めてください。",
            "最終文などで、この回答が全制約を遵守していると明確に述べてください。",
            "結果には、上記条件を満たしたと断言する自己証明の文を必ず添えてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "制約を満たしたと明言する文を含めてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "制約を満たしていると明言する自己宣言の一文を追記してください。"

    def _prompt(self, text: str) -> str:
        return (
            "与えられたテキストが与えられた制約を全て満たしていることを明言する文言を含んでいるか確認してください:\n"
            "テキストが与えられた制約を全て満たしていることを明言する文言を含んでいる場合は'True'、含んでいない場合は'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )
