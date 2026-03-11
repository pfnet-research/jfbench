import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.llm import LLMClient
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class KeywordExcludedContentConstraint(ConstraintGroupMixin):
    def __init__(self, keywords: list[str], *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.keywords = keywords

    def evaluate(self, value: str) -> ConstraintEvaluation:
        for keyword in self.keywords:
            if keyword in value:
                reason = (
                    f"[Keyword Excluded Content] Output {value!r} "
                    f"contains excluded keyword {keyword!r}."
                )
                logger.info(reason)
                return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        joined = "、".join(self.keywords)
        templates = [
            f"出力には禁止語（{joined}）を一切含めないでください。",
            f"文章中に{joined}が登場しないよう、該当語句は避けてください。",
            f"指定されたキーワード（{joined}）を使わずに表現してください。",
            f"{joined}といった語句は禁止なので、別の言い回しに置き換えてください。",
            f"回答には{joined}を含まないよう注意し、該当語は削除してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            joined = "、".join(self.keywords)
            return f"禁止語（{joined}）を使わないでください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        joined = "、".join(self.keywords)
        return f"禁止されたキーワード（{joined}）をすべて削除し、必要なら別の表現に置き換えてください。"


class AbstractExcludedContentConstraint(ConstraintGroupMixin):
    def __init__(
        self,
        client: LLMClient,
        document: str,
        content: str,
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.client = client
        self.document = document.rstrip("\r\n")
        self.content = content

    async def evaluate(self, value: str) -> ConstraintEvaluation:
        prompt = self._prompt(value)
        responses, _ = await self.client.async_ask([prompt])
        response = responses[0].strip()
        if response.startswith("False"):
            reason = (
                f"[Abstract Excluded Content] Output {value!r} includes the excluded content. "
                f"LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        if response.startswith("True"):
            return True, None
        reason = f"[Abstract Excluded Content] Unexpected LLM response {response!r} for value {value!r}."
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"出力には「{self.content}」に該当する内容を含めないでください。",
            f"回答文から{self.content}に触れる記述を排除してください。",
            f"指定された除外事項（{self.content}）を盛り込まずに記述してください。",
            f"{self.content}に関する情報は出さず、それ以外の内容だけを述べてください。",
            f"内容「{self.content}」を避け、別の話題のみで回答を構成してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return f"内容「{self.content}」に触れないでください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return f"出力から内容「{self.content}」に該当する部分を取り除いてください。"

    def _prompt(self, text: str) -> str:
        return (
            "与えられたテキストが次の指示文に関して指定された内容を含んでいないか確認してください:\n"
            "テキストが除外内容を含んでいない場合は'True'、"
            "そうでない場合は'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}\n"
            "\n### 指示文\n"
            f"{self.document}\n"
            "\n### 除外内容\n"
            f"{self.content}"
        )
