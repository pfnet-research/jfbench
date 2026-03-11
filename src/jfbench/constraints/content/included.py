import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.llm import LLMClient
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class KeywordIncludedContentConstraint(ConstraintGroupMixin):
    def __init__(self, keywords: dict[str, int], *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.keywords = keywords

    def evaluate(self, value: str) -> ConstraintEvaluation:
        for keyword, count in self.keywords.items():
            actual_count = value.count(keyword)
            if actual_count != count:
                reason = (
                    "[Keyword Included Content] Output "
                    f"{value!r} contains keyword {keyword!r} {actual_count} times; "
                    f"expected {count} times."
                )
                logger.info(reason)
                return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        keyword_desc = "、".join(f"{k}を{v}回" for k, v in self.keywords.items())
        templates = [
            f"出力には{keyword_desc}が現れるよう、それぞれ指定回数を守ってください。",
            f"各キーワード（{keyword_desc}）を欠かさず、指示された回数だけ使用してください。",
            f"文章中で{keyword_desc}を必ず満たし、回数不足や超過がないよう調整してください。",
            f"{keyword_desc}が正確な出現回数で含まれるよう文を構成してください。",
            f"指定語句（{keyword_desc}）を示された回数分だけ盛り込み、条件を守ってください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return f"指定語句（{keyword_desc}）がちょうど所定回数になるようにしてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        keyword_desc = "、".join(f"{k}を{v}回" for k, v in self.keywords.items())
        return (
            f"各キーワードの指定回数（{keyword_desc}）が守られるように出現回数を調整してください。"
        )


class AbstractIncludedContentConstraint(ConstraintGroupMixin):
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
                f"[Abstract Included Content] Output {value!r} does not include the specified "
                f"content. LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        if response.startswith("True"):
            return True, None
        reason = (
            f"[Abstract Included Content] Unexpected LLM response {response!r} for value "
            f"{value!r}."
        )
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"出力には必ず「{self.content}」の内容を明示的に盛り込んでください。",
            f"文章中で{self.content}に触れ、その情報が含まれる形で回答してください。",
            f"指定された内容（{self.content}）を欠かさず入れてから出力してください。",
            f"{self.content}を出力内で取り上げ、その要素を明言する形にしてください。",
            f"回答のどこかに{self.content}を含め、要求された内容が伝わるようにしてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return f"回答には「{self.content}」を明確に含めてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return f"出力を修正し、内容「{self.content}」が明示的に含まれるようにしてください。"

    def _prompt(self, text: str) -> str:
        return (
            "次のテキストが指示文に関する指定された内容を含んでいるか確認してください:\n"
            "テキストが内容を含んでいる場合は'True'、そうでない場合は'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}\n"
            "\n### 指示文\n"
            f"{self.document}\n"
            "\n### 内容\n"
            f"{self.content}"
        )
