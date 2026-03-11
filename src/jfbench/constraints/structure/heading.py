import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.llm import LLMClient
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class HeadingStructureConstraint(ConstraintGroupMixin):
    def __init__(self, client: LLMClient, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.client = client

    async def evaluate(self, value: str) -> ConstraintEvaluation:
        prompt = self._prompt(value)
        responses, _ = await self.client.async_ask([prompt])
        response = responses[0].strip()
        if response.startswith("False"):
            reason = (
                f"[Heading Structure] Output {value!r} does not have proper heading structure. "
                f"LLM response: {response!r}"
            )
            logger.info(reason)
            return False, reason
        if response.startswith("True"):
            return True, None
        reason = f"[Heading Structure] Unexpected LLM response {response!r} for value {value!r}."
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        body = (
            "\n各セクション見出しはタイトルの下に置き、段落は該当する見出しの配下にまとめてください。\n"
            "- タイトル: 文書全体を示す一行の見出し。\n"
            "- セクション見出し: 各セクションの要旨を表す見出し。\n"
            "- 段落: セクションごとの詳細をまとめる。\n"
        )
        templates = [
            "出力はタイトル→セクション見出し→段落の構造を持つ必要があります:" + body,
            "以下の構造ルールに沿って文書を整理してください:" + body,
            "タイトルと各セクション見出しを配置し、段落をその配下に置いてください:" + body,
            "文章は階層構造を守ってください。要件:" + body,
            "タイトル/見出し/段落の順序を守った構成にしてください:" + body,
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "見出しを含む構造で書いてください。" + body
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "タイトル→セクション見出し→関連段落の流れになるように出力を再構成してください。"

    def _prompt(self, text: str) -> str:
        return (
            "次のテキストが適切な文章構造（タイトル、セクション見出し、段落）を持っているか確認してください:\n"
            "適切な文章構造を持っていれば'True'、そうでなければ'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )
