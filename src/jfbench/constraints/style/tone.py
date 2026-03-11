import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.llm import LLMClient
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class FormalToneStyleConstraint(ConstraintGroupMixin):
    def __init__(self, client: LLMClient, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.client = client

    async def evaluate(self, value: str) -> ConstraintEvaluation:
        prompt = self._prompt(value)
        responses, _ = await self.client.async_ask([prompt])
        response = responses[0].strip()
        if response.startswith("False"):
            reason = (
                f"[Formal Tone Style] Output {value!r} is not in formal tone. "
                f"LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        if response.startswith("True"):
            return True, None
        reason = f"[Formal Tone Style] Unexpected LLM response {response!r} for value {value!r}."
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "文章は格式あるフォーマルなトーンで記述してください。",
            "くだけた表現を避け、丁寧かつ正式な語調で書いてください。",
            "公的文書のようなフォーマルスタイルで表現してください。",
            "改まった敬語を用いたフォーマルトーンに統一してください。",
            "礼儀正しく硬めの表現で、フォーマルな雰囲気を維持してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "フォーマルなトーンで書いてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "丁寧で格式ある表現に言い換え、フォーマルな文章に整えてください。"

    def _prompt(self, text: str) -> str:
        return (
            "次のテキストがフォーマルなトーンであるか確認してください。"
            "評価の際はユーザーが与えた文章を除き、アシスタントが出力した部分のみを対象としてください。\n"
            "フォーマルなトーンであれば'True'、そうでなければ'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )


class CasualToneStyleConstraint(ConstraintGroupMixin):
    def __init__(self, client: LLMClient, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.client = client

    async def evaluate(self, value: str) -> ConstraintEvaluation:
        prompt = self._prompt(value)
        responses, _ = await self.client.async_ask([prompt])
        response = responses[0].strip()
        if response.startswith("False"):
            reason = (
                f"[Casual Tone Style] Output {value!r} is not in casual tone. "
                f"LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        if response.startswith("True"):
            return True, None
        reason = f"[Casual Tone Style] Unexpected LLM response {response!r} for value {value!r}."
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "フランクで親しみやすいカジュアルなトーンで書いてください。",
            "堅苦しさを避け、くだけた口調で表現してください。",
            "友人に話すような軽やかなトーンで文章を仕上げてください。",
            "ラフで柔らかな言い回しを使い、カジュアルな雰囲気を出してください。",
            "日常会話に近い砕けた口調で出力してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "カジュアルなトーンで書いてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return (
            "くだけた口調や親しみのある表現に言い換えて、カジュアルなトーンへ書き直してください。"
        )

    def _prompt(self, text: str) -> str:
        return (
            "次のテキストがカジュアルなトーンであるか確認してください。"
            "評価の際はユーザーが与えた文章を除き、アシスタントが出力した部分のみを対象としてください。\n"
            "カジュアルなトーンであれば'True'、そうでなければ'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )


class AcademicToneStyleConstraint(ConstraintGroupMixin):
    def __init__(self, client: LLMClient, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.client = client

    async def evaluate(self, value: str) -> ConstraintEvaluation:
        prompt = self._prompt(value)
        responses, _ = await self.client.async_ask([prompt])
        response = responses[0].strip()
        if response.startswith("False"):
            reason = (
                f"[Academic Tone Style] Output {value!r} is not in academic tone. "
                f"LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        if response.startswith("True"):
            return True, None
        reason = f"[Academic Tone Style] Unexpected LLM response {response!r} for value {value!r}."
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "学術論文のように客観的で専門的なトーンで記述してください。",
            "研究報告に近い厳密で学術的な語調を用いてください。",
            "エビデンスを重視した学術スタイルで文章を組み立ててください。",
            "専門用語を適切に用いつつ、学術的な文体で書いてください。",
            "中立かつ論理的な語調で、学術トーンを維持してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "学術的なトーンで書いてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "専門的で客観的な語調に整え、学術論文風の表現へ書き換えてください。"

    def _prompt(self, text: str) -> str:
        return (
            "次のテキストが学術的なトーンであるか確認してください。"
            "評価の際はユーザーが与えた文章を除き、アシスタントが出力した部分のみを対象としてください。\n"
            "学術的なトーンであれば'True'、そうでなければ'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )


class BusinessToneStyleConstraint(ConstraintGroupMixin):
    def __init__(self, client: LLMClient, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.client = client

    async def evaluate(self, value: str) -> ConstraintEvaluation:
        prompt = self._prompt(value)
        responses, _ = await self.client.async_ask([prompt])
        response = responses[0].strip()
        if response.startswith("False"):
            reason = (
                f"[Business Tone Style] Output {value!r} is not in business tone. "
                f"LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        if response.startswith("True"):
            return True, None
        reason = f"[Business Tone Style] Unexpected LLM response {response!r} for value {value!r}."
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "ビジネス文書らしい端的で礼儀正しいトーンで記載してください。",
            "実務的かつ丁寧なビジネスライクな語調でまとめてください。",
            "社内外への案内文のようなビジネス調で文章を作成してください。",
            "敬意を保ちつつも簡潔なビジネストーンを維持してください。",
            "業務連絡にふさわしい落ち着いたビジネス文体で記述してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "ビジネス文書のトーンで書いてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "丁寧さと実務的な語尾を意識して、ビジネス文書にふさわしいトーンへ修正してください。"

    def _prompt(self, text: str) -> str:
        return (
            "次のテキストがビジネスライクなトーンであるか確認してください。"
            "評価の際はユーザーが与えた文章を除き、アシスタントが出力した部分のみを対象としてください。\n"
            "ビジネスライクなトーンであれば'True'、そうでなければ'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )
