import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.llm import LLMClient
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class EnglishStyleConstraint(ConstraintGroupMixin):
    def __init__(self, client: LLMClient, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.client = client

    async def evaluate(self, value: str) -> ConstraintEvaluation:
        if not value:
            reason = "[English Style] Empty value provided."
            logger.info(reason)
            return False, reason
        prompt = self._prompt(value)
        responses, _ = await self.client.async_ask([prompt])
        response = responses[0].strip()
        if response.startswith("True"):
            return True, None
        if response.startswith("False"):
            reason = (
                f"[English Style] Output {value!r} is not recognized as English. "
                f"LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        reason = f"[English Style] Unexpected LLM response {response!r} for value {value!r}."
        logger.info(reason)
        return False, reason

    def _prompt(self, text: str) -> str:
        return (
            "次のテキストが自然な英語のみで書かれているか判定してください。"
            "評価の際はユーザーが与えた文章を除き、アシスタントが出力した部分のみを対象としてください。"
            "英語のみで構成されていれば'True'、そうでなければ'False'と回答してください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "出力はすべて自然な英語で記述してください。",
            "英語以外の言語を混ぜず、全文を英語で書いてください。",
            "回答は英語だけで完結させてください。",
            "英語以外の単語や句読点を含めないでください。",
            "英語のみを使用して文章を構成してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "英語だけで書いてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "英語以外の要素を取り除き、全体を英語表現に統一してください。"


class BritishEnglishStyleConstraint(ConstraintGroupMixin):
    def __init__(self, client: LLMClient, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.client = client

    async def evaluate(self, value: str) -> ConstraintEvaluation:
        prompt = self._prompt(value)
        responses, _ = await self.client.async_ask([prompt])
        response = responses[0].strip()
        if response.startswith("False"):
            reason = (
                f"[British English Style] Output {value!r} is not in British English. "
                f"LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        if response.startswith("True"):
            return True, None
        reason = (
            f"[British English Style] Unexpected LLM response {response!r} for value {value!r}."
        )
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "イギリス英語のスペルと語彙を用いて記述してください。",
            "出力は英国式英語で統一し、米語表現を避けてください。",
            "colour, organise などイギリス英語の綴りを採用して文章全体を記述してください。",
            "英国英語の文体でまとめ、米国式表現に置き換えないでください。",
            "イギリス英語の表現ルールで回答してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "イギリス英語で書いてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "スペルや語彙をイギリス英語に合わせて書き換えてください。"

    def _prompt(self, text: str) -> str:
        return (
            "次のテキストがイギリス英語で書かれているか確認してください。"
            "評価の際はユーザーが与えた文章を除き、アシスタントが出力した部分のみを対象としてください。\n"
            "イギリス英語で書かれている場合は'True'、そうでない場合は'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )


class AmericanEnglishStyleConstraint(ConstraintGroupMixin):
    def __init__(self, client: LLMClient, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.client = client

    async def evaluate(self, value: str) -> ConstraintEvaluation:
        prompt = self._prompt(value)
        responses, _ = await self.client.async_ask([prompt])
        response = responses[0].strip()
        if response.startswith("False"):
            reason = (
                f"[American English Style] Output {value!r} is not in American English. "
                f"LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        if response.startswith("True"):
            return True, None
        reason = (
            f"[American English Style] Unexpected LLM response {response!r} for value {value!r}."
        )
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "アメリカ英語のスペルと語彙で文章を作成してください。",
            "米国式英語に合わせ、colour ではなく color などの綴りを使ってください。",
            "出力はアメリカ英語の文体と語法で統一してください。",
            "アメリカ英語の語彙選択を行い、英国式表現を避けてください。",
            "米国英語のルールに基づき、スペルや言い回しを選んでください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "アメリカ英語で書いてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "スペルや語彙をアメリカ英語に合わせて統一してください。"

    def _prompt(self, text: str) -> str:
        return (
            "次のテキストがアメリカ英語で書かれているか確認してください。"
            "評価の際はユーザーが与えた文章を除き、アシスタントが出力した部分のみを対象としてください。\n"
            "アメリカ英語で書かれている場合は'True'、そうでない場合は'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )


class NoEnglishStyleConstraint(ConstraintGroupMixin):
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
                f"[No English Style] Output {value!r} contains English elements. "
                f"LLM response: {response!r}."
            )
            logger.info(reason)
            return False, reason
        reason = f"[No English Style] Unexpected LLM response {response!r} for value {value!r}."
        logger.info(reason)
        return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "回答が英語にならないようにしてください。",
            "英語の文章にならないように表現してください。",
            "英字を使った英語表現を避けて回答してください。",
            "英語ではなく他の言語で記述してください。",
            "英文にならないよう英語表現を排除してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "英語を使わずに書いてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "英字が入らないように書き直してください。"

    def _prompt(self, text: str) -> str:
        return (
            "次のテキストが英語のアルファベットや英語表現を含んでいないか確認してください。"
            "評価の際はユーザーが与えた文章を除き、アシスタントが出力した部分のみを対象としてください。\n"
            "英語の要素を含まなければ'True'、含んでいれば'False'と答えてください。\n"
            "Trueの場合はTrueとだけ出力し余計な内容を決して付け加えないでください。Falseの場合はまずFalseと出力し、その後ろになぜFalseと判定したのかという理由を含めてください。\n"
            "\n### テキスト\n"
            f"{text}"
        )
