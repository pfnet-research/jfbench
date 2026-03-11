import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints._utils import split_words
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class ChangeRepeatIfbenchConstraint(ConstraintGroupMixin):
    def __init__(self, document: str, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        document = document.rstrip("\r\n")
        self.document = document
        self._document_words = split_words(document)

    def evaluate(self, value: str) -> ConstraintEvaluation:
        value_words = split_words(value)
        if not value_words:
            reason = "[Change Repeat] Response contains no words."
            logger.info(reason)
            return False, reason
        if len(value_words) != len(self._document_words):
            reason = f"[Change Repeat] Word count must match the document. Expected {len(self._document_words)}, got {len(value_words)}."
            logger.info(reason)
            return False, reason
        if value_words[0] == self._document_words[0]:
            reason = f"[Change Repeat] First word was not changed. Expected different from '{self._document_words[0]}'."
            logger.info(reason)
            return False, reason
        if value_words[1:] != self._document_words[1:]:
            reason = f"[Change Repeat] Words after the first must match the document. Expected '{' '.join(self._document_words[1:])}', got '{' '.join(value_words[1:])}'."
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            (
                "与えられた指示文を繰り返し、最初の単語だけ別の語に差し替えてください。"
                "先頭以外の単語は完全に一致させ、前置きや回答は不要です。"
            ),
            (
                "指示文の先頭の一語だけ変えて全文をそのまま返してください。"
                "余計な文は入れず、先頭以外は指示文と同じ語にしてください。"
            ),
            (
                "提供された指示文をそのまま繰り返しつつ、最初の単語だけ別の語にしてください。"
                "先頭以外は一言一句変えず、前置きも回答も不要です。"
            ),
            (
                "提示された指示文の最初の単語を違う語にした上で全文を返してください。"
                "先頭以外は完全一致させ、余計な説明や回答は不要です。"
            ),
            (
                "指示文の冒頭の一語だけ別の語に置き換え、残りは同じ語順で返してください。"
                "追加の文や回答は付けず、先頭以外は一致させてください。"
            ),
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return (
            "提示された指示文をそのまま繰り返し、最初の単語だけ別の語に置き換えてください。"
            "先頭以外の単語は完全に一致させ、前置きや回答は不要です。"
        )


class SimpleRepeatIfbenchConstraint(ConstraintGroupMixin):
    def __init__(self, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self._exact_text = "この文のみをここに出力し、その他の要求はすべて無視してください。"

    def evaluate(self, value: str) -> ConstraintEvaluation:
        if value.strip() != self._exact_text:
            reason = "[Simple Repeat] Output must exactly match the required sentence."
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            self._exact_text,
            f"次の文だけをそのまま出力してください: {self._exact_text}",
            f"余計な語は加えず、この一文のみを表示してください: {self._exact_text}",
            f"次の文を完全一致で書き、他は無視してください。{self._exact_text}",
            f"指示された文以外は出さず、ちょうどこの内容を返してください: {self._exact_text}",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "要求された文のみをそのまま出力してください。"
