import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class NoWhitespaceConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        for char in value:
            if char.isspace():
                reason = f"[No Whitespace] Found whitespace character: {char!r}"
                logger.info(reason)
                return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "出力には空白文字（スペース、タブ、改行など）を一切含めないでください。",
            "スペースやタブ、改行を含む空白文字を出さずに結果を記述してください。",
            "任意の種類の空白文字を用いず、文字列を連続させたまま出力してください。",
            "改行・スペース・タブなどの空白を利用せずに回答を構成してください。",
            "空白文字（スペース／タブ／改行）は禁止なので、それらなしで出力してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "空白文字（スペース、タブ、改行など）はなしでお願いします。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "先ほどの出力から空白文字（スペース、タブ、改行など）をすべて取り除いてください。"


class NoSuffixWhitespaceConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        if value != value.rstrip():
            whitespaces = value[len(value.rstrip()) :]
            reason = f"[No Suffix Whitespace] Found trailing whitespace: {whitespaces!r}"
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "出力末尾にスペースや改行などの空白文字を残さないでください。",
            "回答の最後にはタブ・スペース・改行といった空白を付けないでください。",
            "末尾に空白類（スペース／タブ／改行）が入らないように整形してください。",
            "出力の最後は文字で終わらせ、空白文字で締めくくることを避けてください。",
            "回答末尾には空白を含めず、不要なスペースや改行を削除して提出してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "出力末尾にスペース・タブ・改行などの空白はなしにしてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "出力末尾の余分な空白（スペース、タブ、改行など）を削除してください。"
