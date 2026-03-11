import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class JapanesePunctuationConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        for index, char in enumerate(value):
            if char in ",.\uff0c\uff0e":
                if index > 0 and value[index - 1].isdigit():
                    continue
                reason = f"[Japanese Punctuation] Found non-Japanese punctuation character: {char}"
                logger.info(reason)
                return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "使用できる句読点は日本語の「、」と「。」のみです。",
            "英語のコンマやピリオドを使わず、日本語の句読点だけで表現してください。",
            "句読点は必ず「、」「。」に統一してください。",
            "日本語の句読点以外（, .）を使わないでください。",
            "文章の区切りには日本語の「、」「。」のみを採用してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "文章中で英語のコンマ（,）やピリオド（.）を使用せず、日本語の句読点「、」「。」のみを使ってください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "英数字のピリオドやカンマを「。」と「、」に差し替えてください。"


class NoJapanesePunctuationConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        for index, char in enumerate(value):
            if char in "、。":
                if index > 0 and value[index - 1].isdigit():
                    continue
                reason = f"[No Japanese Punctuation] Found Japanese punctuation character: {char}"
                logger.info(reason)
                return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "「、」「。」といった日本語の句読点は使用せずに書いてください。",
            "句読点は英語スタイルに統一し、日本語の「、」「。」は避けてください。",
            "日本語の句読点を含めない文章にしてください。",
            "「、」や「。」を用いずに情報を表現してください。",
            "日本語の句読点の代わりに他の記号を使ってください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "日本語の「、」「。」を使わず、英語のコンマやピリオドなど別の句読点で表記してください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "「、」「。」を使っている箇所を英語のコンマやピリオドなどに置き換えてください。"


class NoCommasConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        for index, char in enumerate(value):
            if char in ",、\uff0c":
                if index > 0 and value[index - 1].isdigit():
                    continue
                reason = f"[No Commas] Found comma character: {char}"
                logger.info(reason)
                return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "文章中で「,」や「、」といったカンマを一切使用しないでください。",
            "英語のカンマ、日本語の読点ともに禁止です。",
            "「,」「、」を含めない表現にしてください。",
            "コンマ記号を排除し、別の区切り方で文章を構成してください。",
            "カンマ類は禁止なので、文を区切る際は他の方法を使ってください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "「,」「、」などカンマを一切含めずに出力してください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "文章中の「,」や「、」を削除するか別の句読点に置き換えてください。"
