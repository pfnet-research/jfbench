import json
import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class JsonFormatConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        try:
            json.loads(value)
            return True, None
        except json.JSONDecodeError as e:
            reason = f"[JSON] Decode error: {e}"
            logger.info(reason)
            return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "正しいJSON文字列を出力し、パース可能であることを保証してください。",
            "JSONフォーマットに従い、無効な構文を含めないでください。",
            "有効なJSONとして読み取れるデータのみを返してください。",
            "キーや値を適切に引用したJSON出力にしてください。",
            "JSONパーサーがエラーを返さない形式で結果を出力してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "JSON形式で出力してください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "キーや引用符、カンマの配置を見直し、正しいJSON文字列に修正してください。"
