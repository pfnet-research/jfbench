import logging
import re
import xml.etree.ElementTree as ET

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class XmlFormatConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        cleaned = self._strip_code_fence(value)
        try:
            ET.fromstring(cleaned)
            return True, None
        except ET.ParseError as e:
            reason = f"[XML] Parsing failed: {e}"
            logger.info(reason)
            return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "妥当なXML文書として解析できる形式で出力してください。",
            "タグの入れ子や属性を整え、正しいXMLだけを生成してください。",
            "XMLパーサーが問題なく読み取れる構造のテキストにしてください。",
            "有効なXMLとして完結するように記述してください。",
            "XMLの構文ルールをすべて満たす出力を返してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "XML形式で出力してください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "不正な構造やタグを修正し、妥当なXMLになるように書き直してください。"

    @staticmethod
    def _strip_code_fence(value: str) -> str:
        stripped = value.strip()
        match = re.match(r"\A```[^\n]*\n(.*)\n```\s*\Z", stripped, re.DOTALL)
        if match is None:
            return value
        return match.group(1)
