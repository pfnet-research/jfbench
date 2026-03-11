import logging
import re

import yaml

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class YamlFormatConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        cleaned = self._strip_code_fence(value)
        try:
            yaml.safe_load(cleaned)
            return True, None
        except yaml.YAMLError as e:
            reason = f"[YAML] Parsing failed: {e}"
            logger.info(reason)
            return False, reason

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "パース可能なYAML文書として出力してください。",
            "YAMLのインデントや記法に従ったデータ構造のみを返してください。",
            "有効なYAMLを生成し、フォーマットエラーを含めないでください。",
            "YAMLシリアライズ済みのデータとして整形されたテキストを出力してください。",
            "YAML仕様上正しい構造だけを持つ回答にしてください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "YAML形式で出力してください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "インデントやコロンの使い方を整え、YAMLとして解釈できる形式に直してください。"

    @staticmethod
    def _strip_code_fence(value: str) -> str:
        stripped = value.strip()
        match = re.match(r"\A```[^\n]*\n(.*)\n```\s*\Z", stripped, re.DOTALL)
        if match is None:
            return value
        return match.group(1)
