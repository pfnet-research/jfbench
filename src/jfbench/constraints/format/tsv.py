import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class TsvFormatConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        lines = value.strip().split("\n")
        num_columns = len(lines[0].split("\t"))
        for line in lines[1:]:
            if len(line.split("\t")) != num_columns:
                reason = f"[TSV] Line does not match column count: {line}"
                logger.info(reason)
                return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "各行をタブ区切りにしたTSV形式で出力してください。",
            "タブ文字で列を区切り、全行が同じ列数になるTSVを生成してください。",
            "TSV（Tab-Separated Values）形式で回答し、余計なテキストを入れないでください。",
            "タブ区切りのテーブルだけを返し、列数が揃っていることを保証してください。",
            "タブで分割された表形式（TSV）でデータを出力してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "TSV形式で出力してください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "各行をタブ区切りでそろえ、同じ列数を持つTSV形式に書き直してください。"
