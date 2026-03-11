import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class CsvFormatConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        lines = value.strip().split("\n")
        num_columns = len(lines[0].split(","))
        for line in lines[1:]:
            if len(line.split(",")) != num_columns:
                reason = "[CSV] Inconsistent number of columns"
                logger.info(reason)
                return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "各行が同じ列数を持つカンマ区切りのCSVとして出力してください。",
            "CSV形式で回答し、すべての行が同じ数のカンマ区切り列を持つようにしてください。",
            "CSVパーサーで読み取れるよう、均一な列数のカンマ区切りデータを生成してください。",
            "行ごとの列数を揃えたCSV（comma-separated values）を出力してください。",
            "カンマ区切りの表形式（全行同じ列数）でデータを返してください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return "CSV形式で出力してください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "カンマ区切りの列数を揃え、CSVとして読み取れる形式に整えてください。"
