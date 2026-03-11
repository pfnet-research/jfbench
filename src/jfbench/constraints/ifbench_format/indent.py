from itertools import pairwise
import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class LineIndentFormatIfbenchConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        lines = value.splitlines()
        indents = [len(line) - len(line.lstrip(" ")) for line in lines if line.strip()]
        for previous, current in pairwise(indents):
            if current <= previous:
                reason = "[Line Indent Format] Indentation must increase step by step."
                logger.info(reason)
                return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "各行のインデントを段階的に増やし、階段状になるようにしてください。",
            "行ごとにスペースを増やして段差のある形にしてください。",
            "行頭の字下げを一行ごとに深くし、階段状のレイアウトにしてください。",
            "行を進むごとにインデントが増える構造にしてください。",
            "先頭の空白を段階的に増やし、階段のようにしてください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "行ごとにインデントが増えるようにスペースを追加してください。"
