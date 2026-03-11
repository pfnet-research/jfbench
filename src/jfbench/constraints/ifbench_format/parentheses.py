import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class ParenthesesFormatIfbenchConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        stack: list[str] = []
        pairs = {")": "(", "]": "[", "}": "{"}
        max_depth = 0
        for char in value:
            if char in "([{":
                stack.append(char)
                max_depth = max(max_depth, len(stack))
            elif char in pairs:
                if not stack or stack[-1] != pairs[char]:
                    reason = "[Parentheses Format] Unbalanced brackets detected."
                    logger.info(reason)
                    return False, reason
                stack.pop()
        if max_depth < 5:
            reason = "[Parentheses Format] Nesting depth must reach at least five."
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "半角の括弧や角括弧、波括弧を少なくとも5段階以上ネストさせてください。",
            "複数種類の半角括弧(), [], {}を使い、5階層以上の入れ子を作ってください。",
            "(), [], {} を組み合わせて5段以上の深い括弧構造を作成してください。",
            "5階層以上の半角の括弧や角括弧、波括弧のネストを含む文章にしてください。",
            "半角括弧や角括弧、波括弧を入れ子にし、少なくとも5段階の深さを持たせてください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "半角の括弧や角括弧、波括弧のネストを5段以上に深めてください。"
