import logging
import re

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints._utils import split_sentences
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


_BASE_EMOJI_PATTERN = (
    "[\U0001f300-\U0001faff\U00002700-\U000027bf\U0001f900-\U0001f9ff\U0001f1e6-\U0001f1ff]"
)
_EMOJI_AT_END_PATTERN = re.compile(_BASE_EMOJI_PATTERN + "\ufe0f?$")


class EmojiFormatIfbenchConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        sentences = split_sentences(value.strip())
        for sentence in sentences:
            tail = re.sub(r"[。．！？.!?\s]+$", "", sentence)
            if not tail:
                reason = "[Emoji Format] Sentence is empty after stripping punctuation."
                logger.info(reason)
                return False, reason

            if not _EMOJI_AT_END_PATTERN.search(tail):
                reason = "[Emoji Format] Each sentence must end with an emoji."
                logger.info(reason)
                return False, reason

        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "各文の末尾には必ず絵文字を付けてください。",
            "文末を絵文字で締めくくってください。",
            "すべての文の最後に絵文字を入れてください。",
            "各センテンスの終わりに1つ以上の絵文字を配置してください。",
            "文ごとに末尾に絵文字を添えてください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "各文末に必ず絵文字を追加してください。"
