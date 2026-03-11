import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints._utils import split_sentences
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class SentenceBalanceRatioIfbenchConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        sentences = split_sentences(value)

        if not sentences:
            reason = "[Sentence Balance Ratio] No sentences to evaluate."
            logger.info(reason)
            return False, reason

        counts = {"statement": 0, "question": 0, "exclamation": 0}

        for sentence in sentences:
            stripped = sentence.rstrip()
            if not stripped:
                counts["statement"] += 1
                continue

            last = stripped[-1]
            if last in ("?", "？"):
                counts["question"] += 1
            elif last in ("!", "！"):
                counts["exclamation"] += 1
            else:
                counts["statement"] += 1

        values = list(counts.values())

        if any(count == 0 for count in values):
            reason = (
                "[Sentence Balance Ratio] Each sentence type must appear at least once. "
                f"counts={counts}, total={len(sentences)}"
            )
            logger.info(reason)
            return False, reason

        max_count = max(values)
        min_count = min(values)

        if max_count - min_count > 1:
            reason = (
                "[Sentence Balance Ratio] Sentence types are not balanced. "
                f"counts={counts}, total={len(sentences)}"
            )
            logger.info(reason)
            return False, reason

        logger.debug(
            "[Sentence Balance Ratio] Passed. counts=%s, total=%d",
            counts,
            len(sentences),
        )
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "平叙文・疑問文・感嘆文の比率がバランスよくなるようにしてください。",
            "文末の形（。？！）が偏らないよう均等に配置してください。",
            "平叙文、疑問文、感嘆文をバランスよく組み合わせてください。",
            "3種類の文型(平叙文、疑問文、感嘆文)を均等に含めてください。",
            "平叙文、疑問文、感嘆文といった文型の分布が偏らないよう調整してください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "平叙文・疑問文・感嘆文がバランスよくなるように文末を整えてください。"
