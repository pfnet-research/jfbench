import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints._utils import split_sentences
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class SentenceTypeRatioIfbenchConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        sentences = split_sentences(value)

        if not sentences:
            reason = "[Sentence Type Ratio] No sentences to evaluate."
            logger.info(reason)
            return False, reason

        declaratives = 0
        questions = 0

        for sentence in sentences:
            stripped = sentence.rstrip()
            if not stripped:
                continue

            last = stripped[-1]
            if last in (".", "。", "．"):
                declaratives += 1
            elif last in ("?", "？"):
                questions += 1

        if questions == 0:
            reason = (
                "[Sentence Type Ratio] At least one question is required to form a 2:1 ratio. "
                f"(declaratives={declaratives}, questions={questions}, total={len(sentences)})"
            )
            logger.info(reason)
            return False, reason

        ratio = declaratives / questions
        diff = abs(ratio - 2.0)

        if diff > 0.2:
            reason = (
                "[Sentence Type Ratio] Declarative to question ratio "
                f"{ratio:.2f} is not close to 2:1. "
                f"(declaratives={declaratives}, questions={questions}, "
                f"total={len(sentences)}, diff={diff:.2f})"
            )
            logger.info(reason)
            return False, reason

        logger.debug(
            "[Sentence Type Ratio] Passed. ratio=%.2f, declaratives=%d, questions=%d, total=%d",
            ratio,
            declaratives,
            questions,
            len(sentences),
        )
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "宣言文と疑問文の比率を2:1に保ってください。",
            "平叙文が疑問文のおよそ2倍になるようにしてください。",
            "平叙文と疑問文の比率を2対1に調整してください。",
            "疑問文1に対して平叙文2の割合で構成してください。",
            "文末の?に対して.が2倍になるようにしてください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "疑問文1に対して平叙文2の比率になるように文数を調整してください。"
