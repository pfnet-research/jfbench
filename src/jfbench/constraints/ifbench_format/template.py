import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class OutputTemplateFormatIfbenchConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        # Handle empty or whitespace-only outputs
        if not value or not value.strip():
            reason = "[Output Template Format] Empty response."
            logger.info(reason)
            return False, reason

        # Normalize full-width colon to half-width to unify matching
        normalized = value.replace("：", ":")

        # Required section markers in the expected order
        markers = ["私の回答:", "私の結論:", "今後の展望:"]

        # Check that all markers are present
        if not all(marker in normalized for marker in markers):
            reason = (
                "[Output Template Format] Output must contain all required sections: "
                "私の回答:, 私の結論:, 今後の展望:."
            )
            logger.info(reason)
            return False, reason

        # Check that each marker appears exactly once
        counts = [normalized.count(marker) for marker in markers]
        if any(count != 1 for count in counts):
            reason = (
                "[Output Template Format] Each section marker must appear exactly once: "
                "私の回答:, 私の結論:, 今後の展望:."
            )
            logger.info(reason)
            return False, reason

        # Find positions of each marker and ensure correct ordering
        positions = [normalized.find(marker) for marker in markers]
        # If positions are not strictly increasing, the order is wrong
        if not (positions[0] < positions[1] < positions[2]):
            reason = (
                "[Output Template Format] Sections must appear in the order: "
                "私の回答: → 私の結論: → 今後の展望:."
            )
            logger.info(reason)
            return False, reason

        # Ensure each section has non-empty content after its marker
        sections = []
        for i, marker in enumerate(markers):
            start = positions[i] + len(marker)
            end = positions[i + 1] if i + 1 < len(markers) else len(normalized)
            section_text = normalized[start:end].strip()
            sections.append(section_text)

        if any(not section for section in sections):
            reason = (
                "[Output Template Format] Each section must contain non-empty content "
                "after its marker."
            )
            logger.info(reason)
            return False, reason

        # All checks passed: template is present, ordered, unique, and filled
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "回答は「私の回答：[回答] 私の結論：[結論] 今後の展望：[展望]」というテンプレートに厳密に従ってください。",
            "「私の回答: ... 私の結論: ... 今後の展望: ...」の形で各項目を埋めてください。",
            "指定の3セクションを「私の回答:」「私の結論:」「今後の展望:」で示してください。",
            "出力は必ずテンプレート「私の回答:◯ 私の結論:◯ 今後の展望:◯」に沿わせてください。",
            "3つの見出し（私の回答/私の結論/今後の展望）をこの順で用いて回答してください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "指定テンプレート「私の回答:… 私の結論:… 今後の展望:…」の形に整えてください。"
