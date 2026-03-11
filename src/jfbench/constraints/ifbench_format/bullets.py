import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class SubBulletsFormatIfbenchConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> ConstraintEvaluation:
        # Handle empty or whitespace-only responses explicitly
        if not value or not value.strip():
            reason = "[Sub Bullets Format] Empty response."
            logger.info(reason)
            return False, reason

        lines = value.splitlines()

        has_any_top = False  # At least one '*' top-level bullet exists somewhere
        has_any_sub = False  # At least one '-' sub bullet exists somewhere
        current_has_sub = False  # The current '*' block has at least one '-' sub bullet
        has_invalid_top = False  # Some '*' item does not have any '-' sub items
        has_orphan_sub = False  # A '-' sub bullet appears before any '*' top-level item

        for raw_line in lines:
            # Remove leading whitespace to detect bullet markers
            line = raw_line.lstrip()

            # Be strict: treat only lines starting with "* " as top-level bullets
            if line.startswith("* "):
                # Close previous top-level item: if it had no sub bullet, mark invalid
                if has_any_top and not current_has_sub:
                    has_invalid_top = True

                has_any_top = True
                current_has_sub = False

            # Be strict: treat only lines starting with "- " as sub bullets
            elif line.startswith("- "):
                has_any_sub = True

                # Sub bullet before any top-level bullet is considered orphan
                if not has_any_top:
                    has_orphan_sub = True
                else:
                    current_has_sub = True

        # After the loop, close the last top-level item if there was any
        if has_any_top and not current_has_sub:
            has_invalid_top = True

        reasons = []

        # Basic presence checks
        if not has_any_top and not has_any_sub:
            reasons.append(
                "[Sub Bullets Format] Missing both '*' top-level and '-' sub bullet items."
            )
        elif not has_any_top:
            reasons.append("[Sub Bullets Format] Missing '*' top-level bullet items.")
        elif not has_any_sub:
            reasons.append("[Sub Bullets Format] Missing '-' sub bullet items.")

        # Structural checks
        if has_invalid_top:
            reasons.append("[Sub Bullets Format] Some '*' items do not have any '-' sub items.")

        if has_orphan_sub:
            reasons.append(
                "[Sub Bullets Format] Found '-' sub bullet items before any '*' top-level item."
            )

        if reasons:
            # Join all detected problems into a single reason string
            reason = " ".join(reasons)
            # Log with a short preview of the value for easier debugging
            logger.info("%s value_preview=%r", reason, value[:80])
            return False, reason

        # All checks passed: every '*' has at least one '-' and both exist
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            "回答には*で示した箇条書きと、各項目に少なくとも1つの-で示す下位項目を含めてください。",
            "*の箇条書きと、それぞれに-のサブ項目を付けてください。",
            "トップレベルを*、サブ項目を-で示したリスト構造を含めてください。",
            "少なくとも一つの*項目と、各*項目に対して1つ以上の-項目を追加してください。",
            "箇条書きを*で作り、各項目に-の下位箇条書きを入れてください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return "*の箇条書きを追加し、それぞれに-の下位項目を追加してください。"
