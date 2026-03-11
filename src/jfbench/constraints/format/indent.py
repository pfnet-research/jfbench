import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class IndentFormatConstraint(ConstraintGroupMixin):
    def __init__(self, indent: str, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.indent = indent

    def evaluate(self, value: str) -> ConstraintEvaluation:
        lines = value.splitlines()

        # If the empty indent string is specified, anycase is passed.
        if not self.indent:
            return True, None

        for line in lines:
            # Skip empty lines.
            if not line:
                continue

            # Extract leading whitespace.
            leading_ws_len = 0
            for ch in line:
                if ch.isspace():
                    leading_ws_len += 1
                else:
                    break

            # Allow lines with no indent (leading non-whitespace).
            if leading_ws_len == 0:
                continue

            leading_ws = line[:leading_ws_len]

            # Check the leading whitespace pattern.
            unit = self.indent
            unit_len = len(unit)

            # If the leading whitespace is not a multiple of the indent unit,
            # or if it contains characters other than the indent unit,
            # the format is invalid.
            if (
                unit_len == 0
                or leading_ws_len % unit_len != 0
                or leading_ws != unit * (leading_ws_len // unit_len)
            ):
                reason = (
                    "[Indent Format] Lines may be unindented, but if a line has "
                    f"indentation, its leading whitespace must consist only of "
                    f"repeated {self.indent!r}."
                )
                logger.info(reason)
                return False, reason

        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"各行の先頭にインデントを付ける場合は、必ず{self.indent!r}（の繰り返し）のみを使用してください。インデントがない行はそのままで構いません。",
            f"行頭のインデントは省略しても構いませんが、付ける場合は必ず{self.indent!r}（の繰り返し）のみを使ってください。",
            f"インデントを使う行では、行頭を{self.indent!r}の繰り返しだけで構成してください。インデントなしの行は許可されます。",
            f"冒頭のインデントは、付ける場合に限り{self.indent!r}を単位とした繰り返しに統一してください。",
            f"各行はインデントなしでもよいですが、インデントを付ける場合は{self.indent!r}のみ（の繰り返し）を行頭インデントとして使ってください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return (
                f"インデントを付ける行では、行頭を{self.indent!r}の繰り返しのみで構成し、"
                "インデントが不要な行は行頭をそのままにしてください。"
            )
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return (
            f"インデントを付ける行では、行頭を{self.indent!r}の繰り返しのみで構成し、"
            "インデントが不要な行は行頭をそのままにしてください。"
        )
