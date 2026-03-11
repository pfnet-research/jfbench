import logging
import re

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class BulletPointsFormatConstraint(ConstraintGroupMixin):
    def __init__(self, starter_token: str = "-", *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        if not isinstance(starter_token, str) or not starter_token.strip():
            raise ValueError("starter_token must be a non-empty, non-whitespace string")
        self.starter_token = starter_token

        tok = re.escape(self.starter_token)
        self._ok = re.compile(rf"^\s*{tok}\s*(?!{tok}\b)\S")
        self._starter_only = re.compile(rf"^\s*{tok}\s*$")

    def evaluate(self, value: str) -> ConstraintEvaluation:
        if not value:
            reason = "[Bullet Points] No content provided."
            logger.info(reason)
            return False, reason

        has_expected = False
        for line in value.splitlines():
            if not line.strip():
                continue
            s = line.lstrip()

            if self._starter_only.match(s):
                reason = f"[Bullet Points] Bullet with starter {self.starter_token!r} has no content: {line!r}"
                logger.info(reason)
                return False, reason

            if s.startswith(self.starter_token):
                tail = s[len(self.starter_token) :]
                if not tail.strip(self.starter_token).strip():
                    reason = f"[Bullet Points] Bullet with starter {self.starter_token!r} has no content: {line!r}"
                    logger.info(reason)
                    return False, reason

            if self._ok.match(s):
                has_expected = True
                continue

        if not has_expected:
            reason = f"[Bullet Points] No bullet points found with starter {self.starter_token!r}."
            logger.info(reason)
            return False, reason

        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"少なくとも1行は先頭が「{self.starter_token}」の箇条書きになるようにしてください。",
            f"箇条書きは{self.starter_token}から始め、最低1つはその形式で記述してください。",
            f"出力内に{self.starter_token}で始まる項目を含むリストを作成してください。",
            f"各箇条書きの先頭記号として{self.starter_token}を用い、最低1項目は必ず含めてください。",
            f"{self.starter_token}を先頭に付けたバレットリストを挿入し、その形式を守ってください。",
        ]
        if train_or_test == "train":
            return self._random_instruction(templates)
        if train_or_test == "test":
            return f"「{self.starter_token}」で始まる箇条書きを含めてください。"
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return f"各箇条書きの先頭を「{self.starter_token}」で始まるように修正してください。"
