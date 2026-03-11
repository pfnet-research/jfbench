import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class ConjunctionCountIfbenchConstraint(ConstraintGroupMixin):
    def __init__(self, minimum_kinds: int, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.minimum_kinds = minimum_kinds
        self._conjunctions: set[str] = {
            "そして",
            "しかし",
            "だから",
            "さらに",
            "また",
            "あるいは",
            "けれども",
            "それに",
            "それとも",
            "ところが",
            "なので",
            "ゆえに",
            "よって",
            "すると",
            "ただし",
            "なお",
            "つまり",
            "もし",
            "および",
            "ならびに",
            "したがって",
            "そのため",
        }

    def evaluate(self, value: str) -> ConstraintEvaluation:
        used = {conjunction for conjunction in self._conjunctions if conjunction in value}
        if len(used) < self.minimum_kinds:
            reason = (
                "[Conjunction Count] Not enough distinct conjunctions. "
                f"Expected at least {self.minimum_kinds}, found {len(used)}. "
                f"Used conjunctions: {', '.join(sorted(used))}"
            )
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"回答には少なくとも{self.minimum_kinds}種類の異なる接続詞を盛り込んでください。",
            f"{self.minimum_kinds}種類以上の接続詞を使い分けて文章を構成してください。",
            f"接続詞のバリエーションが{self.minimum_kinds}種類以上になるようにしてください。",
            f"文中で最低{self.minimum_kinds}種類の接続詞を登場させてください。",
            f"異なる接続詞を{self.minimum_kinds}種類以上用いることを意識してください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return f"接続詞の種類を{self.minimum_kinds}以上になるように追加してください。"
