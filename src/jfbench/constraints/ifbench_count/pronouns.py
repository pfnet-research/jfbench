import logging

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints._utils import split_words
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class PronounsCountIfbenchConstraint(ConstraintGroupMixin):
    def __init__(self, minimum_pronouns: int, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.minimum_pronouns = minimum_pronouns
        self._pronouns = {
            "私",
            "わたし",
            "わたくし",
            "僕",
            "ぼく",
            "俺",
            "おれ",
            "あなた",
            "きみ",
            "君",
            "彼",
            "彼女",
            "彼ら",
            "彼女ら",
            "私たち",
            "わたしたち",
            "わたくしたち",
            "僕ら",
            "ぼくら",
            "俺たち",
            "おれたち",
            "あなたたち",
            "きみたち",
            "君たち",
            "彼らたち",
            "彼女たち",
            "彼女らたち",
            "これ",
            "それ",
            "あれ",
            "どれ",
            "ここ",
            "そこ",
            "あそこ",
            "どこ",
            "こちら",
            "そちら",
            "あちら",
            "どちら",
            "こう",
            "そう",
            "ああ",
            "どう",
            "誰",
            "だれ",
            "どなた",
            "なに",
            "何",
            "なん",
            "どれ",
            "どちら",
            "どこ",
            "いつ",
            "どう",
            "だれか",
            "誰か",
            "なにか",
            "何か",
            "どれか",
            "だれも",
            "誰も",
            "なにも",
            "何も",
            "どれも",
            "みんな",
            "皆",
            "みな",
            "それぞれ",
            "おのおの",
            "どなたか",
            "どなたも",
            "どこか",
            "どこにも",
            "自分",
            "自分自身",
            "じぶん",
            "我",
            "われ",
            "我々",
            "われわれ",
        }

    def evaluate(self, value: str) -> ConstraintEvaluation:
        words = split_words(value)
        count = sum(1 for word in words if word in self._pronouns)
        if count < self.minimum_pronouns:
            reason = (
                "[Pronouns Count] Not enough pronouns. "
                f"Expected at least {self.minimum_pronouns}, found {count}."
            )
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"回答には少なくとも{self.minimum_pronouns}個の代名詞を含めてください。",
            f"代名詞の数が{self.minimum_pronouns}個以上になるように文章を構成してください。",
            f"{self.minimum_pronouns}個以上の代名詞をちりばめてください。",
            f"代名詞を豊富に用い、合計{self.minimum_pronouns}個以上登場させてください。",
            f"文章中の代名詞の出現回数を{self.minimum_pronouns}個以上にしてください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return self._random_instruction(templates)
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return f"代名詞の出現数を{self.minimum_pronouns}以上に増やしてください。"
