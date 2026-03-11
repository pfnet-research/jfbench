import logging
import re

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.protocol import ConstraintEvaluation


logger = logging.getLogger(__name__)


class PersonNamesCountIfbenchConstraint(ConstraintGroupMixin):
    def __init__(self, minimum_names: int, *, seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.minimum_names = minimum_names
        self._allowed_names = {
            "太郎",
            "花子",
            "次郎",
            "さくら",
            "蓮",
            "葵",
            "結衣",
            "悠真",
            "陽菜",
            "大翔",
            "陽斗",
            "結愛",
            "咲良",
            "蒼",
            "心春",
            "陽葵",
            "颯太",
            "莉子",
            "湊",
            "美月",
            "優斗",
            "楓",
            "紗季",
            "奏太",
            "碧",
            "美咲",
            "悠斗",
            "結菜",
            "颯真",
            "雫",
            "新",
            "瑞希",
            "直樹",
            "遥",
            "恵",
            "健太",
            "真央",
            "凛",
            "遼",
            "香織",
            "拓海",
            "翼",
            "彩乃",
            "歩夢",
            "優奈",
            "春樹",
            "美優",
            "航平",
            "光",
        }

    def evaluate(self, value: str) -> ConstraintEvaluation:
        found = set()
        for name in self._allowed_names:
            if re.search(re.escape(name), value, flags=re.IGNORECASE):
                found.add(name)
        if len(found) < self.minimum_names:
            reason = (
                "[Person Names Count] Not enough distinct names from the allowed list. "
                f"Expected at least {self.minimum_names}, found {len(found)}."
            )
            logger.info(reason)
            return False, reason
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        templates = [
            f"回答にはリストから異なる人物名を{self.minimum_names}種類以上含めてください。",
            f"指定の名前一覧から{self.minimum_names}人以上の名前を使って記述してください。",
            f"提供された人物名リストから少なくとも{self.minimum_names}種類挙げてください。",
            f"リスト内の固有名を{self.minimum_names}種類以上盛り込んでください。",
            f"重複しないように、指定された名前を{self.minimum_names}種類以上使ってください。",
        ]
        if train_or_test == "train":
            raise ValueError("train_or_test must be 'test' for ifbench constraints.")
        if train_or_test == "test":
            return (
                self._random_instruction(templates)
                + "\n\n"
                + ("使用可能な名前一覧:\n" + ", ".join(sorted(self._allowed_names)))
            )
        raise AssertionError(f"Unknown train_or_test: {train_or_test}")

    def rewrite_instructions(self) -> str:
        return f"指定リストから{self.minimum_names}種類以上の人物名が含まれるようにしてください。"
