from typing import Any

import pytest
from pytest import MonkeyPatch

from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints.notation import CamelcaseNotationConstraint
from jfbench.constraints.notation import CurrencyNotationConstraint
from jfbench.constraints.notation import DateNotationConstraint
from jfbench.constraints.notation import FuriganaNotationConstraint
from jfbench.constraints.notation import RoundingNotationConstraint
from jfbench.constraints.notation import SnakecaseNotationConstraint
from jfbench.constraints.notation import TimeNotationConstraint
from jfbench.constraints.notation import TitlecaseNotationConstraint
from jfbench.constraints.notation import UnitNotationConstraint
from jfbench.constraints.notation import ZeroPaddingNotationConstraint


def test_currency_notation_constraint_validates_format() -> None:
    constraint = CurrencyNotationConstraint()
    assert constraint.evaluate("価格は¥1,234です")[0] is True
    assert constraint.evaluate("価格は1234円です")[0] is False


def test_date_notation_constraint_detects_format() -> None:
    constraint = DateNotationConstraint()
    assert constraint.evaluate("今日は 2024年05月01日 の天気です")[0] is True
    assert constraint.evaluate("今日は2024年5月1日です")[0] is False
    assert constraint.evaluate("今日は24年5月1日です")[0] is False
    assert constraint.evaluate("今日は2年5月1日です")[0] is False
    assert constraint.evaluate("今日は2024/05/01です")[0] is False
    assert constraint.evaluate("今日は12345年1月1日です")[0] is False


def test_zero_padding_notation_constraint_requires_padding() -> None:
    constraint = ZeroPaddingNotationConstraint(4)
    assert constraint.evaluate("ID: 0001 を付与")[0] is True
    assert constraint.evaluate("ID: 12")[0] is False


def _capture_options(monkeypatch: MonkeyPatch) -> dict[str, list[str]]:
    captured: dict[str, list[str]] = {}

    def fake_random_instruction(self: ConstraintGroupMixin, options: list[str]) -> str:
        captured["options"] = list(options)
        return options[0]

    monkeypatch.setattr(ConstraintGroupMixin, "_random_instruction", fake_random_instruction)
    return captured


@pytest.mark.parametrize(
    ("constraint", "expected_tokens"),
    [
        (TitlecaseNotationConstraint(seed=0), ("Title Case",)),
        (SnakecaseNotationConstraint(seed=0), ("snake_case",)),
        (CamelcaseNotationConstraint(seed=0), ("CamelCase",)),
        (FuriganaNotationConstraint(seed=0), ("<ruby>",)),
        (ZeroPaddingNotationConstraint(4, seed=0), ("4",)),
        (CurrencyNotationConstraint(seed=0), ("¥",)),
        (DateNotationConstraint(seed=0), ("YYYY年MM月DD日",)),
        (TimeNotationConstraint(seed=0), ("HH:MM",)),
        (RoundingNotationConstraint(2, seed=0), ("2", "四捨五入")),
        (UnitNotationConstraint(seed=0), ("SI",)),
    ],
)
def test_notation_custom_instructions_templates(
    monkeypatch: MonkeyPatch,
    constraint: Any,
    expected_tokens: tuple[str, ...],
) -> None:
    captured = _capture_options(monkeypatch)
    instruction = constraint.instructions()
    options = captured["options"]

    assert instruction == options[0]
    assert len(options) == 5
    for token in expected_tokens:
        assert all(token in option for option in options)
