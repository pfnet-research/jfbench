from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from jfbench.benchmark.build import get_constraint_collections
from jfbench.grpo_dataset.generate import parse_args
from jfbench.grpo_dataset.generate import RuleOnlyConstraintInventory


if TYPE_CHECKING:
    import argparse


DEFAULT_SPEC_JSON = '{"name": "stub", "model": "stub-model", "provider": "vllm"}'


def _parse_with_defaults(extra_args: list[str] | None = None) -> "argparse.Namespace":
    base = ["--n-data", "1", "--mode-spec-json", DEFAULT_SPEC_JSON]
    if extra_args:
        base.extend(extra_args)
    return parse_args(base)


def test_parse_args_uses_ja_stackoverflow_by_default() -> None:
    args = _parse_with_defaults()
    assert args.prompt_source == "ja_stackoverflow"
    assert args.target_group == "Logic"


def test_parse_args_accepts_target_group() -> None:
    args = _parse_with_defaults(["--target-group", "Format"])
    assert args.target_group == "Format"


def test_parse_args_rejects_non_ja_stackoverflow_prompt_source() -> None:
    with pytest.raises(SystemExit):
        _parse_with_defaults(["--prompt-source", "ifbench"])


def test_rule_only_constraint_inventory_uses_only_rule_factories() -> None:
    inventory = RuleOnlyConstraintInventory(constraint_set="train")
    assert "MetaOutput" not in inventory.groups
    assert "Structure" not in inventory.groups
    assert all(wrapper.mode == "rule" for wrapper in inventory.all_wrappers)

    collections = get_constraint_collections("train")
    expected = (
        len(collections.format)
        + len(collections.character)
        + len(collections.rule_content)
        + len(collections.length)
        + len(collections.logic)
        + len(collections.notation)
        + len(collections.rule_processing)
        + len(collections.rule_style)
        + len(collections.ifbench_count)
        + len(collections.ifbench_format)
        + len(collections.ifbench_ratio)
        + len(collections.ifbench_repeat)
        + len(collections.ifbench_sentence)
        + len(collections.ifbench_words)
    )
    assert len(inventory.all_wrappers) == expected
