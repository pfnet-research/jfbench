from __future__ import annotations

from typing import cast

import numpy as np

from jfbench.protocol import Constraint
from jfbench.protocol import Prompt
from jfbench.sft_dataset.generate import ConstraintInventory
from jfbench.sft_dataset.generate_including_ifbench import IfbenchConstraintSampler


class DummyPrompt(Prompt):
    def __init__(self, idx: int) -> None:
        self.idx = idx

    def text(self, constraints: list[Constraint], *, train_or_test: str = "train") -> str:
        return f"Prompt-{self.idx} with {len(constraints)} constraints"

    @property
    def document(self) -> str:
        return f"Document-{self.idx}"


class DummyLLMClient:
    async def async_ask(self, prompts: list[str]) -> tuple[list[str], list[None]]:
        return ["True" for _ in prompts], [None] * len(prompts)


def test_ifbench_constraint_sampler_uses_only_ifbench_for_single_constraint() -> None:
    sampler = IfbenchConstraintSampler(
        prompt_source="ja_stackoverflow",
        prompts=[DummyPrompt(0)],
        inventory=ConstraintInventory(constraint_set="test"),
        judge_client=cast("object", DummyLLMClient()),
        rng=np.random.default_rng(17),
    )

    sample = sampler.sample(1)

    assert sample.benchmark_data.meta_data.n_constraints == 1
    groups = sample.benchmark_data.meta_data.constraint_groups
    assert len(groups) == 1
    assert groups[0].startswith("Ifbench")


def test_ifbench_constraint_sampler_includes_ifbench_for_multiple_constraints() -> None:
    sampler = IfbenchConstraintSampler(
        prompt_source="ja_stackoverflow",
        prompts=[DummyPrompt(0)],
        inventory=ConstraintInventory(constraint_set="test"),
        judge_client=cast("object", DummyLLMClient()),
        rng=np.random.default_rng(23),
    )

    sample = sampler.sample(4)

    assert sample.benchmark_data.meta_data.n_constraints == 4
    assert any(
        group.startswith("Ifbench") for group in sample.benchmark_data.meta_data.constraint_groups
    )


def test_ifbench_constraint_sampler_multiplies_max_rewrite_attempts() -> None:
    sampler = IfbenchConstraintSampler(
        prompt_source="ja_stackoverflow",
        prompts=[DummyPrompt(0)],
        inventory=ConstraintInventory(constraint_set="test"),
        judge_client=cast("object", DummyLLMClient()),
        rng=np.random.default_rng(29),
    )

    sample_single = sampler.sample(1)
    sample_multi = sampler.sample(4)

    assert sample_single.benchmark_data.max_rewrite_attempts == 20
    assert sample_multi.benchmark_data.max_rewrite_attempts == 50
