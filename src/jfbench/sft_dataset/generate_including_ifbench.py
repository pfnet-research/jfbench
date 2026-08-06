from __future__ import annotations

import asyncio
from collections import deque
from contextlib import contextmanager
import logging
from typing import Any
from typing import Callable
from typing import cast
from typing import Coroutine
from typing import Iterator
from typing import Sequence

import jfbench.sft_dataset.generate as base_generate
from jfbench.sft_dataset.generate import ConstraintFactoryWrapper
from jfbench.sft_dataset.generate import ConstraintInventory
from jfbench.sft_dataset.generate import ConstraintSampler
from jfbench.sft_dataset.generate import SampledExample
from jfbench.sft_dataset.generate import SingleConstraintPlan


logger = logging.getLogger(__name__)
REWRITE_ATTEMPTS_MULTIPLIER = 2


class IfbenchConstraintSampler(ConstraintSampler):
    def __init__(
        self,
        *,
        prompt_source: str,
        prompts: list[Any],
        inventory: ConstraintInventory,
        judge_client: Any,
        rng: Any,
    ) -> None:
        super().__init__(
            prompt_source=prompt_source,
            prompts=prompts,
            inventory=inventory,
            judge_client=judge_client,
            rng=rng,
        )
        self._ifbench_single_plans = [
            plan for plan in self._single_plans if plan.wrapper.group == "Ifbench"
        ]
        if not self._ifbench_single_plans:
            raise ValueError("No Ifbench constraint wrappers available.")
        self._ifbench_single_queue = self._shuffle_ifbench_single_plans()

    def sample(self, n_constraints: int) -> SampledExample:
        if n_constraints <= 0:
            raise ValueError("n_constraints must be positive.")
        if n_constraints == 1:
            return self._sample_single_ifbench()
        if n_constraints in {2, 4, 8}:
            return self._sample_multi_with_ifbench(n_constraints)
        raise ValueError(f"Unsupported constraint count: {n_constraints}")

    def _sample_single_ifbench(self) -> SampledExample:
        plan = self._next_ifbench_single_plan()
        prompt = self.prompts[plan.prompt_index]
        seed = int(self.rng.integers(0, 2**63 - 1))
        constraint = plan.wrapper.build(
            self.judge_client,
            prompt.document,
            seed=seed,
        )
        benchmark = self._build_benchmark(plan.prompt_index, [constraint])
        benchmark.max_rewrite_attempts = max(
            benchmark.max_rewrite_attempts * REWRITE_ATTEMPTS_MULTIPLIER,
            20,
        )
        return SampledExample(
            prompt_source=self.prompt_source,
            prompt_index=plan.prompt_index,
            prompt_id=self._prompt_id(plan.prompt_index),
            benchmark_data=benchmark,
        )

    def _next_ifbench_single_plan(self) -> SingleConstraintPlan:
        if not self._ifbench_single_plans:
            raise ValueError("No Ifbench single-constraint plans available.")
        if not self._ifbench_single_queue:
            self._ifbench_single_queue = self._shuffle_ifbench_single_plans()
        return self._ifbench_single_queue.popleft()

    def _shuffle_ifbench_single_plans(self) -> deque[SingleConstraintPlan]:
        if len(self._ifbench_single_plans) <= 1:
            return deque(self._ifbench_single_plans)
        order = self.rng.permutation(len(self._ifbench_single_plans))
        return deque(self._ifbench_single_plans[idx] for idx in order)

    def _sample_multi_with_ifbench(self, n_constraints: int) -> SampledExample:
        prompt_index = int(self.rng.integers(0, len(self.prompts)))
        prompt = self.prompts[prompt_index]
        document = prompt.document
        constraints: list[Any] = []

        format_constraint = self.inventory.instantiate(
            "Format", self.judge_client, document, self.rng
        )
        constraints.append(format_constraint)

        ifbench_constraint = self._sample_non_conflicting_constraint_from_group(
            current_constraints=constraints,
            group="Ifbench",
            document=document,
        )
        constraints.append(ifbench_constraint)

        while len(constraints) < n_constraints:
            constraint = self._sample_non_conflicting_constraint(constraints, document)
            constraints.append(constraint)

        if len(constraints) > 1:
            order = self.rng.permutation(len(constraints))
            constraints = [constraints[idx] for idx in order]
        benchmark = self._build_benchmark(prompt_index, constraints)
        benchmark.max_rewrite_attempts = max(
            benchmark.max_rewrite_attempts * REWRITE_ATTEMPTS_MULTIPLIER,
            50,
        )
        return SampledExample(
            prompt_source=self.prompt_source,
            prompt_index=prompt_index,
            prompt_id=self._prompt_id(prompt_index),
            benchmark_data=benchmark,
        )

    def _sample_non_conflicting_constraint_from_group(
        self,
        *,
        current_constraints: list[Any],
        group: str,
        document: str,
    ) -> Any:
        wrappers = self.inventory.wrappers(group)
        max_attempts = max(len(wrappers) * 2, 10)
        for _ in range(max_attempts):
            wrapper: ConstraintFactoryWrapper = self.rng.choice(wrappers)
            seed = int(self.rng.integers(0, 2**63 - 1))
            constraint = wrapper.build(self.judge_client, document, seed=seed)
            if self._has_conflict(current_constraints, constraint):
                continue
            return constraint
        raise RuntimeError(f"Unable to sample a non-conflicting constraint from group '{group}'.")


@contextmanager
def _patched_runtime() -> Iterator[None]:
    original_sampler = base_generate.ConstraintSampler
    setattr(base_generate, "ConstraintSampler", IfbenchConstraintSampler)
    try:
        yield
    finally:
        setattr(base_generate, "ConstraintSampler", original_sampler)


def parse_args(argv: Sequence[str] | None = None) -> Any:
    return base_generate.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    log_level_value = getattr(logging, args.log_level, logging.WARNING)
    logging.basicConfig(
        level=log_level_value,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    generate_async = cast(
        "Callable[[Any], Coroutine[Any, Any, None]]",
        getattr(base_generate, "_generate_async"),
    )
    with _patched_runtime():
        asyncio.run(generate_async(args))


__all__ = [
    "IfbenchConstraintSampler",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    main()
