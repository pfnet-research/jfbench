from __future__ import annotations

import argparse
import asyncio
from contextlib import contextmanager
import logging
from typing import Any
from typing import Callable
from typing import cast
from typing import Coroutine
from typing import Iterator
from typing import Sequence
from typing import TYPE_CHECKING

from jfbench.prompts.ja_stackoverflow import get_all_ja_stackoverflow_prompts
import jfbench.sft_dataset.generate as sft_generate


if TYPE_CHECKING:
    from jfbench.benchmark.build import ConstraintCollections
    from jfbench.benchmark.build import ConstraintFactory
    from jfbench.protocol import Prompt


PROMPT_LOADERS: dict[str, Callable[[], list[Prompt]]] = {
    "ja_stackoverflow": cast(
        "Callable[[], list[Prompt]]",
        get_all_ja_stackoverflow_prompts,
    ),
}


class TargetGroupConstraintSampler(sft_generate.ConstraintSampler):
    target_group = "Logic"
    max_resample_attempts = 10000

    def sample(self, n_constraints: int) -> sft_generate.SampledExample:
        for _ in range(self.max_resample_attempts):
            sample = super().sample(n_constraints)
            groups = sample.benchmark_data.meta_data.constraint_groups
            if self.target_group in groups:
                return sample
        raise RuntimeError(
            f"Unable to sample record containing target group '{self.target_group}'."
        )


def _wrap_factories(
    group: str,
    factories: Sequence[ConstraintFactory],
) -> list[sft_generate.ConstraintFactoryWrapper]:
    return [
        sft_generate.ConstraintFactoryWrapper(group=group, mode="rule", factory=factory)
        for factory in factories
    ]


class RuleOnlyConstraintInventory(sft_generate.ConstraintInventory):
    @staticmethod
    def _build_wrappers(
        collections: ConstraintCollections,
    ) -> dict[str, list[sft_generate.ConstraintFactoryWrapper]]:
        registry: dict[str, list[sft_generate.ConstraintFactoryWrapper]] = {}
        registry["Format"] = _wrap_factories("Format", collections.format)
        registry["Character"] = _wrap_factories("Character", collections.character)
        registry["Content"] = _wrap_factories(
            "Content",
            collections.rule_content,
        )
        registry["Length"] = _wrap_factories("Length", collections.length)
        registry["Logic"] = _wrap_factories("Logic", collections.logic)
        registry["Notation"] = _wrap_factories("Notation", collections.notation)
        registry["Processing"] = _wrap_factories(
            "Processing",
            collections.rule_processing,
        )
        registry["Style"] = _wrap_factories(
            "Style",
            collections.rule_style,
        )
        registry["Ifbench"] = _wrap_factories(
            "Ifbench",
            collections.ifbench_count
            + collections.ifbench_format
            + collections.ifbench_ratio
            + collections.ifbench_repeat
            + collections.ifbench_sentence
            + collections.ifbench_words,
        )
        sft_generate.logger.info(
            "Registered number of constraints per group: "
            + str({group: len(wrappers) for group, wrappers in registry.items()})
        )
        sft_generate.logger.info(
            "Total registered constraint wrappers: "
            + str(sum(len(wrappers) for wrappers in registry.values()))
        )
        return registry


@contextmanager
def _patched_runtime() -> Iterator[None]:
    original_loaders = sft_generate.PROMPT_LOADERS
    original_inventory = sft_generate.ConstraintInventory
    original_sampler = sft_generate.ConstraintSampler
    sft_generate.PROMPT_LOADERS = PROMPT_LOADERS
    setattr(sft_generate, "ConstraintInventory", RuleOnlyConstraintInventory)
    setattr(sft_generate, "ConstraintSampler", TargetGroupConstraintSampler)
    try:
        yield
    finally:
        sft_generate.PROMPT_LOADERS = original_loaders
        setattr(sft_generate, "ConstraintInventory", original_inventory)
        setattr(sft_generate, "ConstraintSampler", original_sampler)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--target-group", type=str, default="Logic")
    known_args, remaining = parser.parse_known_args(argv)
    with _patched_runtime():
        args = sft_generate.parse_args(remaining)
    args.prompt_source = "ja_stackoverflow"
    args.target_group = known_args.target_group
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    log_level_value = getattr(logging, args.log_level, logging.WARNING)
    logging.basicConfig(
        level=log_level_value,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    generate_async = cast(
        "Callable[[argparse.Namespace], Coroutine[Any, Any, None]]",
        getattr(sft_generate, "_generate_async"),
    )
    TargetGroupConstraintSampler.target_group = args.target_group
    with _patched_runtime():
        groups = set(RuleOnlyConstraintInventory(constraint_set=args.constraint_set).groups)
    if args.target_group not in groups:
        available = ", ".join(sorted(groups))
        raise ValueError(
            f"Unknown target group '{args.target_group}'. Available groups: {available}"
        )
    with _patched_runtime():
        asyncio.run(generate_async(args))


__all__ = [
    "PROMPT_LOADERS",
    "RuleOnlyConstraintInventory",
    "TargetGroupConstraintSampler",
    "main",
    "parse_args",
]

if __name__ == "__main__":
    main()
