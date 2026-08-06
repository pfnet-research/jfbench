from __future__ import annotations

import argparse
import logging
from pathlib import Path
import random
from typing import Any
from typing import cast
from typing import TYPE_CHECKING

from datasets import Dataset

from jfbench.sft_dataset.generate import ConstraintInventory
from jfbench.sft_dataset.generate import PROMPT_LOADERS


if TYPE_CHECKING:  # pragma: no cover - type checking only
    from jfbench.llm import LLMClient
    from jfbench.protocol import Constraint
    from jfbench.protocol import Prompt


logger = logging.getLogger(__name__)
_PROMPT_CACHE: dict[str, list[Prompt]] = {}


class _RandomizerLLMClient:
    async def async_ask(self, prompts: list[str]) -> tuple[list[str], list[None]]:
        raise RuntimeError("Randomizer client cannot perform LLM calls.")


def _build_wrapper_map(inventory: ConstraintInventory) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    dummy_client = cast("LLMClient", _RandomizerLLMClient())
    for wrapper in inventory.all_wrappers:
        constraint = wrapper.build(dummy_client, "", seed=0)
        mapping[constraint.__class__.__name__] = wrapper
    return mapping


def _extract_existing_instructions(row: dict[str, Any], count: int) -> list[str]:
    existing = row.get("constraint_instructions") or []
    instructions = [str(value) for value in existing][:count]
    if len(instructions) == count:
        return instructions
    constraints_meta = row.get("constraints") or []
    fallback = [str(item.get("instructions", "")) for item in constraints_meta][:count]
    if len(fallback) == count:
        return fallback
    return (instructions + fallback + [""])[:count]


def _resolve_prompt(prompt_source: str, prompt_index: int) -> Prompt:
    prompts = _PROMPT_CACHE.get(prompt_source)
    if prompts is None:
        loader = PROMPT_LOADERS.get(prompt_source)
        if loader is None:
            raise ValueError(f"Unsupported prompt source '{prompt_source}'.")
        prompts = loader()
        _PROMPT_CACHE[prompt_source] = prompts
    if not 0 <= prompt_index < len(prompts):
        raise ValueError(f"Prompt index {prompt_index} out of range for source '{prompt_source}'.")
    return prompts[prompt_index]


def _randomize_constraint(
    *,
    wrapper: Any,
    document: str,
    old_instruction: str,
    rng: random.Random,
    judge_client: _RandomizerLLMClient,
) -> Constraint:
    for _ in range(16):
        seed = rng.getrandbits(63)
        constraint = wrapper.build(judge_client, document, seed=seed)
        new_instruction = constraint.instructions()
        if not old_instruction or new_instruction != old_instruction:
            return constraint
    return constraint


def _update_constraint_records(row: dict[str, Any], instructions: list[str]) -> None:
    constraints_meta = row.get("constraints")
    if not isinstance(constraints_meta, list):
        return
    updated: list[dict[str, Any]] = []
    for idx, entry in enumerate(constraints_meta):
        entry_dict = dict(entry)
        if idx < len(instructions):
            entry_dict["instructions"] = instructions[idx]
        updated.append(entry_dict)
    row["constraints"] = updated


def randomize_dataset(input_dir: Path, output_dir: Path, *, seed: int | None = None) -> None:
    dataset = Dataset.load_from_disk(str(input_dir))
    rng = random.Random(seed)
    inventory = ConstraintInventory()
    wrapper_map = _build_wrapper_map(inventory)
    judge_client = _RandomizerLLMClient()

    def _process(row: dict[str, Any]) -> dict[str, Any]:
        constraint_types = list(row.get("constraint_types") or [])
        if not constraint_types:
            if "constraint_instructions" not in row:
                row["constraint_instructions"] = []
            row.setdefault("prompt", row.get("instruction", ""))
            return row

        existing_instructions = _extract_existing_instructions(row, len(constraint_types))
        document = str(row.get("prompt_document", ""))
        randomized_constraints: list[Constraint] = []
        updated_instructions: list[str] = []
        for idx, class_name in enumerate(constraint_types):
            wrapper = wrapper_map.get(class_name)
            if wrapper is None:
                raise ValueError(f"No constraint factory registered for '{class_name}'.")
            constraint = _randomize_constraint(
                wrapper=wrapper,
                document=document,
                old_instruction=existing_instructions[idx]
                if idx < len(existing_instructions)
                else "",
                rng=rng,
                judge_client=judge_client,
            )
            randomized_constraints.append(constraint)
            updated_instructions.append(constraint.instructions())

        prompt_source = str(row.get("prompt_source", "")).strip()
        if not prompt_source:
            raise ValueError("prompt_source is required to rebuild prompt text.")
        prompt_index = int(row.get("prompt_index", 0))
        prompt = _resolve_prompt(prompt_source, prompt_index)
        prompt_text = prompt.text(randomized_constraints)

        row["constraint_instructions"] = updated_instructions
        row["instruction"] = prompt_text
        row["prompt"] = prompt_text
        _update_constraint_records(row, updated_instructions)
        return row

    randomized = dataset.map(_process, desc="Randomizing constraints")
    randomized.save_to_disk(str(output_dir))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Randomize constraint instructions in a dataset.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Path to the input dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path to the output dataset directory.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    randomize_dataset(input_dir=args.input_dir, output_dir=args.output_dir, seed=args.seed)


if __name__ == "__main__":  # pragma: no cover - manual invocation
    main()
