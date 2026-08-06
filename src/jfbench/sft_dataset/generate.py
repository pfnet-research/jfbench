from __future__ import annotations

import argparse
import asyncio
from collections import deque
from dataclasses import dataclass
from dataclasses import field
import inspect
import json
import logging
from pathlib import Path
import re
import shutil
import tempfile
import textwrap
from typing import Any
from typing import Callable
from typing import cast
from typing import Iterable
from typing import Literal
from typing import Mapping
from typing import Optional
from typing import Protocol
from typing import Sequence
from typing import TYPE_CHECKING
import uuid

from datasets import concatenate_datasets
from datasets import Dataset
import numpy as np
from openai import APIConnectionError
from openai import APITimeoutError
from openai import AsyncOpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

from jfbench.benchmark.build import BenchmarkData
from jfbench.benchmark.build import ConstraintCollections
from jfbench.benchmark.build import ConstraintFactory
from jfbench.benchmark.build import ConstraintSetName
from jfbench.benchmark.build import get_constraint_collections
from jfbench.llm import extract_reasoning_content
from jfbench.llm import LLMClient
from jfbench.prompts.ifbench import get_all_ifbench_prompts
from jfbench.prompts.ja_stackoverflow import get_all_ja_stackoverflow_prompts


if TYPE_CHECKING:
    from jfbench.protocol import Constraint
    from jfbench.protocol import Prompt


logger = logging.getLogger(__name__)


MAX_REASONING_CONTENT_TOKENS = 65536
CANDIDATES_PER_INSTRUCTION = 10
REWARD_TOKEN_LIMIT_DEFAULT = 512
REWARD_CONCURRENCY_DEFAULT = 40
REWARD_BASE_URL_DEFAULT = "http://localhost:8000/v1"
REWARD_API_KEY_DEFAULT = "unused"
REWARD_MODEL_DEFAULT = "openai/gpt-oss-120b"
TIMEOUT = 120000
LOGPROB_CHUNK_SIZE = 1024
LOGPROB_CHUNK_OVERLAP = 32


PROMPT_LOADERS: dict[str, Callable[[], list[Prompt]]] = {
    "ifbench": cast("Callable[[], list[Prompt]]", get_all_ifbench_prompts),
    "ja_stackoverflow": cast("Callable[[], list[Prompt]]", get_all_ja_stackoverflow_prompts),
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model: str
    temperature: float = 0.6
    provider: Literal["openrouter", "vllm"] = "openrouter"
    extra_body: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConstraintFactoryWrapper:
    group: str
    mode: Literal["rule", "llm"]
    factory: ConstraintFactory

    def build(
        self,
        judge_client: LLMClient,
        document: str,
        *,
        seed: int | None = None,
    ) -> Constraint:
        factory = self.factory
        if self.mode == "rule":
            signature = inspect.signature(factory)
            accepts_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            if "document" in signature.parameters or accepts_kwargs:
                return factory(seed=seed, document=document)
            return factory(seed=seed)
        return factory(judge_client, document, seed=seed)


def _wrap_factories(
    group: str,
    mode: Literal["rule", "llm"],
    factories: Sequence[ConstraintFactory],
) -> list[ConstraintFactoryWrapper]:
    return [
        ConstraintFactoryWrapper(group=group, mode=mode, factory=factory) for factory in factories
    ]


class ConstraintInventory:
    def __init__(self, *, constraint_set: ConstraintSetName = "train") -> None:
        self.constraint_set = constraint_set
        collections = get_constraint_collections(constraint_set)
        self._group_to_wrappers = self._build_wrappers(collections)
        self._all_wrappers: tuple[ConstraintFactoryWrapper, ...] = tuple(
            wrapper for wrappers in self._group_to_wrappers.values() for wrapper in wrappers
        )

    @staticmethod
    def _build_wrappers(
        collections: ConstraintCollections,
    ) -> dict[str, list[ConstraintFactoryWrapper]]:
        registry: dict[str, list[ConstraintFactoryWrapper]] = {}
        registry["Format"] = _wrap_factories("Format", "rule", collections.format)
        registry["Character"] = _wrap_factories("Character", "rule", collections.character)
        registry["Content"] = _wrap_factories(
            "Content",
            "rule",
            collections.rule_content,
        ) + _wrap_factories("Content", "llm", collections.llm_content)
        registry["Length"] = _wrap_factories("Length", "rule", collections.length)
        registry["Logic"] = _wrap_factories("Logic", "rule", collections.logic)
        registry["MetaOutput"] = _wrap_factories("MetaOutput", "llm", collections.meta_output)
        registry["Notation"] = _wrap_factories("Notation", "rule", collections.notation)
        registry["Processing"] = _wrap_factories(
            "Processing",
            "rule",
            collections.rule_processing,
        ) + _wrap_factories("Processing", "llm", collections.llm_processing)
        registry["Structure"] = _wrap_factories("Structure", "llm", collections.structure)
        registry["Style"] = _wrap_factories(
            "Style",
            "rule",
            collections.rule_style,
        ) + _wrap_factories("Style", "llm", collections.llm_style)
        registry["Ifbench"] = _wrap_factories(
            "Ifbench",
            "rule",
            collections.ifbench_count
            + collections.ifbench_format
            + collections.ifbench_ratio
            + collections.ifbench_repeat
            + collections.ifbench_sentence
            + collections.ifbench_words,
        )
        logger.info(
            "Registered number of constraints per group: "
            + str({group: len(wrappers) for group, wrappers in registry.items()})
        )
        logger.info(
            f"Total registered constraint wrappers: {sum(len(wrappers) for wrappers in registry.values())}"
        )
        return registry

    @property
    def groups(self) -> tuple[str, ...]:
        return tuple(sorted(self._group_to_wrappers.keys()))

    def wrappers(self, group: str) -> list[ConstraintFactoryWrapper]:
        wrappers = self._group_to_wrappers.get(group)
        if not wrappers:
            raise ValueError(f"No constraint wrappers registered for group '{group}'.")
        return wrappers

    @property
    def all_wrappers(self) -> tuple[ConstraintFactoryWrapper, ...]:
        return self._all_wrappers

    def random_wrapper(self, rng: np.random.Generator) -> ConstraintFactoryWrapper:
        if not self._all_wrappers:
            raise ValueError("No constraint wrappers available.")
        return rng.choice(self._all_wrappers)

    def instantiate(
        self,
        group: str,
        judge_client: LLMClient,
        document: str,
        rng: np.random.Generator,
    ) -> Constraint:
        options = self.wrappers(group)
        wrapper = rng.choice(options)
        seed = int(rng.integers(0, 2**63 - 1))
        return wrapper.build(judge_client, document, seed=seed)

    def iter_single_constraint_plans(
        self,
        prompt_count: int,
    ) -> list[tuple[int, ConstraintFactoryWrapper]]:
        plans: list[tuple[int, ConstraintFactoryWrapper]] = []
        for prompt_idx in range(prompt_count):
            for wrappers in self._group_to_wrappers.values():
                for wrapper in wrappers:
                    plans.append((prompt_idx, wrapper))
        return plans


@dataclass
class SingleConstraintPlan:
    prompt_index: int
    wrapper: ConstraintFactoryWrapper


@dataclass
class SampledExample:
    prompt_source: str
    prompt_index: int
    prompt_id: str
    benchmark_data: BenchmarkData


class ConstraintSampler:
    def __init__(
        self,
        *,
        prompt_source: str,
        prompts: list[Prompt],
        inventory: ConstraintInventory,
        judge_client: LLMClient,
        rng: np.random.Generator,
    ) -> None:
        if not prompts:
            raise ValueError("At least one prompt is required for sampling.")
        self.prompt_source = prompt_source
        self.prompts = prompts
        self.inventory = inventory
        self.judge_client = judge_client
        self.rng = rng
        plans = [
            SingleConstraintPlan(prompt_idx, wrapper)
            for prompt_idx, wrapper in inventory.iter_single_constraint_plans(len(prompts))
        ]
        self._single_plans: list[SingleConstraintPlan] = plans
        self._single_queue: deque[SingleConstraintPlan] = self._shuffle_single_plans()
        self._sample_counter = 0

    def sample(self, n_constraints: int) -> SampledExample:
        if n_constraints <= 0:
            raise ValueError("n_constraints must be positive.")
        if n_constraints == 1:
            plan = self._next_single_plan()
            prompt = self.prompts[plan.prompt_index]
            seed = int(self.rng.integers(0, 2**63 - 1))
            constraint = plan.wrapper.build(
                self.judge_client,
                prompt.document,
                seed=seed,
            )
            benchmark = self._build_benchmark(plan.prompt_index, [constraint])
            return SampledExample(
                prompt_source=self.prompt_source,
                prompt_index=plan.prompt_index,
                prompt_id=self._prompt_id(plan.prompt_index),
                benchmark_data=benchmark,
            )
        if n_constraints in {2, 4, 8}:
            return self._sample_multi(n_constraints)
        raise ValueError(f"Unsupported constraint count: {n_constraints}")

    def _next_single_plan(self) -> SingleConstraintPlan:
        if not self._single_plans:
            raise ValueError("No single-constraint plans available.")
        if not self._single_queue:
            self._single_queue = self._shuffle_single_plans()
        return self._single_queue.popleft()

    def _shuffle_single_plans(self) -> deque[SingleConstraintPlan]:
        if len(self._single_plans) <= 1:
            return deque(self._single_plans)
        order = self.rng.permutation(len(self._single_plans))
        return deque(self._single_plans[idx] for idx in order)

    def _sample_multi(self, n_constraints: int) -> SampledExample:
        prompt_index = int(self.rng.integers(0, len(self.prompts)))
        prompt = self.prompts[prompt_index]
        document = prompt.document
        constraints: list[Constraint] = []
        format_constraint = self.inventory.instantiate(
            "Format", self.judge_client, document, self.rng
        )
        constraints.append(format_constraint)

        while len(constraints) < n_constraints:
            constraint = self._sample_non_conflicting_constraint(constraints, document)
            constraints.append(constraint)

        if len(constraints) > 1:
            order = self.rng.permutation(len(constraints))
            constraints = [constraints[idx] for idx in order]
        benchmark = self._build_benchmark(prompt_index, constraints)
        return SampledExample(
            prompt_source=self.prompt_source,
            prompt_index=prompt_index,
            prompt_id=self._prompt_id(prompt_index),
            benchmark_data=benchmark,
        )

    def _sample_non_conflicting_constraint(
        self,
        current_constraints: list[Constraint],
        document: str,
    ) -> Constraint:
        max_attempts = max(len(self.inventory.all_wrappers) * 2, 10)
        for _ in range(max_attempts):
            wrapper = self.inventory.random_wrapper(self.rng)
            seed = int(self.rng.integers(0, 2**63 - 1))
            constraint = wrapper.build(self.judge_client, document, seed=seed)
            if self._has_conflict(current_constraints, constraint):
                continue
            return constraint
        raise RuntimeError("Unable to sample a non-conflicting constraint.")

    @staticmethod
    def _has_conflict(
        constraints: list[Constraint],
        candidate: Constraint,
    ) -> bool:
        candidate_name = candidate.__class__.__name__
        candidate_competitives = getattr(candidate, "competitives", []) or []
        for existing in constraints:
            existing_name = existing.__class__.__name__
            if existing_name == candidate_name:
                return True
            existing_competitives = getattr(existing, "competitives", []) or []
            if existing_name in candidate_competitives or candidate_name in existing_competitives:
                return True
        return False

    def _build_benchmark(self, prompt_index: int, constraints: list[Constraint]) -> BenchmarkData:
        prompt = self.prompts[prompt_index]
        data_id = f"{self.prompt_source}-{prompt_index}-{self._sample_counter}"
        self._sample_counter += 1
        meta = BenchmarkData.build_meta_data(
            prompt_source=self.prompt_source,
            data_id=data_id,
            prompt=prompt,
            constraints=constraints,
            constraint_set=self.inventory.constraint_set,
        )
        return BenchmarkData(prompt=prompt, constraints=constraints, meta_data=meta)

    def _prompt_id(self, prompt_index: int) -> str:
        return f"{self.prompt_source}-{prompt_index}"


@dataclass
class ModelRunner:
    spec: ModelSpec
    client: LLMClient

    async def generate(self, prompt: str) -> tuple[str, str]:
        responses, response_details = await self.client.async_ask([prompt])
        response = responses[0]
        reasoning_content = extract_reasoning_content(
            self.client.provider, response_details[0] if response_details else None
        )
        return response, reasoning_content


def _extract_token_logprob(entry: Mapping[str, Any]) -> tuple[str, float]:
    token = entry.get("token") or entry.get("text") or ""
    if "logprob" in entry and isinstance(entry["logprob"], (int, float)):
        return token, float(entry["logprob"])
    top_candidates = entry.get("top_logprobs") or []
    for candidate in top_candidates:
        if (candidate.get("token") == token) or (candidate.get("text") == token):
            return token, float(candidate.get("logprob"))
    if top_candidates:
        fallback = top_candidates[0]
        return (
            fallback.get("token") or fallback.get("text") or "",
            float(fallback.get("logprob")),
        )
    return token, 0.0


def _chunk_by_tokens(
    token_ids: list[int],
    *,
    chunk_size: int,
    overlap: int,
) -> list[list[int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")
    chunks: list[list[int]] = []
    i = 0
    n = len(token_ids)
    while i < n:
        chunks.append(token_ids[i : min(i + chunk_size, n)])
        if i + chunk_size >= n:
            break
        i += chunk_size - overlap
    return chunks


async def _completions_prompt_logprobs_chunked(
    client: AsyncOpenAI,
    *,
    model: str,
    token_ids: list[int],
    tokenizer: Any,
    chunk_size: int,
    overlap: int,
) -> dict[str, Any]:
    chunks = _chunk_by_tokens(token_ids, chunk_size=chunk_size, overlap=overlap)
    all_tokens: list[str] = []
    all_logprobs: list[float] = []

    for idx, ids in enumerate(chunks):
        prompt_chunk = tokenizer.decode(ids, clean_up_tokenization_spaces=False)
        while True:
            try:
                resp = await client.completions.create(
                    model=model,
                    prompt=prompt_chunk,
                    max_tokens=0,
                    temperature=0,
                    logprobs=1,
                    echo=True,
                )
                break
            except APITimeoutError as exc:
                logger.warning(f"Timeout error during logprob request, retrying: {exc}")
                continue
            except APIConnectionError as exc:
                logger.warning(f"Connection error during logprob request, retrying: {exc}")

        logprob_block: Optional[Iterable[Mapping[str, Any]]] = None
        choice = resp.choices[0]
        raw_logprobs = getattr(choice, "logprobs", None)
        if raw_logprobs and isinstance(raw_logprobs, Mapping):
            content = raw_logprobs.get("content")
            if content:
                logprob_block = content
        if logprob_block is None and raw_logprobs:
            tokens_attr = getattr(raw_logprobs, "tokens", None)
            token_lps_attr = getattr(raw_logprobs, "token_logprobs", None)
            if tokens_attr is not None and token_lps_attr is not None:
                logprob_block = [
                    {"token": tok, "logprob": lp}
                    for tok, lp in zip(tokens_attr, token_lps_attr, strict=False)
                ]
        if not logprob_block:
            raise RuntimeError("logprobs not returned; ensure server supports logprobs.")

        tokens: list[str] = []
        token_lps: list[float] = []
        for entry in logprob_block:
            tok, lp = _extract_token_logprob(entry)
            tokens.append(tok)
            token_lps.append(lp)

        if idx > 0 and overlap > 0:
            tokens = tokens[overlap:]
            token_lps = token_lps[overlap:]

        all_tokens.extend(tokens)
        all_logprobs.extend(token_lps)

    return {
        "tokens": all_tokens,
        "token_logprobs": all_logprobs,
        "sum_logprob": float(sum(all_logprobs)),
    }


def _build_scoring_prompt(prompt: str, answer: str) -> tuple[str, int]:
    prefix = f"User:\n{prompt}\nAssistant:\n"
    full_text = prefix + answer
    start_offset = len(prefix)
    return full_text, start_offset


async def _score_one_with_echo(
    client: AsyncOpenAI,
    model: str,
    *,
    token_ids: list[int],
    candidate_start_idx: int,
    tokenizer: Any,
    token_limit: int,
    chunk_size: int = LOGPROB_CHUNK_SIZE,
    overlap: int = LOGPROB_CHUNK_OVERLAP,
) -> tuple[float, int]:
    logprob_result = await _completions_prompt_logprobs_chunked(
        client,
        model=model,
        token_ids=token_ids,
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        overlap=overlap,
    )
    tokens = logprob_result["tokens"]
    token_lps = logprob_result["token_logprobs"]
    collected: list[float] = []
    count = 0
    for idx, (tok, lp) in enumerate(zip(tokens, token_lps, strict=False)):
        if idx < candidate_start_idx:
            continue
        if tok.strip() == "":
            continue
        collected.append(float(lp))
        count += 1
        if count >= token_limit:
            break
    if not collected:
        return float("-inf"), 0
    return sum(collected) / len(collected), len(collected)


class RewardScorer:
    def __init__(
        self,
        *,
        base_url: str = REWARD_BASE_URL_DEFAULT,
        api_key: str = REWARD_API_KEY_DEFAULT,
        model: str = REWARD_MODEL_DEFAULT,
        token_limit: int = REWARD_TOKEN_LIMIT_DEFAULT,
        concurrency: int = REWARD_CONCURRENCY_DEFAULT,
    ) -> None:
        if token_limit <= 0:
            raise ValueError("reward_token_limit must be positive.")
        if concurrency <= 0:
            raise ValueError("reward_concurrency must be positive.")
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=TIMEOUT)
        self.model = model
        self.token_limit = token_limit
        self._semaphore = asyncio.Semaphore(concurrency)
        self._tokenizer: Any | None = None

    def token_count(self, text: str) -> int:
        if not text:
            return 0
        if AutoTokenizer is None:
            raise RuntimeError("transformers is required for reward scoring.")
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model, use_fast=True)
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    async def score_many(self, prompt: str, answers: list[str]) -> list[float]:
        async def _run(answer: str) -> float:
            async with self._semaphore:
                if self._tokenizer is None:
                    if AutoTokenizer is None:
                        raise RuntimeError("transformers is required for reward scoring.")
                    self._tokenizer = AutoTokenizer.from_pretrained(self.model, use_fast=True)
                text, _ = _build_scoring_prompt(prompt, answer)
                token_ids = self._tokenizer.encode(text, add_special_tokens=False)
                prefix_tokens = self._tokenizer.encode(
                    f"User:\n{prompt}\nAssistant:\n", add_special_tokens=False
                )
                score, _ = await _score_one_with_echo(
                    self.client,
                    self.model,
                    token_ids=token_ids,
                    candidate_start_idx=len(prefix_tokens),
                    tokenizer=self._tokenizer,
                    token_limit=self.token_limit,
                )
                return score

        return await asyncio.gather(*[_run(answer) for answer in answers])


RepetitionLevel = Literal["none", "minor", "major"]


@dataclass(frozen=True)
class ReasoningReview:
    repetition_level: RepetitionLevel
    is_rambling: bool


class LLMJudge:
    def __init__(self, client: LLMClient) -> None:
        self.client = client

    async def is_refusal(self, response: str) -> bool:
        prompt = (
            "You are a judge. Determine whether the assistant response is a refusal or a non-answer. "
            "Return only 'yes' or 'no'.\n"
            f"Response:\n{response}\n"
            "Answer:"
        )
        verdict = await self._ask(prompt)
        return self._parse_yes_no(verdict, default=False)

    async def repetition_level(self, text: str, *, label: str) -> RepetitionLevel:
        prompt = (
            "You are a judge. Rate repetition in the provided text. "
            "Return only one word: none, minor, or major.\n"
            "Definitions: major = many repeated phrases or word sequences; minor = some repetition "
            "but still readable; none = no meaningful repetition.\n"
            f"Text type: {label}\n"
            f"Text:\n{text}\n"
            "Answer:"
        )
        verdict = await self._ask(prompt)
        return self._parse_repetition_level(verdict)

    async def review_reasoning(self, reasoning_content: str) -> ReasoningReview:
        prompt = (
            "You are a judge. Review the reasoning content for repetition and rambling. "
            "Return exactly: repetition=<none|minor|major>; rambling=<yes|no>.\n"
            "Definitions of repetition: major = many repeated phrases or word sequences; minor = some repetition "
            "but still readable; none = no meaningful repetition.\n"
            "Definitions of rambling: unfocused or overly verbose reasoning.\n"
            f"Reasoning content:\n{reasoning_content}\n"
            "Answer:"
        )
        verdict = await self._ask(prompt)
        repetition_level = self._parse_repetition_level(verdict)
        rambling = self._parse_rambling(verdict)
        return ReasoningReview(repetition_level=repetition_level, is_rambling=rambling)

    async def _ask(self, prompt: str) -> str:
        responses, _ = await self.client.async_ask([prompt])
        return (responses[0] if responses else "").strip().lower()

    @staticmethod
    def _parse_yes_no(text: str, *, default: bool) -> bool:
        if text.startswith("yes"):
            return True
        if text.startswith("no"):
            return False
        match = re.search(r"\b(yes|no)\b", text)
        if match:
            return match.group(1) == "yes"
        return default

    @staticmethod
    def _parse_repetition_level(text: str) -> RepetitionLevel:
        lowered = text.strip().lower()
        if lowered.startswith("minor"):
            return "minor"
        if lowered.startswith("major"):
            return "major"
        if lowered.startswith("none"):
            return "none"
        match = re.search(r"\b(none|minor|major)\b", lowered)
        if match:
            return cast("RepetitionLevel", match.group(1))
        return "none"

    @staticmethod
    def _parse_rambling(text: str) -> bool:
        lowered = text.strip().lower()
        if "rambling=yes" in lowered:
            return True
        if "rambling=no" in lowered:
            return False
        match = re.search(r"\brambling\s*=\s*(yes|no)\b", lowered)
        if match:
            return match.group(1) == "yes"
        return False


class JudgeProtocol(Protocol):
    async def is_refusal(self, response: str) -> bool: ...

    async def repetition_level(self, text: str, *, label: str) -> RepetitionLevel: ...

    async def review_reasoning(self, reasoning_content: str) -> ReasoningReview: ...


class RewardScorerProtocol(Protocol):
    async def score_many(self, prompt: str, answers: list[str]) -> list[float]: ...

    def token_count(self, text: str) -> int: ...


@dataclass
class GenerationResult:
    model: str
    model_name: str
    temperature: float
    attempt: int
    response: str
    reasoning_content: str
    evaluation: dict[str, bool]


@dataclass
class ScoredCandidate:
    result: GenerationResult
    response_reward: float
    reasoning_reward: float
    average_reward: float


class InteractiveSkip(Exception):
    """Raised when the operator skips the remaining steps in interactive mode."""


class InteractiveController:
    def __init__(
        self,
        *,
        enabled: bool,
        prompt_func: Callable[[str], str] | None = None,
    ) -> None:
        self.enabled = enabled
        self._prompt_func = prompt_func

    async def confirm_sample(self, sample: SampledExample) -> bool:
        if not self.enabled:
            return True
        instruction_mode = sample.benchmark_data.meta_data.constraint_set
        constraint_lines = []
        for idx, constraint in enumerate(sample.benchmark_data.constraints, start=1):
            constraint_lines.append(
                f"{idx}. [{constraint.group}] {constraint.__class__.__name__}: "
                f"{constraint.instructions(train_or_test=instruction_mode)}"
            )
        formatted_constraints = "\n".join(constraint_lines) or "None"
        constraints_block = textwrap.indent(formatted_constraints, "  ")
        prompt_block = textwrap.indent(sample.benchmark_data.text(), "  ")
        logger.info(
            (
                "Sample ready for generation.\n"
                f"Prompt ID: {sample.prompt_id}\n"
                "Constraints:\n"
                f"{constraints_block}\n"
                "Prompt:\n"
                f"{prompt_block}\n"
            )
        )
        return await self._ask_yes_no("Proceed with generation? [y/n]: ")

    async def confirm_generation(
        self,
        *,
        sample: SampledExample,
        spec: ModelSpec,
        attempt: int,
        response: str,
    ) -> bool:
        if not self.enabled:
            return True
        response_block = textwrap.indent(response, "  ")
        logger.info(
            (
                "Generation finished.\n"
                f"Prompt ID: {sample.prompt_id}\n"
                f"Model: {spec.model} (attempt {attempt})\n"
                "Response:\n"
                f"{response_block}"
            )
        )
        return await self._ask_yes_no("Proceed with evaluation? [y/n]: ")

    async def confirm_rewrite(
        self,
        *,
        sample: SampledExample,
        spec: ModelSpec,
        attempt: int,
        response: str,
    ) -> bool:
        if not self.enabled:
            return True
        response_block = textwrap.indent(response, "  ")
        logger.info(
            (
                "Rewrite requested.\n"
                f"Prompt ID: {sample.prompt_id}\n"
                f"Model: {spec.model} (attempt {attempt})\n"
                "Failed response:\n"
                f"{response_block}"
            )
        )
        return await self._ask_yes_no("Attempt rewrite for this response? [y/n]: ")

    async def confirm_save(self, record: Mapping[str, Any]) -> bool:
        if not self.enabled:
            return True
        prompt_id = str(record.get("prompt_id", ""))
        model = str(record.get("model", ""))
        response = textwrap.indent(str(record.get("response", "")), "  ")
        evaluation = record.get("evaluation", {})
        if isinstance(evaluation, Mapping):
            evaluation_lines = [f"{key}: {value}" for key, value in evaluation.items()]
            evaluation_text = "\n".join(evaluation_lines) or "None"
        else:
            evaluation_text = str(evaluation)
        evaluation_block = textwrap.indent(evaluation_text, "  ")
        logger.info(
            (
                "Ready to save record.\n"
                f"Prompt ID: {prompt_id}\n"
                f"Model: {model}\n"
                "Response:\n"
                f"{response}\n"
                "Evaluation:\n"
                f"{evaluation_block}"
            )
        )
        return await self._ask_yes_no("Save this record? [y/n]: ")

    def _prompt(self, message: str) -> str:
        if self._prompt_func is not None:
            return self._prompt_func(message)
        return input(message)

    async def _ask_yes_no(self, message: str) -> bool:
        loop = asyncio.get_running_loop()
        while True:
            answer = await loop.run_in_executor(None, self._prompt, message)
            normalized = answer.strip().lower()
            if normalized in {"y", "yes"}:
                return True
            if normalized in {"n", "no"}:
                return False
            logger.warning("Please answer with 'y' or 'n'.")


class CandidateProcessor:
    def __init__(
        self,
        model_runner: ModelRunner,
        *,
        max_attempts: int,
        judge: JudgeProtocol,
        reward_scorer: RewardScorerProtocol,
        candidate_count: int = CANDIDATES_PER_INSTRUCTION,
        interactive_controller: InteractiveController | None = None,
        seed: int | None = None,
    ) -> None:
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive.")
        if candidate_count <= 0:
            raise ValueError("candidate_count must be positive.")
        self.model_runner = model_runner
        self.max_attempts = max_attempts
        self.candidate_count = candidate_count
        self.judge = judge
        self.reward_scorer = reward_scorer
        self._interactive_controller = interactive_controller
        self.seed = seed

    async def process(self, sample: SampledExample) -> dict[str, Any] | None:
        instruction = sample.benchmark_data.text()
        constraints = [
            cast("Any", constraint).to_json() for constraint in sample.benchmark_data.constraints
        ]
        result = await self._run_model(self.model_runner, sample, instruction)
        if result is None:
            return None
        return {
            "prompt_source": sample.prompt_source,
            "prompt_index": sample.prompt_index,
            "prompt_id": sample.prompt_id,
            "prompt_document": sample.benchmark_data.prompt.document,
            "instruction": instruction,
            "prompt": sample.benchmark_data.meta_data.prompt,
            "model": result.model,
            "model_name": result.model_name,
            "attempt": result.attempt,
            "response": result.response,
            "reasoning_content": result.reasoning_content,
            "evaluation": result.evaluation,
            "constraint_count": sample.benchmark_data.meta_data.n_constraints,
            "constraint_types": sample.benchmark_data.meta_data.constraint_types,
            "constraint_groups": sample.benchmark_data.meta_data.constraint_groups,
            "constraint_instructions": sample.benchmark_data.meta_data.constraint_instructions,
            "constraints": constraints,
            "data_id": sample.benchmark_data.meta_data.data_id,
        }

    def _estimate_token_count(self, text: str) -> int:
        return self.reward_scorer.token_count(text)

    async def _rewrite_reasoning_content(
        self,
        *,
        instruction: str,
        response: str,
        reasoning_content: str,
    ) -> str:
        prompt = (
            "Rewrite the reasoning content to be concise and focused. "
            "Remove minor repetition and keep only the essential steps needed to reach the final answer. "
            "Return only the revised reasoning content.\n"
            f"Instruction:\n{instruction}\n"
            f"Final answer:\n{response}\n"
            f"Reasoning content:\n{reasoning_content}\n"
            "Rewritten reasoning content:"
        )
        responses, _ = await self.model_runner.client.async_ask([prompt])
        return (responses[0] if responses else "").strip()

    async def _review_reasoning_content(
        self,
        *,
        instruction: str,
        response: str,
        reasoning_content: str,
    ) -> tuple[str, dict[str, bool]] | None:
        evaluation = {
            "reasoning_not_major_repetition": True,
            "reasoning_not_rambling": True,
            "reasoning_token_limit_ok": True,
        }
        if not reasoning_content:
            return reasoning_content, evaluation

        try:
            review = await self.judge.review_reasoning(reasoning_content)
        except Exception as exc:
            logger.warning(f"Failed to review reasoning content: {exc}")
            return None
        evaluation["reasoning_not_major_repetition"] = review.repetition_level != "major"
        evaluation["reasoning_not_rambling"] = not review.is_rambling
        if review.repetition_level == "major":
            return None

        if review.repetition_level == "minor" or review.is_rambling:
            rewritten = await self._rewrite_reasoning_content(
                instruction=instruction,
                response=response,
                reasoning_content=reasoning_content,
            )
            if self._estimate_token_count(rewritten) > MAX_REASONING_CONTENT_TOKENS:
                evaluation["reasoning_token_limit_ok"] = False
                return None
            review = await self.judge.review_reasoning(rewritten)
            evaluation["reasoning_not_major_repetition"] = review.repetition_level != "major"
            evaluation["reasoning_not_rambling"] = not review.is_rambling
            if review.repetition_level == "major":
                return None
            reasoning_content = rewritten

        return reasoning_content, evaluation

    async def _score_candidates(
        self,
        instruction: str,
        candidates: list[GenerationResult],
    ) -> list[ScoredCandidate]:
        responses = [candidate.response for candidate in candidates]
        reasonings = [candidate.reasoning_content for candidate in candidates]
        response_scores = await self.reward_scorer.score_many(instruction, responses)
        reasoning_scores = await self.reward_scorer.score_many(instruction, reasonings)
        scored: list[ScoredCandidate] = []
        for candidate, response_score, reasoning_score in zip(
            candidates, response_scores, reasoning_scores, strict=True
        ):
            average = (response_score + reasoning_score) / 2
            scored.append(
                ScoredCandidate(
                    result=candidate,
                    response_reward=response_score,
                    reasoning_reward=reasoning_score,
                    average_reward=average,
                )
            )
        return scored

    async def _select_best_candidate(
        self,
        instruction: str,
        candidates: list[GenerationResult],
    ) -> GenerationResult | None:
        if not candidates:
            return None
        scored_candidates = await self._score_candidates(instruction, candidates)
        best = max(scored_candidates, key=lambda entry: entry.average_reward)
        return best.result

    async def _generate_candidate(
        self,
        runner: ModelRunner,
        sample: SampledExample,
        instruction: str,
    ) -> GenerationResult | None:
        benchmark_data = sample.benchmark_data
        for attempt in range(1, self.max_attempts + 1):
            try:
                response, reasoning_content = await runner.generate(instruction)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"Model {runner.spec.model} failed on attempt {attempt}: {exc}")
                continue
            if not reasoning_content.strip():
                logger.info(
                    "Regenerating candidate for %s attempt %d due to empty reasoning content.",
                    sample.prompt_id,
                    attempt,
                )
                continue
            reasoning_history = [reasoning_content] if reasoning_content else []
            if self._interactive_controller is not None:
                proceed = await self._interactive_controller.confirm_generation(
                    sample=sample,
                    spec=runner.spec,
                    attempt=attempt,
                    response=response,
                )
                if not proceed:
                    raise InteractiveSkip()
            if await self.judge.is_refusal(response):
                logger.info(
                    "Skipping candidate for %s attempt %d due to refusal response.",
                    sample.prompt_id,
                    attempt,
                )
                continue
            response_repetition = await self.judge.repetition_level(response, label="response")
            if response_repetition == "major":
                logger.info(
                    "Skipping candidate for %s attempt %d due to major repetition in response.",
                    sample.prompt_id,
                    attempt,
                )
                continue
            constraint_evaluation = await benchmark_data.evaluate(response)
            constraints_passed = all(constraint_evaluation.values())
            if constraints_passed:
                reviewed = await self._review_reasoning_content(
                    instruction=instruction,
                    response=response,
                    reasoning_content="".join(reasoning_history),
                )
                if reviewed is None:
                    logger.info(
                        "Skipping candidate for %s attempt %d due to invalid reasoning content.",
                        sample.prompt_id,
                        attempt,
                    )
                    continue
                reviewed_reasoning, reasoning_eval = reviewed
                evaluation = dict(constraint_evaluation)
                evaluation["constraints_passed"] = constraints_passed
                evaluation["response_not_refusal"] = True
                evaluation["response_not_major_repetition"] = True
                evaluation.update(reasoning_eval)
                if all(evaluation.values()):
                    return GenerationResult(
                        model=runner.spec.model,
                        model_name=runner.spec.name,
                        temperature=runner.spec.temperature,
                        attempt=attempt,
                        response=response,
                        reasoning_content=reviewed_reasoning,
                        evaluation=evaluation,
                    )
                logger.info(
                    "Skipping candidate for %s attempt %d due to failed evaluation checks.",
                    sample.prompt_id,
                    attempt,
                )
                continue
            client = getattr(runner, "client", None)
            if client is None:
                logger.info(
                    "Skipping rewrite for %s attempt %d because no client is available.",
                    sample.prompt_id,
                    attempt,
                )
                continue
            if self._interactive_controller is not None:
                should_rewrite = await self._interactive_controller.confirm_rewrite(
                    sample=sample,
                    spec=runner.spec,
                    attempt=attempt,
                    response=response,
                )
                if not should_rewrite:
                    logger.info(
                        "Skipping rewrite for %s attempt %d after interactive decline.",
                        sample.prompt_id,
                        attempt,
                    )
                    continue
            try:
                rewritten, rewritten_eval = await benchmark_data.rewrite(
                    response,
                    client,
                    reasoning_history=reasoning_history,
                )
            except Exception as exc:
                logger.warning(
                    f"Rewrite failed for model {runner.spec.model} on attempt {attempt}: {exc}"
                )
                continue
            if await self.judge.is_refusal(rewritten):
                logger.info(
                    "Skipping rewritten candidate for %s attempt %d due to refusal response.",
                    sample.prompt_id,
                    attempt,
                )
                continue
            response_repetition = await self.judge.repetition_level(rewritten, label="response")
            if response_repetition == "major":
                logger.info(
                    "Skipping rewritten candidate for %s attempt %d due to major repetition.",
                    sample.prompt_id,
                    attempt,
                )
                continue
            constraints_passed = all(rewritten_eval.values())
            if not constraints_passed:
                logger.info(
                    "Skipping rewritten candidate for %s attempt %d due to constraint failures.",
                    sample.prompt_id,
                    attempt,
                )
                continue
            reviewed = await self._review_reasoning_content(
                instruction=instruction,
                response=rewritten,
                reasoning_content="".join(reasoning_history),
            )
            if reviewed is None:
                logger.info(
                    "Skipping rewritten candidate for %s attempt %d due to invalid reasoning.",
                    sample.prompt_id,
                    attempt,
                )
                continue
            reviewed_reasoning, reasoning_eval = reviewed
            evaluation = dict(rewritten_eval)
            evaluation["constraints_passed"] = constraints_passed
            evaluation["response_not_refusal"] = True
            evaluation["response_not_major_repetition"] = True
            evaluation.update(reasoning_eval)
            if all(evaluation.values()):
                return GenerationResult(
                    model=runner.spec.model,
                    model_name=runner.spec.name,
                    temperature=runner.spec.temperature,
                    attempt=attempt,
                    response=rewritten,
                    reasoning_content=reviewed_reasoning,
                    evaluation=evaluation,
                )
            logger.info(
                "Skipping rewritten candidate for %s attempt %d due to failed evaluation checks.",
                sample.prompt_id,
                attempt,
            )
        return None

    async def _run_model(
        self,
        runner: ModelRunner,
        sample: SampledExample,
        instruction: str,
    ) -> GenerationResult | None:
        benchmark_data = sample.benchmark_data
        logger.info(
            f"Starting generation with model {runner.spec.model} for prompt ID {sample.prompt_id}."
        )
        logger.info(f"Constraints:\n{benchmark_data.meta_data.constraint_types}")
        candidates: list[GenerationResult] = []
        for _ in range(self.candidate_count):
            candidate = await self._generate_candidate(runner, sample, instruction)
            if candidate is not None:
                candidates.append(candidate)
        if not candidates:
            logger.warning("No valid candidates generated for prompt ID %s.", sample.prompt_id)
        return await self._select_best_candidate(instruction, candidates)


class DatasetWriter:
    def __init__(
        self,
        output_dir: Path,
        save_every: int,
        *,
        flush_each_add: bool = False,
    ) -> None:
        self.output_dir = output_dir
        self.save_every = max(1, save_every)
        self.flush_each_add = flush_each_add
        self.buffer: list[dict[str, Any]] = []
        self.dataset: Dataset | None = None
        if self.output_dir.exists():
            self.dataset = Dataset.load_from_disk(str(self.output_dir))

    def add(self, record: dict[str, Any]) -> None:
        self.buffer.append(record)
        if len(self.buffer) >= self.save_every or self.flush_each_add:
            self._flush()

    def flush(self) -> None:
        if self.buffer:
            self._flush()

    def _flush(self) -> None:
        records = list(self.buffer)
        chunk = Dataset.from_list(records)
        if self.dataset is None:
            combined = chunk
        else:
            combined = concatenate_datasets([self.dataset, chunk])
        self._write_dataset(combined)
        self.dataset = Dataset.load_from_disk(str(self.output_dir))
        self.buffer.clear()

    def existing_dataset(self) -> Dataset | None:
        if self.dataset is None and self.output_dir.exists():
            self.dataset = Dataset.load_from_disk(str(self.output_dir))
        return self.dataset

    def _write_dataset(self, dataset: Dataset) -> None:
        self.output_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.output_dir.parent) as tmpdir:
            tmp_path = Path(tmpdir) / "dataset"
            dataset.save_to_disk(str(tmp_path))
            backup_dir: Path | None = None
            try:
                if self.output_dir.exists():
                    backup_dir = self._build_backup_path()
                    self.output_dir.rename(backup_dir)
                shutil.move(str(tmp_path), str(self.output_dir))
            except Exception:
                if backup_dir is not None and backup_dir.exists():
                    if not self.output_dir.exists():
                        backup_dir.rename(self.output_dir)
                raise
            else:
                if backup_dir is not None and backup_dir.exists():
                    shutil.rmtree(backup_dir, ignore_errors=True)

    def _build_backup_path(self) -> Path:
        base = self.output_dir.with_name(f"{self.output_dir.name}.backup")
        if not base.exists():
            return base
        return self.output_dir.with_name(f"{self.output_dir.name}.backup-{uuid.uuid4().hex}")


def collect_existing_keys(dataset: Dataset | None) -> list[str]:
    if dataset is None:
        return []
    unique_keys: list[str] = []
    seen: set[str] = set()
    for row in dataset:
        try:
            key = CombinationTracker.key_from_mapping(row)
        except ValueError:
            continue
        if key in seen:
            continue
        seen.add(key)
        unique_keys.append(key)
    return unique_keys


class CombinationTracker:
    def __init__(self, existing_keys: Sequence[str] | None = None) -> None:
        self._final_keys: set[str] = set(existing_keys or [])
        self._pending_keys: set[str] = set()
        self._lock = asyncio.Lock()

    @staticmethod
    def _normalize_constraints(
        constraints: Any,
        constraint_instructions: Sequence[str],
    ) -> list[tuple[str, str]]:
        if not constraints:
            return []
        normalized: list[tuple[str, str]] = []
        for idx, item in enumerate(constraints):
            if not isinstance(item, dict):
                if not isinstance(item, str):
                    continue
                try:
                    payload = json.loads(item)
                except json.JSONDecodeError:
                    payload = {"name": item}
                if not isinstance(payload, dict):
                    continue
                name = str(payload.get("name", ""))
                instructions = (
                    constraint_instructions[idx] if idx < len(constraint_instructions) else ""
                )
                normalized.append((name, instructions))
                continue
            name = str(item.get("name", ""))
            instructions = str(item.get("instructions", ""))
            if not instructions and idx < len(constraint_instructions):
                instructions = constraint_instructions[idx]
            normalized.append((name, instructions))
        return normalized

    @staticmethod
    def _normalize_string_sequence(value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            normalized: list[str] = []
            for item in value:
                normalized.append(str(item))
            return normalized
        return [str(value)]

    @staticmethod
    def _signature_from_pairs(pairs: Sequence[tuple[str, str]]) -> str:
        if not pairs:
            raise ValueError("No constraints provided for deduplication.")
        parts = [f"{name}::{instructions}" for name, instructions in pairs]
        return "|".join(sorted(parts))

    @classmethod
    def _key_from_pairs(cls, prompt_id: str, pairs: Sequence[tuple[str, str]]) -> str:
        prompt_id = prompt_id.strip()
        if not prompt_id:
            raise ValueError("prompt_id is required for deduplication.")
        signature = cls._signature_from_pairs(pairs)
        return f"{prompt_id}||{signature}"

    @classmethod
    def key_from_sample(cls, sample: SampledExample) -> str:
        meta_data = sample.benchmark_data.meta_data
        pairs = list(
            zip(
                meta_data.constraint_types,
                meta_data.constraint_instructions,
                strict=True,
            )
        )
        return cls._key_from_pairs(sample.prompt_id, pairs)

    @classmethod
    def key_from_mapping(cls, mapping: Mapping[str, Any]) -> str:
        prompt_id = str(mapping.get("prompt_id", ""))
        constraint_types = cls._normalize_string_sequence(mapping.get("constraint_types"))
        constraint_instructions = cls._normalize_string_sequence(
            mapping.get("constraint_instructions")
        )
        if len(constraint_types) == len(constraint_instructions) and constraint_types:
            pairs = list(zip(constraint_types, constraint_instructions, strict=True))
            return cls._key_from_pairs(prompt_id, pairs)
        constraints = cls._normalize_constraints(
            mapping.get("constraints"), constraint_instructions
        )
        return cls._key_from_pairs(prompt_id, constraints)

    async def reserve(self, key: str) -> bool:
        async with self._lock:
            if key in self._final_keys or key in self._pending_keys:
                return False
            self._pending_keys.add(key)
            return True

    async def release(self, key: str) -> None:
        async with self._lock:
            self._pending_keys.discard(key)

    async def mark_final(self, key: str) -> None:
        async with self._lock:
            self._pending_keys.discard(key)
            self._final_keys.add(key)

    async def stats(self) -> tuple[int, int]:
        async with self._lock:
            return len(self._final_keys), len(self._pending_keys)


class ConstraintCountScheduler:
    def __init__(
        self,
        total: int,
        counts: Sequence[int],
        rng: np.random.Generator,
        *,
        favor_smaller_counts: bool = False,
    ) -> None:
        if total <= 0:
            raise ValueError("Total number of samples must be positive.")
        if not counts:
            raise ValueError("At least one constraint count is required.")
        self._rng = rng
        self.targets = self._build_targets(total, counts)
        plan: list[int] = []
        for count, target in self.targets.items():
            plan.extend([count] * target)
        if not favor_smaller_counts:
            plan = self._shuffle_plan(plan)
        self._pending: deque[int] = deque(plan)
        self._lock = asyncio.Lock()

    def _shuffle_plan(self, plan: list[int]) -> list[int]:
        if len(plan) <= 1:
            return plan
        indices = self._rng.permutation(len(plan))
        return [plan[idx] for idx in indices]

    @staticmethod
    def _build_targets(total: int, counts: Sequence[int]) -> dict[int, int]:
        unique_counts = sorted(set(counts))
        allocation = total // len(unique_counts)
        remainder = total % len(unique_counts)
        targets = {count: allocation for count in unique_counts}
        for idx in range(remainder):
            targets[unique_counts[idx]] += 1
        return targets

    async def next_count(self) -> int | None:
        async with self._lock:
            if not self._pending:
                return None
            return self._pending.popleft()

    async def requeue(self, count: int) -> None:
        async with self._lock:
            if self._rng.random() < 0.5:
                self._pending.append(count)
            else:
                self._pending.appendleft(count)


class FailureTracker:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self.count = 0
        self._lock = asyncio.Lock()

    async def increment(self) -> None:
        async with self._lock:
            self.count += 1
            if self.count > self.limit:
                raise RuntimeError("Exceeded maximum number of failed candidates.")


def _resolve_prompt_loader(name: str) -> Callable[[], list[Prompt]]:
    loader = PROMPT_LOADERS.get(name)
    if loader is None:
        raise ValueError(f"Unsupported prompt source '{name}'.")
    return loader


def _build_model_runner(spec: ModelSpec) -> ModelRunner:
    extra_body = dict(spec.extra_body)
    extra_body.setdefault("temperature", spec.temperature)
    return ModelRunner(
        spec=spec,
        client=LLMClient(provider=spec.provider, model=spec.model, extra_body=extra_body),
    )


async def _run_interactive_generation(
    *,
    sampler: ConstraintSampler,
    scheduler: ConstraintCountScheduler,
    combination_tracker: CombinationTracker,
    processor: CandidateProcessor,
    interactive_controller: InteractiveController,
    writer: DatasetWriter,
    writer_lock: asyncio.Lock,
    failure_tracker: FailureTracker,
    progress: tqdm,
) -> None:
    while True:
        finalized, pending = await combination_tracker.stats()
        count = await scheduler.next_count()
        if count is None:
            break
        sample = sampler.sample(count)
        try:
            sample_key = CombinationTracker.key_from_sample(sample)
        except ValueError:
            await scheduler.requeue(count)
            continue
        reserved = await combination_tracker.reserve(sample_key)
        if not reserved:
            await scheduler.requeue(count)
            continue
        proceed_with_generation = await interactive_controller.confirm_sample(sample)
        if not proceed_with_generation:
            logger.info(f"Sample {sample.prompt_id} skipped before generation.")
            await combination_tracker.release(sample_key)
            await scheduler.requeue(count)
            continue
        try:
            record = await processor.process(sample)
        except InteractiveSkip:
            logger.info(f"Sample {sample.prompt_id} skipped before evaluation.")
            await combination_tracker.release(sample_key)
            await scheduler.requeue(count)
            continue
        if record is None:
            await combination_tracker.release(sample_key)
            await scheduler.requeue(count)
            await failure_tracker.increment()
            continue
        proceed_with_save = await interactive_controller.confirm_save(record)
        if not proceed_with_save:
            logger.info(f"Sample {sample.prompt_id} skipped before saving.")
            await combination_tracker.release(sample_key)
            await scheduler.requeue(count)
            continue
        async with writer_lock:
            writer.add(record)
        await combination_tracker.mark_final(sample_key)
        progress.update(1)


async def _run_parallel_generation(
    *,
    sampler: ConstraintSampler,
    scheduler: ConstraintCountScheduler,
    combination_tracker: CombinationTracker,
    processor: CandidateProcessor,
    writer: DatasetWriter,
    writer_lock: asyncio.Lock,
    failure_tracker: FailureTracker,
    progress: tqdm,
    max_workers: int,
) -> None:
    tracker_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max_workers)
    pending: set[asyncio.Task[None]] = set()

    async def process_count(count: int) -> None:
        async with semaphore:
            sample = sampler.sample(count)
            try:
                sample_key = CombinationTracker.key_from_sample(sample)
            except ValueError:
                await scheduler.requeue(count)
                return
            async with tracker_lock:
                reserved = await combination_tracker.reserve(sample_key)
            if not reserved:
                await scheduler.requeue(count)
                return
            record = await processor.process(sample)
            if record is None:
                await combination_tracker.release(sample_key)
                await scheduler.requeue(count)
                await failure_tracker.increment()
                return
            async with writer_lock:
                writer.add(record)
            await combination_tracker.mark_final(sample_key)
            progress.update(1)

    async def drain_completed(*, block: bool) -> None:
        if not pending:
            return
        if block:
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        else:
            done = {task for task in pending if task.done()}
            if not done:
                return
        for task in done:
            pending.remove(task)
            task.result()

    while True:
        count = await scheduler.next_count()
        if count is None:
            if not pending:
                break
            await drain_completed(block=True)
            continue
        task = asyncio.create_task(process_count(count))
        pending.add(task)
        await drain_completed(block=False)

    while pending:
        await drain_completed(block=True)


async def _generate_async(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    loader = _resolve_prompt_loader(args.prompt_source)
    prompts = loader()
    judge_client = LLMClient(
        provider="vllm",
        model=REWARD_MODEL_DEFAULT,
        extra_body={"base_url": REWARD_BASE_URL_DEFAULT, "temperature": 0.0, "max_tokens": 65536},
    )
    judge = LLMJudge(judge_client)
    inventory = ConstraintInventory(constraint_set=args.constraint_set)
    sampler = ConstraintSampler(
        prompt_source=args.prompt_source,
        prompts=prompts,
        inventory=inventory,
        judge_client=judge_client,
        rng=rng,
    )
    spec: ModelSpec = args.model_spec
    model_runner = _build_model_runner(spec)
    reward_scorer = RewardScorer(
        base_url=args.reward_base_url,
        api_key=args.reward_api_key,
        model=args.reward_model,
        token_limit=args.reward_token_limit,
        concurrency=args.reward_concurrency,
    )
    writer = DatasetWriter(
        Path(args.output_dir),
        save_every=args.save_every,
        flush_each_add=args.interactive_mode,
    )
    existing_dataset = writer.existing_dataset()
    existing_keys = collect_existing_keys(existing_dataset)
    logger.info("Existing dataset contains %d unique records.", len(existing_keys))
    combination_tracker = CombinationTracker(existing_keys)
    initial_completed = len(existing_keys)
    remaining_target = max(args.n_data - initial_completed, 0)
    failure_tracker = FailureTracker(limit=args.max_failures)
    progress = tqdm(
        total=args.n_data,
        desc="Building dataset",
        unit="sample",
        initial=min(initial_completed, args.n_data),
    )
    if remaining_target == 0:
        progress.close()
        return
    scheduler = ConstraintCountScheduler(
        remaining_target,
        args.constraint_counts,
        rng,
        favor_smaller_counts=args.interactive_mode,
    )
    writer_lock = asyncio.Lock()
    if args.interactive_mode:
        interactive_controller = InteractiveController(enabled=True)
        processor = CandidateProcessor(
            model_runner,
            max_attempts=args.max_attempts,
            judge=judge,
            reward_scorer=reward_scorer,
            interactive_controller=interactive_controller,
            seed=args.seed,
        )
        await _run_interactive_generation(
            sampler=sampler,
            scheduler=scheduler,
            combination_tracker=combination_tracker,
            processor=processor,
            interactive_controller=interactive_controller,
            writer=writer,
            writer_lock=writer_lock,
            failure_tracker=failure_tracker,
            progress=progress,
        )
    else:
        processor = CandidateProcessor(
            model_runner,
            max_attempts=args.max_attempts,
            judge=judge,
            reward_scorer=reward_scorer,
            seed=args.seed,
        )
        await _run_parallel_generation(
            sampler=sampler,
            scheduler=scheduler,
            combination_tracker=combination_tracker,
            processor=processor,
            writer=writer,
            writer_lock=writer_lock,
            failure_tracker=failure_tracker,
            progress=progress,
            max_workers=args.max_workers,
        )
    async with writer_lock:
        writer.flush()
    progress.close()


def _parse_mode_spec_json(value: str) -> ModelSpec:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:  # pragma: no cover - simple error path
        raise ValueError("mode_spec_json must be a valid JSON object.") from exc
    if not isinstance(payload, dict):
        raise ValueError("mode_spec_json must describe a single object.")
    try:
        spec = ModelSpec(**payload)
    except TypeError as exc:
        raise ValueError("mode_spec_json must include model name and identifier.") from exc
    if not isinstance(spec.extra_body, dict):
        raise ValueError("mode_spec_json.extra_body must be an object when provided.")
    allowed_providers = {"openrouter", "vllm"}
    if spec.provider not in allowed_providers:
        raise ValueError(
            f"Unsupported provider '{spec.provider}'. Choose from openrouter or vllm."
        )
    return spec


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate HuggingFace datasets for jfbench prompts."
    )
    parser.add_argument(
        "--n-data", type=int, required=True, help="Total number of successful samples to collect."
    )
    parser.add_argument(
        "--prompt-source",
        type=str,
        default="ja_stackoverflow",
        choices=sorted(PROMPT_LOADERS.keys()),
        help="Prompt source module under jfbench.prompts.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/generated_dataset",
        help="Directory where the HuggingFace dataset will be stored.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20250101,
        help="Random seed used for sampling prompts and constraints.",
    )
    parser.add_argument(
        "--constraint-counts",
        type=str,
        default="1,2,4,8",
        help="Comma-separated list of constraint counts to include.",
    )
    parser.add_argument(
        "--constraint-set",
        type=str,
        default="train",
        choices=("train", "test"),
        help="Constraint set to use when sampling constraints.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum number of attempts per model for each sample.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=100,
        help="Number of concurrent sampling workers.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1000,
        help="Number of successful samples collected before forcing a save.",
    )
    parser.add_argument(
        "--mode-spec-json",
        type=str,
        required=True,
        help=(
            "JSON string describing the generation model. "
            "Must include the model name and identifier."
        ),
    )
    parser.add_argument(
        "--reward-base-url",
        type=str,
        default=REWARD_BASE_URL_DEFAULT,
        help="Base URL for the reward model endpoint.",
    )
    parser.add_argument(
        "--reward-api-key",
        type=str,
        default=REWARD_API_KEY_DEFAULT,
        help="API key used for the reward model.",
    )
    parser.add_argument(
        "--reward-model",
        type=str,
        default=REWARD_MODEL_DEFAULT,
        help="Reward model name or path.",
    )
    parser.add_argument(
        "--reward-token-limit",
        type=int,
        default=REWARD_TOKEN_LIMIT_DEFAULT,
        help="Maximum number of candidate tokens considered for reward scoring.",
    )
    parser.add_argument(
        "--reward-concurrency",
        type=int,
        default=REWARD_CONCURRENCY_DEFAULT,
        help="Maximum concurrent reward scoring requests.",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=1000,
        help="Maximum number of failed candidate generations before aborting.",
    )
    parser.add_argument(
        "--interactive-mode",
        action="store_true",
        help=("Run sequentially and ask for confirmation before generation and evaluation steps."),
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="WARNING",
        help="Logging level passed to logging.basicConfig (e.g., INFO, DEBUG).",
    )
    args = parser.parse_args(argv)
    args.constraint_counts = tuple(
        int(value.strip()) for value in args.constraint_counts.split(",") if value.strip()
    )
    if not args.constraint_counts:
        raise ValueError("At least one constraint count must be provided.")
    args.log_level = str(args.log_level).upper()
    level_value = getattr(logging, args.log_level, None)
    if not isinstance(level_value, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    if args.reward_token_limit <= 0:
        raise ValueError("reward_token_limit must be positive.")
    if args.reward_concurrency <= 0:
        raise ValueError("reward_concurrency must be positive.")
    args.model_spec = _parse_mode_spec_json(args.mode_spec_json)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    log_level_value = getattr(logging, args.log_level, logging.WARNING)
    logging.basicConfig(
        level=log_level_value,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        # level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(_generate_async(args))


__all__ = [
    "MAX_REASONING_CONTENT_TOKENS",
    "CandidateProcessor",
    "CombinationTracker",
    "ConstraintInventory",
    "ConstraintSampler",
    "DatasetWriter",
    "FailureTracker",
    "InteractiveController",
    "InteractiveSkip",
    "LLMJudge",
    "ModelRunner",
    "ModelSpec",
    "ReasoningReview",
    "RewardScorer",
    "SampledExample",
    "ScoredCandidate",
    "collect_existing_keys",
    "main",
    "parse_args",
]

if __name__ == "__main__":
    main()
