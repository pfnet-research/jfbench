from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Literal
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import TYPE_CHECKING

import numpy as np
from openai import APIConnectionError
from openai import APITimeoutError
from openai import AsyncOpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

from jfbench.llm import extract_reasoning_content
from jfbench.llm import LLMClient
from jfbench.sft_dataset.generate import collect_existing_keys
from jfbench.sft_dataset.generate import CombinationTracker
from jfbench.sft_dataset.generate import ConstraintCountScheduler
from jfbench.sft_dataset.generate import ConstraintInventory
from jfbench.sft_dataset.generate import ConstraintSampler
from jfbench.sft_dataset.generate import DatasetWriter
from jfbench.sft_dataset.generate import FailureTracker
from jfbench.sft_dataset.generate import JudgeProtocol
from jfbench.sft_dataset.generate import LLMJudge
from jfbench.sft_dataset.generate import MAX_REASONING_CONTENT_TOKENS
from jfbench.sft_dataset.generate import PROMPT_LOADERS
from jfbench.sft_dataset.generate import SampledExample


if TYPE_CHECKING:
    from jfbench.benchmark.build import BenchmarkData


logger = logging.getLogger(__name__)


@dataclass
class ResponseSummary:
    response: str
    evaluation: dict[str, bool]
    origin: str
    attempts: int
    reward: float | None = None
    reasoning_content: str = ""


@dataclass
class ModelSpec:
    provider: Literal["openrouter", "vllm"]
    model: str
    name: str
    extra_body: dict[str, Any] | None = None


DEFAULT_REWARD_BASE_URL = "http://localhost:8000/v1"
DEFAULT_REWARD_API_KEY = "unused"
DEFAULT_REWARD_MODEL = "openai/gpt-oss-120b"
DEFAULT_REWARD_TOKENS = 512
DEFAULT_REWARD_CONCURRENCY = 40
TIMEOUT = 120000


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
    tokenizer: AutoTokenizer,
    text: str,
    *,
    chunk_size: int,
    overlap: int,
) -> list[list[int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size.")
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks: list[list[int]] = []
    i = 0
    n = len(ids)
    while i < n:
        chunks.append(ids[i : min(i + chunk_size, n)])
        if i + chunk_size >= n:
            break
        i += chunk_size - overlap
    return chunks


async def _completions_prompt_logprobs_chunked(
    client: AsyncOpenAI,
    *,
    model: str,
    text: str,
    tokenizer: AutoTokenizer,
    chunk_size: int,
    overlap: int,
) -> dict[str, Any]:
    chunks = _chunk_by_tokens(tokenizer, text, chunk_size=chunk_size, overlap=overlap)
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
            except APITimeoutError as e:
                logger.warning(f"Timeout error during logprob request, retrying: {e}")
                continue
            except APIConnectionError as e:
                logger.warning(f"Connection error during logprob request, retrying: {e}")

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


class RewardScorer:
    def __init__(
        self,
        *,
        base_url: str = DEFAULT_REWARD_BASE_URL,
        api_key: str = DEFAULT_REWARD_API_KEY,
        model: str = DEFAULT_REWARD_MODEL,
        token_limit: int = DEFAULT_REWARD_TOKENS,
        concurrency: int = DEFAULT_REWARD_CONCURRENCY,
    ) -> None:
        if token_limit <= 0:
            raise ValueError("reward_token_limit must be positive.")
        if concurrency <= 0:
            raise ValueError("reward_concurrency must be positive.")
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=TIMEOUT)
        self.model = model
        self.token_limit = token_limit
        self._semaphore = asyncio.Semaphore(concurrency)
        self._tokenizer: AutoTokenizer | None = None

    def token_count(self, text: str) -> int:
        if text == "":
            return 0
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model, use_fast=True)
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    async def pick(
        self,
        prompt: str,
        chosen_candidates: list[str],
        rejected_candidates: list[str],
    ) -> dict[str, Any]:
        chosen_scores = (
            await self._score_many(prompt, chosen_candidates) if chosen_candidates else []
        )
        rejected_scores = (
            await self._score_many(prompt, rejected_candidates) if rejected_candidates else []
        )
        chosen_idx = (
            max(range(len(chosen_scores)), key=lambda i: chosen_scores[i])
            if chosen_scores
            else None
        )
        rejected_idx = (
            min(range(len(rejected_scores)), key=lambda i: rejected_scores[i])
            if rejected_scores
            else None
        )
        return {
            "chosen_idx": chosen_idx,
            "chosen_text": None if chosen_idx is None else chosen_candidates[chosen_idx],
            "chosen_score": None if chosen_idx is None else chosen_scores[chosen_idx],
            "chosen_scores_all": chosen_scores,
            "rejected_idx": rejected_idx,
            "rejected_text": None if rejected_idx is None else rejected_candidates[rejected_idx],
            "rejected_score": None if rejected_idx is None else rejected_scores[rejected_idx],
            "rejected_scores_all": rejected_scores,
            "token_limit": self.token_limit,
        }

    async def _score_many(self, prompt: str, answers: list[str]) -> list[float]:
        async def _run(answer: str) -> float:
            async with self._semaphore:
                score, _ = await _score_one_with_echo(
                    self.client,
                    self.model,
                    prompt,
                    answer,
                    token_limit=self.token_limit,
                )
                return score

        return await asyncio.gather(*[_run(a) for a in answers])


def _build_scoring_prompt(prompt: str, answer: str) -> tuple[str, int]:
    prefix = f"User:\n{prompt}\nAssistant:\n"
    full_text = prefix + answer
    start_offset = len(prefix)
    return full_text, start_offset


async def _score_one_with_echo(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    answer: str,
    *,
    token_limit: int,
) -> tuple[float, int]:
    text, _ = _build_scoring_prompt(prompt, answer)
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    prefix_tokens = tokenizer.encode(f"User:\n{prompt}\nAssistant:\n", add_special_tokens=False)
    candidate_start_idx = len(prefix_tokens)
    logprob_result = await _completions_prompt_logprobs_chunked(
        client,
        model=model,
        text=text,
        tokenizer=tokenizer,
        chunk_size=1024,
        overlap=32,
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


class DPORecordBuilder:
    def __init__(
        self,
        client: LLMClient,
        judge: JudgeProtocol,
        *,
        model: str,
        model_name: str,
        temperature: float,
        n_chains: int = 5,
        n_iterations: int | None = None,
        reward_model: str = DEFAULT_REWARD_MODEL,
        reward_base_url: str = DEFAULT_REWARD_BASE_URL,
        reward_api_key: str = DEFAULT_REWARD_API_KEY,
        reward_token_limit: int = DEFAULT_REWARD_TOKENS,
        reward_concurrency: int = DEFAULT_REWARD_CONCURRENCY,
    ) -> None:
        if n_chains <= 0:
            raise ValueError("n_chains must be positive.")
        if n_iterations is not None and n_iterations <= 0:
            raise ValueError("n_iterations must be positive when provided.")
        if reward_token_limit <= 0:
            raise ValueError("reward_token_limit must be positive.")
        if reward_concurrency <= 0:
            raise ValueError("reward_concurrency must be positive.")
        self.client = client
        self.judge = judge
        self.model = model
        self.model_name = model_name
        self.temperature = temperature
        self.n_chains = n_chains
        self.n_iterations = n_iterations
        self.reward_scorer = RewardScorer(
            base_url=reward_base_url,
            api_key=reward_api_key,
            model=reward_model,
            token_limit=reward_token_limit,
            concurrency=reward_concurrency,
        )

    async def build_record(self, sample: SampledExample) -> dict[str, Any] | None:
        instruction = sample.benchmark_data.text()
        instruction_mode = sample.benchmark_data.meta_data.constraint_set
        constraints_info = [
            {
                "name": constraint.__class__.__name__,
                "group": constraint.group,
                "instructions": constraint.instructions(train_or_test=instruction_mode),
            }
            for constraint in sample.benchmark_data.constraints
        ]
        pair = await self._generate_pair(sample, instruction)
        if pair is None:
            return None
        chosen_reasoning = str(pair["chosen"].get("reasoning_content", "") or "")
        rejected_reasoning = str(pair["rejected"].get("reasoning_content", "") or "")
        return {
            "prompt_source": sample.prompt_source,
            "prompt_index": sample.prompt_index,
            "prompt_id": sample.prompt_id,
            "prompt_document": sample.benchmark_data.prompt.document,
            "instruction": instruction,
            "prompt": sample.benchmark_data.meta_data.prompt,
            "constraint_count": sample.benchmark_data.meta_data.n_constraints,
            "constraint_types": sample.benchmark_data.meta_data.constraint_types,
            "constraint_groups": sample.benchmark_data.meta_data.constraint_groups,
            "constraint_instructions": sample.benchmark_data.meta_data.constraint_instructions,
            "constraints": constraints_info,
            "data_id": sample.benchmark_data.meta_data.data_id,
            "model": self.model,
            "model_name": self.model_name,
            "temperature": self.temperature,
            "chosen": pair["chosen"],
            "rejected": pair["rejected"],
            "chosen_reasoning": chosen_reasoning,
            "rejected_reasoning": rejected_reasoning,
        }

    async def _generate_pair(
        self,
        sample: SampledExample,
        instruction: str,
    ) -> dict[str, dict[str, Any]] | None:
        chosen_candidates, rejected_candidates = await self._collect_candidates(
            sample=sample,
            instruction=instruction,
        )
        if not chosen_candidates or not rejected_candidates:
            logger.warning(
                "Insufficient candidates for %s (chosen: %d, rejected: %d)",
                sample.prompt_id,
                len(chosen_candidates),
                len(rejected_candidates),
            )
            return None

        selection = await self._score_candidates(
            prompt=instruction,
            chosen_candidates=chosen_candidates,
            rejected_candidates=rejected_candidates,
        )
        if selection is None:
            logger.warning("Failed to select chosen/rejected for %s", sample.prompt_id)
            return None
        return selection

    async def _collect_candidates(
        self,
        *,
        sample: SampledExample,
        instruction: str,
    ) -> tuple[list[ResponseSummary], list[ResponseSummary]]:
        benchmark = sample.benchmark_data
        chosen_candidates: list[ResponseSummary] = []
        rejected_candidates: list[ResponseSummary] = []
        iteration_limit = self._resolve_iteration_limit(sample)

        for chain_idx in range(self.n_chains):
            try:
                initial_response, reasoning_content = await self._generate_once(instruction)
            except Exception as exc:
                logger.warning(
                    "Failed to generate initial response for %s (chain %d): %s",
                    sample.prompt_id,
                    chain_idx,
                    exc,
                )
                continue
            reviewed = await self._review_candidate(
                instruction=instruction,
                response=initial_response,
                reasoning_content=reasoning_content,
                prompt_id=sample.prompt_id,
                attempt=1,
                origin="generation",
            )
            if reviewed is None:
                continue
            reviewed_reasoning, review_eval = reviewed
            evaluations, summary = await benchmark._evaluate_constraints_with_details(  # noqa: SLF001
                initial_response
            )
            evaluation = dict(summary)
            evaluation["constraints_passed"] = _all_constraints_passed(summary)
            evaluation.update(review_eval)
            if _all_constraints_passed(summary):
                chosen_candidates.append(
                    ResponseSummary(
                        response=initial_response,
                        evaluation=evaluation,
                        origin="generation",
                        attempts=1,
                        reasoning_content=reviewed_reasoning,
                    )
                )
                await self._collect_after_success(
                    benchmark=benchmark,
                    instruction=instruction,
                    iteration_limit=iteration_limit,
                    chosen_candidates=chosen_candidates,
                    rejected_candidates=rejected_candidates,
                )
            else:
                rejected_candidates.append(
                    ResponseSummary(
                        response=initial_response,
                        evaluation=evaluation,
                        origin="generation",
                        attempts=1,
                        reasoning_content=reviewed_reasoning,
                    )
                )
                await self._collect_after_failure(
                    benchmark=benchmark,
                    instruction=instruction,
                    iteration_limit=iteration_limit,
                    chosen_candidates=chosen_candidates,
                    rejected_candidates=rejected_candidates,
                    evaluations=evaluations,
                    current_value=initial_response,
                    sample=sample,
                    reasoning_history=[reviewed_reasoning],
                )
        return chosen_candidates, rejected_candidates

    async def _score_candidates(
        self,
        *,
        prompt: str,
        chosen_candidates: list[ResponseSummary],
        rejected_candidates: list[ResponseSummary],
    ) -> dict[str, dict[str, Any]] | None:
        result = await self.reward_scorer.pick(
            prompt,
            [c.response for c in chosen_candidates],
            [c.response for c in rejected_candidates],
        )
        chosen_idx = result.get("chosen_idx")
        rejected_idx = result.get("rejected_idx")
        if chosen_idx is None or rejected_idx is None:
            return None
        chosen = chosen_candidates[chosen_idx]
        rejected = rejected_candidates[rejected_idx]
        chosen.reward = result.get("chosen_score")
        rejected.reward = result.get("rejected_score")
        return {"chosen": vars(chosen), "rejected": vars(rejected)}

    def _resolve_iteration_limit(self, sample: SampledExample) -> int:
        if self.n_iterations is not None:
            return self.n_iterations
        return sample.benchmark_data.meta_data.n_constraints + 1

    async def _collect_after_success(
        self,
        *,
        benchmark: BenchmarkData,
        instruction: str,
        iteration_limit: int,
        chosen_candidates: list[ResponseSummary],
        rejected_candidates: list[ResponseSummary],
    ) -> None:
        for iteration in range(1, iteration_limit + 1):
            try:
                response, reasoning_content = await self._generate_once(instruction)
            except Exception as exc:
                logger.warning("Failed to generate follow-up response: %s", exc)
                continue
            reviewed = await self._review_candidate(
                instruction=instruction,
                response=response,
                reasoning_content=reasoning_content,
                prompt_id=benchmark.meta_data.data_id,
                attempt=iteration + 1,
                origin="generation_after_success",
            )
            if reviewed is None:
                continue
            reviewed_reasoning, review_eval = reviewed
            evaluation = await benchmark.evaluate(response)
            evaluation["constraints_passed"] = _all_constraints_passed(evaluation)
            evaluation.update(review_eval)
            attempts = iteration + 1
            summary = ResponseSummary(
                response=response,
                evaluation=evaluation,
                origin="generation_after_success",
                attempts=attempts,
                reasoning_content=reviewed_reasoning,
            )
            if _all_constraints_passed(evaluation):
                chosen_candidates.append(summary)
            else:
                rejected_candidates.append(summary)

    async def _collect_after_failure(
        self,
        *,
        benchmark: BenchmarkData,
        instruction: str,
        iteration_limit: int,
        chosen_candidates: list[ResponseSummary],
        rejected_candidates: list[ResponseSummary],
        evaluations: list[tuple[bool, str | None]],
        current_value: str,
        sample: SampledExample,
        reasoning_history: list[str],
    ) -> None:
        reasoning_chain = list(reasoning_history)
        for iteration in range(1, iteration_limit + 1):
            failure_reasons = {
                constraint: reason
                for constraint, (passed, reason) in zip(
                    benchmark.constraints, evaluations, strict=True
                )
                if not passed
            }
            try:
                rewritten = await benchmark._rewrite_once(  # noqa: SLF001
                    current_value,
                    benchmark.constraints,
                    self.client,
                    failure_reasons,
                    reasoning_history=reasoning_chain,
                )
            except Exception as exc:
                logger.warning(
                    "Rewrite attempt %d failed for %s: %s", iteration, sample.prompt_id, exc
                )
                break

            evaluations, summary = await benchmark._evaluate_constraints_with_details(  # noqa: SLF001
                rewritten
            )
            attempts = iteration + 1
            combined_reasoning = "".join(reasoning_chain)
            reviewed = await self._review_candidate(
                instruction=instruction,
                response=rewritten,
                reasoning_content=combined_reasoning,
                prompt_id=sample.prompt_id,
                attempt=attempts,
                origin="rewrite",
            )
            evaluation = dict(summary)
            evaluation["constraints_passed"] = _all_constraints_passed(summary)
            if reviewed is None:
                current_value = rewritten
                continue
            reviewed_reasoning, review_eval = reviewed
            evaluation.update(review_eval)
            reasoning_chain = [reviewed_reasoning]
            if _all_constraints_passed(summary):
                chosen_candidates.append(
                    ResponseSummary(
                        response=rewritten,
                        evaluation=evaluation,
                        origin="rewrite",
                        attempts=attempts,
                        reasoning_content=reviewed_reasoning,
                    )
                )
                return
            rejected_candidates.append(
                ResponseSummary(
                    response=rewritten,
                    evaluation=evaluation,
                    origin="rewrite",
                    attempts=attempts,
                    reasoning_content=reviewed_reasoning,
                )
            )
            current_value = rewritten

    async def _generate_once(self, instruction: str) -> tuple[str, str]:
        responses, response_details = await self.client.async_ask([instruction])
        if not responses:
            raise RuntimeError(f"No response received from client. Instruction: {instruction}")
        reasoning_content = self._extract_reasoning_content(
            response_details[0] if response_details else None
        )
        if not reasoning_content:
            raise RuntimeError(
                f"Empty reasoning content received from client. Instruction: {instruction}. Response: {responses[0].strip()}"
            )
        return responses[0].strip(), reasoning_content

    def _extract_reasoning_content(self, response_detail: Any) -> str:
        return extract_reasoning_content(self.client.provider, response_detail)

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
        responses, _ = await self.client.async_ask([prompt])
        return (responses[0] if responses else "").strip()

    async def _review_reasoning_content(
        self,
        *,
        instruction: str,
        response: str,
        reasoning_content: str,
        prompt_id: str,
        attempt: int,
        origin: str,
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
            logger.info(
                "Skipping candidate for %s attempt %d (%s) due to major reasoning repetition.",
                prompt_id,
                attempt,
                origin,
            )
            return None

        if review.repetition_level == "minor" or review.is_rambling:
            rewritten = await self._rewrite_reasoning_content(
                instruction=instruction,
                response=response,
                reasoning_content=reasoning_content,
            )
            if self._estimate_token_count(rewritten) > MAX_REASONING_CONTENT_TOKENS:
                evaluation["reasoning_token_limit_ok"] = False
                logger.info(
                    "Skipping candidate for %s attempt %d (%s) due to reasoning length.",
                    prompt_id,
                    attempt,
                    origin,
                )
                return None
            review = await self.judge.review_reasoning(rewritten)
            evaluation["reasoning_not_major_repetition"] = review.repetition_level != "major"
            evaluation["reasoning_not_rambling"] = not review.is_rambling
            if review.repetition_level == "major":
                logger.info(
                    "Skipping candidate for %s attempt %d (%s) after reasoning rewrite.",
                    prompt_id,
                    attempt,
                    origin,
                )
                return None
            reasoning_content = rewritten

        return reasoning_content, evaluation

    async def _review_candidate(
        self,
        *,
        instruction: str,
        response: str,
        reasoning_content: str,
        prompt_id: str,
        attempt: int,
        origin: str,
    ) -> tuple[str, dict[str, bool]] | None:
        if await self.judge.is_refusal(response):
            logger.info(
                "Skipping candidate for %s attempt %d (%s) due to refusal response.",
                prompt_id,
                attempt,
                origin,
            )
            return None
        response_repetition = await self.judge.repetition_level(response, label="response")
        if response_repetition == "major":
            logger.info(
                "Skipping candidate for %s attempt %d (%s) due to major repetition in response.",
                prompt_id,
                attempt,
                origin,
            )
            return None
        reviewed = await self._review_reasoning_content(
            instruction=instruction,
            response=response,
            reasoning_content=reasoning_content,
            prompt_id=prompt_id,
            attempt=attempt,
            origin=origin,
        )
        if reviewed is None:
            return None
        reviewed_reasoning, reasoning_eval = reviewed
        evaluation = {
            "response_not_refusal": True,
            "response_not_major_repetition": True,
        }
        evaluation.update(reasoning_eval)
        if not all(evaluation.values()):
            logger.info(
                "Skipping candidate for %s attempt %d (%s) due to failed review checks.",
                prompt_id,
                attempt,
                origin,
            )
            return None
        return reviewed_reasoning, evaluation


def _all_constraints_passed(evaluation: Mapping[str, bool]) -> bool:
    return all(evaluation.values())


def _resolve_prompt_loader(name: str) -> Callable[[], list[Any]]:
    loader = PROMPT_LOADERS.get(name)
    if loader is None:
        raise ValueError(f"Unsupported prompt source '{name}'.")
    return loader


async def _run_parallel_generation(
    *,
    sampler: ConstraintSampler,
    scheduler: ConstraintCountScheduler,
    combination_tracker: CombinationTracker,
    builder: DPORecordBuilder,
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
            record = await builder.build_record(sample)
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
    print("Instantiating judge LLM client...")
    judge_client = LLMClient(
        provider="vllm",
        model=DEFAULT_REWARD_MODEL,
        extra_body={
            "base_url": DEFAULT_REWARD_BASE_URL,
            "temperature": 0.0,
            "max_tokens": 65536,
            "timeout": TIMEOUT,
        },
    )
    print(f"Instantiated judge LLM client. {judge_client.provider} - {judge_client.model}")
    judge = LLMJudge(judge_client)
    inventory = ConstraintInventory()
    sampler = ConstraintSampler(
        prompt_source=args.prompt_source,
        prompts=prompts,
        inventory=inventory,
        judge_client=judge_client,
        rng=rng,
    )
    writer = DatasetWriter(Path(args.output_dir), save_every=args.save_every)
    existing_dataset = writer.existing_dataset()
    existing_keys = collect_existing_keys(existing_dataset)
    print(f"Loaded existing dataset with {len(existing_keys)} unique records.")
    logger.info("Existing DPO dataset contains %d unique records.", len(existing_keys))
    combination_tracker = CombinationTracker(existing_keys)
    initial_completed = len(existing_keys)
    remaining_target = max(args.n_data - initial_completed, 0)
    failure_tracker = FailureTracker(limit=args.max_failures)
    print(
        f"Generation will stop after {remaining_target} new records or {args.max_failures} failures."
    )
    progress = tqdm(
        total=args.n_data,
        desc="Building DPO dataset",
        unit="pair",
        initial=min(initial_completed, args.n_data),
    )
    print(f"Starting generation to collect {remaining_target} new records...")
    if remaining_target == 0:
        progress.close()
        return
    scheduler = ConstraintCountScheduler(
        remaining_target,
        args.constraint_counts,
        rng,
        favor_smaller_counts=True,
    )
    spec: ModelSpec = args.model_spec
    print("Instantiating generation LLM client...")
    client = LLMClient(
        provider=spec.provider,
        model=spec.model,
        extra_body=spec.extra_body,
    )
    print(f"Instantiated generation LLM client. {client.provider} - {client.model}")
    builder = DPORecordBuilder(
        client=client,
        judge=judge,
        model=spec.model,
        model_name=spec.name,
        temperature=client.temperature,
        n_chains=args.n_chains,
        n_iterations=args.n_iterations,
        reward_model=args.reward_model,
        reward_base_url=args.reward_base_url,
        reward_api_key=args.reward_api_key,
        reward_token_limit=args.reward_token_limit,
        reward_concurrency=args.reward_concurrency,
    )
    writer_lock = asyncio.Lock()
    await _run_parallel_generation(
        sampler=sampler,
        scheduler=scheduler,
        combination_tracker=combination_tracker,
        builder=builder,
        writer=writer,
        writer_lock=writer_lock,
        failure_tracker=failure_tracker,
        progress=progress,
        max_workers=args.max_workers,
    )
    async with writer_lock:
        writer.flush()
    progress.close()


def _parse_model_spec_json(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("model_spec_json must be a valid JSON object.") from exc
    if not isinstance(parsed, dict):
        raise ValueError("model_spec_json must describe a single object.")
    if "model_short" in parsed and "name" not in parsed:
        parsed["name"] = parsed["model_short"]
        parsed.pop("model_short", None)
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DPO dataset from jfbench prompts as a HuggingFace dataset."
    )
    parser.add_argument("--n-data", type=int, required=True, help="Number of preference pairs.")
    parser.add_argument(
        "--prompt-source",
        type=str,
        choices=sorted(PROMPT_LOADERS.keys()),
        default="ja_stackoverflow",
        help="Prompt source under jfbench.prompts.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/dpo_dataset",
        help="Directory where the HuggingFace dataset will be stored.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20250101,
        help="Sampling seed.",
    )
    parser.add_argument(
        "--constraint-counts",
        type=str,
        default="1,2,4,8",
        help="Comma separated constraint counts.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=100,
        help="Maximum number of concurrent generation workers.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save dataset to disk after collecting this many new records.",
    )
    parser.add_argument(
        "--model-spec-json",
        type=str,
        required=True,
        help=(
            "JSON string describing the generation model. Must contain provider, model, and name."
        ),
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=500000,
        help="Abort once this many failures occur.",
    )
    parser.add_argument(
        "--n-chains",
        type=int,
        default=5,
        help="Number of independent initial generations per sample.",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        default=None,
        help="Maximum iterations after the initial generation. Defaults to n_constraints + 1.",
    )
    parser.add_argument(
        "--reward-base-url",
        type=str,
        default=DEFAULT_REWARD_BASE_URL,
        help="Base URL for the reward model endpoint.",
    )
    parser.add_argument(
        "--reward-api-key",
        type=str,
        default=DEFAULT_REWARD_API_KEY,
        help="API key used for the reward model.",
    )
    parser.add_argument(
        "--reward-model",
        type=str,
        default=DEFAULT_REWARD_MODEL,
        help="Reward model name or path.",
    )
    parser.add_argument(
        "--reward-token-limit",
        type=int,
        default=DEFAULT_REWARD_TOKENS,
        help="Maximum number of candidate tokens considered for reward scoring.",
    )
    parser.add_argument(
        "--reward-concurrency",
        type=int,
        default=DEFAULT_REWARD_CONCURRENCY,
        help="Maximum concurrent reward scoring requests.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level passed to logging.basicConfig.",
    )
    args = parser.parse_args(argv)
    args.constraint_counts = tuple(
        int(part.strip()) for part in args.constraint_counts.split(",") if part.strip()
    )
    if not args.constraint_counts:
        raise ValueError("At least one constraint count is required.")
    args.log_level = str(args.log_level).upper()
    level_value = getattr(logging, args.log_level, None)
    if not isinstance(level_value, int):
        raise ValueError(f"Invalid log level: {args.log_level}")
    if args.max_workers <= 0:
        raise ValueError("max_workers must be positive.")
    if args.save_every <= 0:
        raise ValueError("save_every must be positive.")
    if args.n_chains <= 0:
        raise ValueError("n_chains must be positive.")
    if args.n_iterations is not None and args.n_iterations <= 0:
        raise ValueError("n_iterations must be positive when provided.")
    if args.reward_token_limit <= 0:
        raise ValueError("reward_token_limit must be positive.")
    if args.reward_concurrency <= 0:
        raise ValueError("reward_concurrency must be positive.")
    spec_payload = _parse_model_spec_json(args.model_spec_json)
    try:
        args.model_spec = ModelSpec(**spec_payload)
    except TypeError as exc:
        raise ValueError("model_spec_json must include provider, model, and name.") from exc
    if args.model_spec.extra_body is not None and not isinstance(args.model_spec.extra_body, dict):
        raise ValueError("model_spec_json.extra_body must be an object when provided.")
    allowed_providers = {"openrouter", "vllm"}
    if args.model_spec.provider not in allowed_providers:
        raise ValueError(
            f"Unsupported provider '{args.model_spec.provider}'. Choose from openrouter or vllm."
        )
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level, logging.INFO),
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(_generate_async(args))


__all__ = [
    "DPORecordBuilder",
    "main",
    "parse_args",
]


if __name__ == "__main__":
    main()
