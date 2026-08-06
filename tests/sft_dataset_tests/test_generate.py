from __future__ import annotations

import asyncio
from itertools import zip_longest
import json
import shutil
from typing import cast
from typing import TYPE_CHECKING

from datasets import Dataset
from datasets import Value
import numpy as np
import pytest

from jfbench.benchmark.build import _rule_constraint_factory
from jfbench.benchmark.build import BenchmarkData
from jfbench.benchmark.build import get_constraint_collections
from jfbench.benchmark.build import MetaData
from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.constraints.logic import NegationLogicConstraint
from jfbench.protocol import Constraint
from jfbench.protocol import Prompt
import jfbench.sft_dataset.generate as sft_generate_module
from jfbench.sft_dataset.generate import CandidateProcessor
from jfbench.sft_dataset.generate import collect_existing_keys
from jfbench.sft_dataset.generate import CombinationTracker
from jfbench.sft_dataset.generate import ConstraintCountScheduler
from jfbench.sft_dataset.generate import ConstraintFactoryWrapper
from jfbench.sft_dataset.generate import ConstraintInventory
from jfbench.sft_dataset.generate import ConstraintSampler
from jfbench.sft_dataset.generate import DatasetWriter
from jfbench.sft_dataset.generate import InteractiveController
from jfbench.sft_dataset.generate import InteractiveSkip
from jfbench.sft_dataset.generate import MAX_REASONING_CONTENT_TOKENS
from jfbench.sft_dataset.generate import ModelRunner
from jfbench.sft_dataset.generate import ModelSpec
from jfbench.sft_dataset.generate import parse_args
from jfbench.sft_dataset.generate import ReasoningReview
from jfbench.sft_dataset.generate import RepetitionLevel
from jfbench.sft_dataset.generate import SampledExample


if TYPE_CHECKING:  # pragma: no cover - typing helper
    import argparse
    from pathlib import Path

    from jfbench.llm import LLMClient


class DummyPrompt(Prompt):
    def __init__(self, idx: int) -> None:
        self.idx = idx

    def text(
        self, constraints: list[Constraint], *, train_or_test: str = "train"
    ) -> str:  # pragma: no cover - trivial wrapper
        return f"Prompt-{self.idx} with {len(constraints)} constraints"

    @property
    def document(self) -> str:  # pragma: no cover - trivial wrapper
        return f"Document-{self.idx}"


class DummyLLMClient:
    async def async_ask(
        self, prompts: list[str]
    ) -> tuple[list[str], list[None]]:  # pragma: no cover
        return ["True" for _ in prompts], [None] * len(prompts)


class ContainsTokenConstraint(ConstraintGroupMixin):
    def __init__(self, token: str) -> None:
        self.token = token

    def evaluate(self, value: str) -> tuple[bool, None]:
        return (self.token in value), None

    def instructions(
        self, train_or_test: str = "train"
    ) -> str:  # pragma: no cover - simple string builder
        return f"Include token {self.token}."

    def rewrite_instructions(self) -> str:
        return f"{self.token}を含むように修正してください。"


class _TestOnlyInstructionsConstraint(ConstraintGroupMixin):
    def evaluate(self, value: str) -> tuple[bool, None]:
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        if train_or_test != "test":
            raise ValueError("train_or_test must be 'test'.")
        return "Test-only instruction."

    def rewrite_instructions(self) -> str:
        return "Rewrite instruction."


def _stub_constraint_factory(*, seed: int | None = None) -> Constraint:
    token = f"token-{seed}" if seed is not None else "token"
    return ContainsTokenConstraint(token)


class _FakeInventory:
    def __init__(
        self, wrappers: list[ConstraintFactoryWrapper], constraint_set: str = "train"
    ) -> None:
        self._wrappers = wrappers
        self.constraint_set = constraint_set

    def iter_single_constraint_plans(
        self, prompt_count: int
    ) -> list[tuple[int, ConstraintFactoryWrapper]]:
        return [
            (prompt_idx, wrapper)
            for prompt_idx in range(prompt_count)
            for wrapper in self._wrappers
        ]

    @property
    def all_wrappers(self) -> tuple[ConstraintFactoryWrapper, ...]:
        return tuple(self._wrappers)


class _StubReasoningMessage:
    def __init__(self, reasoning: str) -> None:
        self.reasoning_content = reasoning
        self.reasoning = reasoning
        self.reasoning_details = [{"text": reasoning}]


class _StubReasoningChoice:
    def __init__(self, message: _StubReasoningMessage) -> None:
        self.message = message


class _StubReasoningDetail:
    def __init__(self, reasoning: str) -> None:
        self.choices = [_StubReasoningChoice(_StubReasoningMessage(reasoning))]


class _RewriteLLMClient:
    def __init__(
        self,
        responses: list[str],
        reasonings: list[str] | None = None,
        provider: str = "openrouter",
    ) -> None:
        self._responses = list(responses)
        self._reasonings = list(reasonings or [])
        self.prompts: list[str] = []
        self.provider = provider

    async def async_ask(
        self, prompts: list[str]
    ) -> tuple[list[str], list[_StubReasoningDetail]]:  # pragma: no cover - exercised indirectly
        self.prompts.extend(prompts)
        if not self._responses:
            raise RuntimeError("No rewrite responses available")
        value = self._responses.pop(0)
        reasoning = self._reasonings.pop(0) if self._reasonings else ""
        detail = _StubReasoningDetail(reasoning)
        return [value], [detail]


class _StubJudge:
    def __init__(
        self,
        *,
        refusals: set[str] | None = None,
        response_repetition: dict[str, RepetitionLevel] | None = None,
        reasoning_reviews: dict[str, ReasoningReview] | None = None,
        client: object | None = None,
    ) -> None:
        self._refusals = refusals or set()
        self._response_repetition = response_repetition or {}
        self._reasoning_reviews = reasoning_reviews or {}
        self.client = client

    async def is_refusal(self, response: str) -> bool:
        return response in self._refusals

    async def repetition_level(self, text: str, *, label: str) -> RepetitionLevel:
        return self._response_repetition.get(text, "none")

    async def review_reasoning(self, reasoning_content: str) -> ReasoningReview:
        return self._reasoning_reviews.get(reasoning_content, ReasoningReview("none", False))


class _StubRewardScorer:
    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self._scores = scores or {}

    async def score_many(self, prompt: str, answers: list[str]) -> list[float]:
        return [self._scores.get(answer, 0.0) for answer in answers]

    def token_count(self, text: str) -> int:
        return len(text.split())


class StubRunner:
    def __init__(
        self,
        spec: ModelSpec,
        responses: list[str],
        reasoning_contents: list[str] | None = None,
        delay: float = 0.0,
        rewrite_responses: list[str] | None = None,
        rewrite_reasonings: list[str] | None = None,
    ) -> None:
        self.spec = spec
        self._responses = iter(zip_longest(responses, reasoning_contents or [], fillvalue=""))
        self.delay = delay
        self.client = _RewriteLLMClient(rewrite_responses or [], rewrite_reasonings)

    async def generate(
        self, prompt: str
    ) -> tuple[str, str]:  # pragma: no cover - exercised indirectly
        if self.delay:
            await asyncio.sleep(self.delay)
        try:
            response, reasoning_content = next(self._responses)
            return response, reasoning_content
        except StopIteration:  # pragma: no cover - defensive
            return "", ""


def _make_sampled_example(token: str) -> SampledExample:
    prompt = DummyPrompt(0)
    constraint = ContainsTokenConstraint(token)
    meta = MetaData(
        prompt_source="ja_stackoverflow",
        data_id="sample-0",
        n_constraints=1,
        constraint_types=[constraint.__class__.__name__],
        constraint_groups=[constraint.group],
        constraint_instructions=[constraint.instructions()],
        prompt=prompt.text([constraint]),
    )
    benchmark = BenchmarkData(prompt=prompt, constraints=[constraint], meta_data=meta)
    return SampledExample(
        prompt_source="ja_stackoverflow",
        prompt_index=0,
        prompt_id="ja_stackoverflow-0",
        benchmark_data=benchmark,
    )


def test_constraint_sampler_respects_required_groups() -> None:
    inventory = ConstraintInventory()
    rng = np.random.default_rng(7)
    sampler = ConstraintSampler(
        prompt_source="ja_stackoverflow",
        prompts=[DummyPrompt(0)],
        inventory=inventory,
        judge_client=cast("LLMClient", DummyLLMClient()),
        rng=rng,
    )

    sample = sampler.sample(4)
    groups = sample.benchmark_data.meta_data.constraint_groups
    assert "Format" in groups
    constraints = sample.benchmark_data.constraints
    names = [constraint.__class__.__name__ for constraint in constraints]
    assert len(names) == len(set(names))
    for idx, constraint in enumerate(constraints):
        competitive_names = set(constraint.competitives)
        for other in constraints[idx + 1 :]:
            assert other.__class__.__name__ not in competitive_names


def test_constraint_inventory_includes_all_test_constraints() -> None:
    inventory = ConstraintInventory(constraint_set="test")
    collections = get_constraint_collections("test")

    expected_ifbench_factories = (
        collections.ifbench_count
        + collections.ifbench_format
        + collections.ifbench_ratio
        + collections.ifbench_repeat
        + collections.ifbench_sentence
        + collections.ifbench_words
    )
    expected_total = (
        len(collections.format)
        + len(collections.character)
        + len(collections.rule_content)
        + len(collections.llm_content)
        + len(collections.length)
        + len(collections.logic)
        + len(collections.meta_output)
        + len(collections.notation)
        + len(collections.rule_processing)
        + len(collections.llm_processing)
        + len(collections.structure)
        + len(collections.rule_style)
        + len(collections.llm_style)
        + len(expected_ifbench_factories)
    )

    registered_factories = {wrapper.factory for wrapper in inventory.all_wrappers}

    assert len(inventory.all_wrappers) == expected_total
    assert set(collections.logic).issubset(registered_factories)
    assert set(expected_ifbench_factories).issubset(registered_factories)


def test_constraint_sampler_shuffles_single_plans_by_prompt() -> None:
    wrapper = ConstraintFactoryWrapper(
        group="Format", mode="rule", factory=_stub_constraint_factory
    )
    inventory = _FakeInventory([wrapper])
    rng = np.random.default_rng(3)
    sampler = ConstraintSampler(
        prompt_source="ja_stackoverflow",
        prompts=[DummyPrompt(0), DummyPrompt(1)],
        inventory=cast("ConstraintInventory", inventory),
        judge_client=cast("LLMClient", DummyLLMClient()),
        rng=rng,
    )

    first = sampler.sample(1)
    second = sampler.sample(1)

    assert [first.prompt_index, second.prompt_index] == [1, 0]


def test_constraint_sampler_records_constraint_kwargs() -> None:
    wrapper = ConstraintFactoryWrapper(
        group="Format",
        mode="rule",
        factory=_rule_constraint_factory(ContainsTokenConstraint, token="seeded-token"),
    )
    inventory = _FakeInventory([wrapper])
    rng = np.random.default_rng(3)
    sampler = ConstraintSampler(
        prompt_source="ja_stackoverflow",
        prompts=[DummyPrompt(0)],
        inventory=cast("ConstraintInventory", inventory),
        judge_client=cast("LLMClient", DummyLLMClient()),
        rng=rng,
    )

    sample = sampler.sample(1)

    constraint = sample.benchmark_data.constraints[0]
    assert constraint.to_serializable_kwargs() == {"token": "seeded-token"}


def test_constraint_sampler_serializes_nested_constraint_kwargs() -> None:
    def _factory(*, seed: int | None = None, document: str | None = None) -> Constraint:
        _ = document
        return NegationLogicConstraint(
            positive_constraint=ContainsTokenConstraint("nested-token"),
            seed=seed,
        )

    wrapper = ConstraintFactoryWrapper(group="Logic", mode="rule", factory=_factory)
    inventory = _FakeInventory([wrapper])
    rng = np.random.default_rng(3)
    sampler = ConstraintSampler(
        prompt_source="ja_stackoverflow",
        prompts=[DummyPrompt(0)],
        inventory=cast("ConstraintInventory", inventory),
        judge_client=cast("LLMClient", DummyLLMClient()),
        rng=rng,
    )

    sample = sampler.sample(1)
    kwargs_payload = sample.benchmark_data.constraints[0].to_serializable_kwargs()
    assert kwargs_payload == {
        "positive_constraint": {
            "__type__": "constraint",
            "value": {
                "module": ContainsTokenConstraint.__module__,
                "name": "ContainsTokenConstraint",
                "kwargs": {"token": "nested-token"},
            },
        }
    }


def test_candidate_processor_generates_successful_response() -> None:
    sample = _make_sampled_example("valid")
    spec = ModelSpec(name="single", model="single-model", temperature=0.6)
    runner = StubRunner(
        spec,
        responses=["valid generation"],
        reasoning_contents=["reasoning for valid generation"],
    )
    processor = CandidateProcessor(
        cast("ModelRunner", runner),
        max_attempts=3,
        candidate_count=1,
        judge=_StubJudge(),
        reward_scorer=_StubRewardScorer(
            {"valid generation": 0.2, "reasoning for valid generation": 0.1}
        ),
    )

    record = asyncio.run(processor.process(sample))

    assert record is not None
    assert record["model"] == spec.model
    assert record["attempt"] == 1
    assert record["response"] == "valid generation"
    assert record["reasoning_content"] == "reasoning for valid generation"
    assert all(record["evaluation"].values())
    assert record["evaluation"]["constraints_passed"] is True
    assert record["evaluation"]["reasoning_not_major_repetition"] is True
    assert record["evaluation"]["reasoning_not_rambling"] is True
    assert record["evaluation"]["reasoning_token_limit_ok"] is True
    assert "constraint_kwargs" not in record
    assert len(record["constraints"]) == 1
    assert json.loads(record["constraints"][0]) == {
        "module": ContainsTokenConstraint.__module__,
        "name": "ContainsTokenConstraint",
        "kwargs": {"token": "valid"},
    }


def test_candidate_processor_retries_until_successful_attempt() -> None:
    sample = _make_sampled_example("token")
    spec = ModelSpec(name="single", model="single-model", temperature=0.6)
    runner = StubRunner(
        spec,
        responses=["absent", "token present"],
        reasoning_contents=["first reasoning", "second reasoning"],
    )
    processor = CandidateProcessor(
        cast("ModelRunner", runner),
        max_attempts=2,
        candidate_count=1,
        judge=_StubJudge(response_repetition={"absent": "major"}),
        reward_scorer=_StubRewardScorer({"token present": 0.2, "second reasoning": 0.2}),
    )

    record = asyncio.run(processor.process(sample))

    assert record is not None
    assert record["attempt"] == 2
    assert record["response"] == "token present"
    assert record["reasoning_content"] == "second reasoning"


def test_candidate_processor_regenerates_when_reasoning_empty() -> None:
    sample = _make_sampled_example("token")
    spec = ModelSpec(name="single", model="single-model", temperature=0.6)
    runner = StubRunner(
        spec,
        responses=["token present", "token present again"],
        reasoning_contents=["", "reasoning follows"],
    )
    processor = CandidateProcessor(
        cast("ModelRunner", runner),
        max_attempts=2,
        candidate_count=1,
        judge=_StubJudge(),
        reward_scorer=_StubRewardScorer({"token present again": 0.2, "reasoning follows": 0.2}),
    )

    record = asyncio.run(processor.process(sample))

    assert record is not None
    assert record["attempt"] == 2
    assert record["response"] == "token present again"
    assert record["reasoning_content"] == "reasoning follows"


def test_candidate_processor_rejects_duplicate_reasoning_phrases() -> None:
    sample = _make_sampled_example("valid")
    spec = ModelSpec(name="single", model="single-model", temperature=0.6)
    runner = StubRunner(
        spec,
        responses=["valid generation"],
        reasoning_contents=["This is a repeated phrase. This is a repeated phrase."],
    )
    processor = CandidateProcessor(
        cast("ModelRunner", runner),
        max_attempts=1,
        candidate_count=1,
        judge=_StubJudge(
            reasoning_reviews={
                "This is a repeated phrase. This is a repeated phrase.": ReasoningReview(
                    "major", False
                )
            }
        ),
        reward_scorer=_StubRewardScorer(),
    )

    record = asyncio.run(processor.process(sample))

    assert record is None


def test_candidate_processor_rejects_long_reasoning_content() -> None:
    sample = _make_sampled_example("valid")
    spec = ModelSpec(name="single", model="single-model", temperature=0.6)
    long_reasoning = "a " * (MAX_REASONING_CONTENT_TOKENS + 1)
    runner = StubRunner(
        spec,
        responses=["valid generation"],
        reasoning_contents=["rambling reasoning"],
        rewrite_responses=[long_reasoning],
    )
    processor = CandidateProcessor(
        cast("ModelRunner", runner),
        max_attempts=1,
        candidate_count=1,
        judge=_StubJudge(reasoning_reviews={"rambling reasoning": ReasoningReview("minor", True)}),
        reward_scorer=_StubRewardScorer(),
    )

    record = asyncio.run(processor.process(sample))

    assert record is None


def test_candidate_processor_returns_rewritten_output_when_initial_response_invalid() -> None:
    sample = _make_sampled_example("token")
    spec = ModelSpec(name="rewriter", model="rewriter-model", temperature=0.6)
    runner = StubRunner(
        spec,
        responses=["missing"],
        reasoning_contents=["needs rewrite"],
        rewrite_responses=["contains token"],
        rewrite_reasonings=["rewrite reason"],
    )
    processor = CandidateProcessor(
        cast("ModelRunner", runner),
        max_attempts=1,
        candidate_count=1,
        judge=_StubJudge(),
        reward_scorer=_StubRewardScorer(
            {"contains token": 0.2, "needs rewriterewrite reason": 0.1}
        ),
    )

    record = asyncio.run(processor.process(sample))

    assert record is not None
    assert record["response"] == "contains token"
    assert record["reasoning_content"] == "needs rewriterewrite reason"
    assert all(record["evaluation"].values())


def test_candidate_processor_selects_best_reward_candidate() -> None:
    sample = _make_sampled_example("token")
    spec = ModelSpec(name="chooser", model="chooser-model", temperature=0.6)
    runner = StubRunner(
        spec,
        responses=["token candidate one", "token candidate two"],
        reasoning_contents=["reasoning one", "reasoning two"],
    )
    rewards = {
        "token candidate one": 0.5,
        "token candidate two": 0.9,
        "reasoning one": 0.9,
        "reasoning two": 0.1,
    }
    processor = CandidateProcessor(
        cast("ModelRunner", runner),
        max_attempts=1,
        candidate_count=2,
        judge=_StubJudge(),
        reward_scorer=_StubRewardScorer(rewards),
    )

    record = asyncio.run(processor.process(sample))

    assert record is not None
    assert record["response"] == "token candidate one"


def test_candidate_processor_raises_skip_when_user_declines_evaluation() -> None:
    sample = _make_sampled_example("accept")
    spec = ModelSpec(name="interactive", model="interactive-model", temperature=0.6)
    runner = StubRunner(spec, responses=["accept"], reasoning_contents=["reasoning content"])
    controller = InteractiveController(enabled=True, prompt_func=lambda _: "n")
    processor = CandidateProcessor(
        cast("ModelRunner", runner),
        max_attempts=1,
        candidate_count=1,
        judge=_StubJudge(),
        reward_scorer=_StubRewardScorer({"accept": 0.1}),
        interactive_controller=controller,
    )

    with pytest.raises(InteractiveSkip):
        asyncio.run(processor.process(sample))


def test_candidate_processor_skips_rewrite_when_declined() -> None:
    sample = _make_sampled_example("token")
    spec = ModelSpec(name="interactive", model="interactive-model", temperature=0.6)
    runner = StubRunner(spec, responses=["missing"], rewrite_responses=["contains token"])
    answers = iter(["y", "n"])

    def prompt(_: str) -> str:
        try:
            return next(answers)
        except StopIteration:  # pragma: no cover - defensive fallback
            return "y"

    controller = InteractiveController(enabled=True, prompt_func=prompt)
    processor = CandidateProcessor(
        cast("ModelRunner", runner),
        max_attempts=1,
        candidate_count=1,
        judge=_StubJudge(),
        reward_scorer=_StubRewardScorer(),
        interactive_controller=controller,
    )

    record = asyncio.run(processor.process(sample))

    assert record is None
    assert runner.client.prompts == []


def test_interactive_controller_confirm_save_respects_prompt() -> None:
    controller = InteractiveController(enabled=True, prompt_func=lambda _: "n")
    record = {
        "prompt_id": "ja_stackoverflow-0",
        "model": "fast",
        "response": "sample response",
        "evaluation": {"Format": True},
    }

    should_save = asyncio.run(controller.confirm_save(record))

    assert should_save is False


def test_combination_tracker_blocks_duplicates() -> None:
    async def _exercise() -> None:
        tracker = CombinationTracker()
        record = {
            "prompt_id": "ja_stackoverflow-0",
            "constraints": [
                {"name": "FmtA", "instructions": "A"},
                {"name": "CharA", "instructions": "B"},
            ],
        }
        key = CombinationTracker.key_from_mapping(record)
        assert await tracker.reserve(key)
        assert not await tracker.reserve(key)
        await tracker.release(key)
        assert await tracker.reserve(key)
        await tracker.mark_final(key)
        assert not await tracker.reserve(key)

    asyncio.run(_exercise())


def test_combination_tracker_reservations_are_atomic() -> None:
    async def _exercise() -> None:
        tracker = CombinationTracker()
        record = {
            "prompt_id": "ja_stackoverflow-1",
            "constraints": [
                {"name": "FmtA", "instructions": "A"},
                {"name": "CharA", "instructions": "B"},
            ],
        }
        key = CombinationTracker.key_from_mapping(record)

        async def _attempt() -> bool:
            return await tracker.reserve(key)

        results = await asyncio.gather(*[_attempt() for _ in range(10)])
        assert sum(1 for value in results if value) == 1

    asyncio.run(_exercise())


def test_combination_tracker_key_from_sample_uses_metadata_instructions() -> None:
    prompt = DummyPrompt(0)
    constraint = _TestOnlyInstructionsConstraint()
    meta = BenchmarkData.build_meta_data(
        prompt_source="ja_stackoverflow",
        data_id="sample-test-only-0",
        prompt=prompt,
        constraints=[constraint],
        constraint_set="test",
    )
    benchmark = BenchmarkData(prompt=prompt, constraints=[constraint], meta_data=meta)
    sample = SampledExample(
        prompt_source="ja_stackoverflow",
        prompt_index=0,
        prompt_id="ja_stackoverflow-0",
        benchmark_data=benchmark,
    )

    key = CombinationTracker.key_from_sample(sample)

    assert key == "ja_stackoverflow-0||_TestOnlyInstructionsConstraint::Test-only instruction."


def test_dataset_writer_flushes_each_add(tmp_path: Path) -> None:
    output_dir = tmp_path / "dataset"
    writer = DatasetWriter(output_dir, save_every=10, flush_each_add=True)
    writer.add({"value": 1})
    assert output_dir.exists()
    writer.add({"value": 2})
    writer.flush()
    dataset = Dataset.load_from_disk(str(output_dir))
    assert len(dataset) == 2


def test_dataset_writer_appends_to_existing_dataset(tmp_path: Path) -> None:
    output_dir = tmp_path / "dataset"
    Dataset.from_list([{"value": 1}]).save_to_disk(str(output_dir))
    writer = DatasetWriter(output_dir, save_every=1)

    writer.add({"value": 2})
    writer.flush()

    dataset = Dataset.load_from_disk(str(output_dir))
    assert dataset["value"] == [1, 2]


def test_dataset_writer_restores_backup_when_move_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "dataset"
    Dataset.from_list([{"value": 1}]).save_to_disk(str(output_dir))
    writer = DatasetWriter(output_dir, save_every=1, flush_each_add=True)

    def _fail_move(src: str, dst: str) -> None:
        raise RuntimeError("move failed")

    monkeypatch.setattr(shutil, "move", _fail_move)

    with pytest.raises(RuntimeError):
        writer.add({"value": 2})

    dataset = Dataset.load_from_disk(str(output_dir))
    assert dataset["value"] == [1]


def test_dataset_writer_normalizes_constraints_schema(tmp_path: Path) -> None:
    output_dir = tmp_path / "dataset"
    writer = DatasetWriter(output_dir, save_every=1)

    writer.add(
        {
            "value": 1,
            "constraints": [
                json.dumps(
                    {
                        "module": "jfbench.constraints.length.sentences",
                        "name": "SentencesLengthConstraint",
                        "kwargs": {"min_sentences": 1, "max_sentences": 2},
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            ],
        }
    )
    writer.add(
        {
            "value": 2,
            "constraints": [
                json.dumps(
                    {
                        "module": "jfbench.constraints.processing.prefix",
                        "name": "PrefixProcessingConstraint",
                        "kwargs": {"prefix": "beta"},
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            ],
        }
    )
    writer.flush()

    dataset = Dataset.load_from_disk(str(output_dir))
    assert len(dataset) == 2
    assert isinstance(dataset.features["constraints"].feature, Value)
    assert dataset.features["constraints"].feature.dtype == "string"


def test_dataset_writer_aligns_constraints_to_existing_feature(tmp_path: Path) -> None:
    output_dir = tmp_path / "dataset"
    writer = DatasetWriter(output_dir, save_every=1)
    writer.add(
        {
            "value": 1,
            "constraints": [
                json.dumps(
                    {
                        "module": "jfbench.constraints.content.included",
                        "name": "IncludedContentConstraint",
                        "kwargs": {"keywords": ["月", "火"]},
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            ],
        }
    )
    writer.add(
        {
            "value": 2,
            "constraints": [
                json.dumps(
                    {
                        "module": "jfbench.constraints.content.included",
                        "name": "IncludedContentConstraint",
                        "kwargs": {"keywords": {"日本": 1, "東京": 2}},
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            ],
        }
    )
    writer.flush()

    dataset = Dataset.load_from_disk(str(output_dir))
    assert len(dataset) == 2
    assert json.loads(dataset["constraints"][0][0]) == {
        "module": "jfbench.constraints.content.included",
        "name": "IncludedContentConstraint",
        "kwargs": {"keywords": ["月", "火"]},
    }
    assert json.loads(dataset["constraints"][1][0]) == {
        "module": "jfbench.constraints.content.included",
        "name": "IncludedContentConstraint",
        "kwargs": {"keywords": {"日本": 1, "東京": 2}},
    }


def test_collect_existing_keys_ignores_duplicates() -> None:
    dataset = Dataset.from_list(
        [
            {
                "prompt_id": "ja_stackoverflow-0",
                "constraint_instructions": ["A"],
                "constraints": [
                    '{"module":"x.y","name":"FmtA","kwargs":{}}',
                ],
            },
            {
                "prompt_id": "ja_stackoverflow-0",
                "constraint_instructions": ["A"],
                "constraints": [
                    '{"module":"x.y","name":"FmtA","kwargs":{}}',
                ],
            },
        ]
    )

    keys = collect_existing_keys(dataset)

    assert len(keys) == 1


def test_constraint_count_scheduler_samples_counts_randomly() -> None:
    scheduler = ConstraintCountScheduler(total=6, counts=(4, 2, 8), rng=np.random.default_rng(0))

    async def _drain() -> list[int]:
        values: list[int] = []
        while True:
            count = await scheduler.next_count()
            if count is None:
                break
            values.append(count)
        return values

    counts = asyncio.run(_drain())

    assert counts == [4, 4, 8, 8, 2, 2]


def test_constraint_count_scheduler_prefers_smaller_counts_when_requested() -> None:
    scheduler = ConstraintCountScheduler(
        total=6,
        counts=(4, 2, 8),
        rng=np.random.default_rng(0),
        favor_smaller_counts=True,
    )

    async def _drain() -> list[int]:
        values: list[int] = []
        while True:
            count = await scheduler.next_count()
            if count is None:
                break
            values.append(count)
        return values

    counts = asyncio.run(_drain())

    assert counts == [2, 2, 4, 4, 8, 8]


def test_parse_args_sets_default_log_level() -> None:
    args = _parse_with_defaults()
    assert args.log_level == "WARNING"


def test_parse_args_accepts_custom_log_level() -> None:
    args = _parse_with_defaults(["--log-level", "info"])
    assert args.log_level == "INFO"


def test_parse_args_defaults_max_attempts_to_three() -> None:
    args = _parse_with_defaults()
    assert args.max_attempts == 3


def test_parse_args_populates_model_spec() -> None:
    args = _parse_with_defaults()
    assert args.model_spec.name == "stub"
    assert args.model_spec.model == "stub-model"


def test_parse_args_accepts_constraint_set() -> None:
    args = _parse_with_defaults(["--constraint-set", "test"])
    assert args.constraint_set == "test"


def test_generate_async_passes_constraint_set_to_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _StopGeneration(Exception):
        pass

    received: list[str] = []

    class _InventoryStub:
        def __init__(self, *, constraint_set: str = "train") -> None:
            received.append(constraint_set)
            self.constraint_set = constraint_set
            self.all_wrappers: tuple[ConstraintFactoryWrapper, ...] = ()

    class _ClientStub:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class _JudgeStub:
        def __init__(self, client: object) -> None:
            self.client = client

    def _prompt_loader() -> list[Prompt]:
        return []

    class _SamplerStub:
        def __init__(self, **kwargs: object) -> None:
            inventory = cast("_InventoryStub", kwargs["inventory"])
            assert inventory.constraint_set == "test"
            raise _StopGeneration()

    monkeypatch.setattr(
        sft_generate_module, "_resolve_prompt_loader", lambda _name: _prompt_loader
    )
    monkeypatch.setattr(sft_generate_module, "LLMClient", _ClientStub)
    monkeypatch.setattr(sft_generate_module, "LLMJudge", _JudgeStub)
    monkeypatch.setattr(sft_generate_module, "ConstraintInventory", _InventoryStub)
    monkeypatch.setattr(sft_generate_module, "ConstraintSampler", _SamplerStub)

    args = _parse_with_defaults(["--constraint-set", "test"])
    with pytest.raises(_StopGeneration):
        asyncio.run(getattr(sft_generate_module, "_generate_async")(args))
    assert received == ["test"]


DEFAULT_SPEC_JSON = '{"name": "stub", "model": "stub-model", "provider": "vllm"}'


def _parse_with_defaults(extra_args: list[str] | None = None) -> argparse.Namespace:
    base = ["--n-data", "1", "--mode-spec-json", DEFAULT_SPEC_JSON]
    if extra_args:
        base.extend(extra_args)
    return parse_args(base)
