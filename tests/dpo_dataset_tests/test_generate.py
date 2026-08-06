from __future__ import annotations

import asyncio
import json
from typing import Any
from typing import Awaitable
from typing import Callable
from typing import cast
from typing import TYPE_CHECKING

import pytest


pytest.importorskip("transformers")

from jfbench.benchmark.build import BenchmarkData
from jfbench.benchmark.build import MetaData
from jfbench.dpo_dataset.generate import DPORecordBuilder
from jfbench.dpo_dataset.generate import parse_args
from jfbench.dpo_dataset.generate import RewardScorer
from jfbench.protocol import Constraint
from jfbench.protocol import Prompt
from jfbench.sft_dataset.generate import JudgeProtocol
from jfbench.sft_dataset.generate import ReasoningReview
from jfbench.sft_dataset.generate import RepetitionLevel
from jfbench.sft_dataset.generate import SampledExample


if TYPE_CHECKING:  # pragma: no cover - typing helper
    from pathlib import Path

    from jfbench.llm import LLMClient


class _DummyPrompt(Prompt):
    def __init__(self, text_value: str) -> None:
        self._text = text_value

    def text(self, constraints: list[Constraint], *, train_or_test: str = "train") -> str:
        return f"{self._text} ({len(constraints)} constraints)"

    @property
    def document(self) -> str:
        return "Document"


class _TokenConstraint:
    def __init__(self, token: str) -> None:
        self.token = token

    def evaluate(self, value: str) -> tuple[bool, None]:
        return (self.token in value), None

    def instructions(self, train_or_test: str = "train") -> str:
        return f"Include {self.token}"

    def rewrite_instructions(self) -> str:
        return f"{self.token}を含めてください。"

    @property
    def group(self) -> str:
        return "Test"

    @property
    def competitives(self) -> list[str]:
        return []

    def to_serializable_kwargs(self) -> dict[str, object]:
        return {"token": self.token}


class _StubJudge(JudgeProtocol):
    async def is_refusal(self, response: str) -> bool:
        return False

    async def repetition_level(self, text: str, *, label: str) -> RepetitionLevel:
        return "none"

    async def review_reasoning(self, reasoning_content: str) -> ReasoningReview:
        return ReasoningReview(repetition_level="none", is_rambling=False)


class _QueueJudge(_StubJudge):
    def __init__(self, reviews: list[ReasoningReview]) -> None:
        self._reviews = list(reviews)

    async def review_reasoning(self, reasoning_content: str) -> ReasoningReview:
        return self._reviews.pop(0)


class _StubLLMClient:
    def __init__(
        self,
        responses: list[str],
        *,
        provider: str = "openrouter",
        reasonings: list[str] | None = None,
    ) -> None:
        self._responses = list(responses)
        self._reasonings = list(reasonings) if reasonings is not None else None
        self.provider = provider

    async def async_ask(self, prompts: list[str]) -> tuple[list[str], list[Any]]:
        if not self._responses:
            raise RuntimeError("No responses available")
        value = self._responses.pop(0)
        detail = None
        if self._reasonings is not None:
            reasoning_value = self._reasonings.pop(0) if self._reasonings else ""
            message = _StubReasoningMessage(reasoning_value, self.provider)
            detail = _StubReasoningDetail(message)
        return [value], [detail]


class _StubReasoningMessage:
    def __init__(self, reasoning: str, provider: str) -> None:
        self.reasoning_content = reasoning if provider == "vllm" else ""
        self.reasoning = reasoning if provider == "openrouter" else ""
        self.reasoning_details = [{"text": reasoning}]


class _StubReasoningChoice:
    def __init__(self, message: _StubReasoningMessage) -> None:
        self.message = message


class _StubReasoningDetail:
    def __init__(self, message: _StubReasoningMessage) -> None:
        self.choices = [_StubReasoningChoice(message)]


class _DummyRewardScorer:
    def token_count(self, text: str) -> int:
        return 1


def _make_sample(token: str) -> SampledExample:
    prompt = _DummyPrompt("Prompt")
    constraint = _TokenConstraint(token)
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


def _sample_model_spec_json() -> str:
    payload = {
        "provider": "vllm",
        "model": "models/test-model",
        "model_short": "VllmModel",
        "extra_body": {"temperature": 0.7},
    }
    return json.dumps(payload)


class _StubbedDPORecordBuilder(DPORecordBuilder):
    def __init__(
        self,
        picker: Callable[[str, list[str], list[str]], Awaitable[dict[str, float | int | None]]],
        judge: JudgeProtocol,
        **kwargs: Any,
    ) -> None:
        super().__init__(judge=judge, **kwargs)
        self.reward_scorer = _RewardScorerStub(picker)


class _RewardScorerStub(RewardScorer):
    def __init__(
        self,
        picker: Callable[[str, list[str], list[str]], Awaitable[dict[str, float | int | None]]],
    ) -> None:
        super().__init__()
        self._picker = picker

    async def pick(
        self,
        prompt: str,
        chosen_candidates: list[str],
        rejected_candidates: list[str],
    ) -> dict[str, float | int | None]:
        return await self._picker(prompt, chosen_candidates, rejected_candidates)


def test_builder_collects_success_and_rejection() -> None:
    client = _StubLLMClient(
        ["token response", "failure response"],
        reasonings=["chosen reasoning", "rejected reasoning"],
    )
    judge = _StubJudge()

    async def _reward_picker_first(
        prompt: str,
        chosen_candidates: list[str],
        rejected_candidates: list[str],
    ) -> dict[str, float | int | None]:
        del prompt
        return {
            "chosen_idx": 0 if chosen_candidates else None,
            "chosen_score": 1.0,
            "rejected_idx": 0 if rejected_candidates else None,
            "rejected_score": -1.0,
        }

    builder = _StubbedDPORecordBuilder(
        client=cast("LLMClient", client),
        judge=judge,
        model="openai/gpt-oss-120b",
        model_name="gpt-oss-120b",
        temperature=0.7,
        n_chains=1,
        n_iterations=1,
        picker=_reward_picker_first,
    )
    sample = _make_sample("token")

    record = asyncio.run(builder.build_record(sample))

    assert record is not None
    assert record["chosen"]["origin"] == "generation"
    assert record["chosen"]["response"] == "token response"
    assert record["rejected"]["origin"] == "generation_after_success"
    assert record["rejected"]["response"] == "failure response"
    assert record["rejected"]["attempts"] == 2
    assert record["chosen_reasoning"] == "chosen reasoning"
    assert record["rejected_reasoning"] == "rejected reasoning"


def test_builder_returns_rewrite_pair_when_initial_response_invalid() -> None:
    client = _StubLLMClient(
        ["missing", "still missing", "final token"],
        reasonings=["r1", "r2", "r3"],
        provider="openrouter",
    )
    judge = _StubJudge()

    async def _reward_picker_last(
        prompt: str,
        chosen_candidates: list[str],
        rejected_candidates: list[str],
    ) -> dict[str, float | int | None]:
        del prompt
        chosen_idx = len(chosen_candidates) - 1 if chosen_candidates else None
        rejected_idx = len(rejected_candidates) - 1 if rejected_candidates else None
        return {
            "chosen_idx": chosen_idx,
            "chosen_score": 0.5,
            "rejected_idx": rejected_idx,
            "rejected_score": -0.5,
        }

    builder = _StubbedDPORecordBuilder(
        client=cast("LLMClient", client),
        judge=judge,
        model="openai/gpt-oss-120b",
        model_name="gpt-oss-120b",
        temperature=0.7,
        n_chains=1,
        n_iterations=2,
        picker=_reward_picker_last,
    )
    sample = _make_sample("token")

    record = asyncio.run(builder.build_record(sample))

    assert record is not None
    assert record["chosen"]["origin"] == "rewrite"
    assert record["chosen"]["response"] == "final token"
    assert record["chosen"]["attempts"] == 3
    assert record["rejected"]["origin"] == "rewrite"
    assert record["rejected"]["response"] == "still missing"
    assert record["chosen_reasoning"] == "r1r2r3"
    assert record["rejected_reasoning"] == "r1r2"


def test_builder_populates_reasoning_content_from_responses() -> None:
    client = _StubLLMClient(
        ["token response", "failure response"],
        provider="openrouter",
        reasonings=["chosen reasoning", "rejected reasoning"],
    )
    judge = _StubJudge()

    async def _reward_picker_first(
        prompt: str,
        chosen_candidates: list[str],
        rejected_candidates: list[str],
    ) -> dict[str, float | int | None]:
        del prompt
        return {
            "chosen_idx": 0 if chosen_candidates else None,
            "chosen_score": 1.0,
            "rejected_idx": 0 if rejected_candidates else None,
            "rejected_score": -1.0,
        }

    builder = _StubbedDPORecordBuilder(
        client=cast("LLMClient", client),
        judge=judge,
        model="openai/gpt-oss-120b",
        model_name="gpt-oss-120b",
        temperature=0.7,
        n_chains=1,
        n_iterations=1,
        picker=_reward_picker_first,
    )
    sample = _make_sample("token")

    record = asyncio.run(builder.build_record(sample))

    assert record is not None
    assert record["chosen_reasoning"] == "chosen reasoning"
    assert record["rejected_reasoning"] == "rejected reasoning"
    assert record["chosen"]["reasoning_content"] == "chosen reasoning"
    assert record["rejected"]["reasoning_content"] == "rejected reasoning"


def test_parse_args_requires_model_spec_json() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--n-data", "1"])


def test_parse_args_accepts_model_spec_json_override() -> None:
    spec = {
        "provider": "vllm",
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "model_short": "Llama-3-8B",
        "extra_body": {"base_url": "http://example:9000/v1"},
    }
    args = parse_args(["--n-data", "1", "--model-spec-json", json.dumps(spec)])

    assert args.model_spec.provider == "vllm"
    assert args.model_spec.model == spec["model"]
    assert args.model_spec.extra_body == spec["extra_body"]


def test_parse_args_exposes_output_dir_and_save_every(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--n-data",
            "1",
            "--output-dir",
            str(tmp_path / "dataset"),
            "--save-every",
            "5",
            "--model-spec-json",
            _sample_model_spec_json(),
        ]
    )

    assert args.output_dir == str(tmp_path / "dataset")
    assert args.save_every == 5


def test_parse_args_handles_reward_and_chain_options() -> None:
    args = parse_args(
        [
            "--n-data",
            "1",
            "--model-spec-json",
            _sample_model_spec_json(),
            "--n-chains",
            "3",
            "--n-iterations",
            "7",
            "--reward-base-url",
            "http://example:9999/v1",
            "--reward-api-key",
            "key",
            "--reward-model",
            "custom-reward",
            "--reward-token-limit",
            "128",
            "--reward-concurrency",
            "2",
        ]
    )

    assert args.n_chains == 3
    assert args.n_iterations == 7
    assert args.reward_base_url == "http://example:9999/v1"
    assert args.reward_api_key == "key"
    assert args.reward_model == "custom-reward"
    assert args.reward_token_limit == 128
    assert args.reward_concurrency == 2


@pytest.mark.anyio
async def test_review_reasoning_rewrites_minor() -> None:
    judge = _QueueJudge(
        [
            ReasoningReview(repetition_level="minor", is_rambling=True),
            ReasoningReview(repetition_level="none", is_rambling=False),
        ]
    )
    client = _StubLLMClient(["rewritten"], reasonings=["reasoning"])
    builder = DPORecordBuilder(
        client=cast("LLMClient", client),
        judge=judge,
        model="test",
        model_name="test",
        temperature=0.1,
        reward_model="test",
    )
    builder.reward_scorer = cast("RewardScorer", _DummyRewardScorer())
    reviewed = await builder._review_reasoning_content(  # noqa: SLF001
        instruction="Do something.",
        response="Answer.",
        reasoning_content="Some reasoning.",
        prompt_id="p1",
        attempt=1,
        origin="generation",
    )
    assert reviewed is not None
    reasoning_content, evaluation = reviewed
    assert reasoning_content == "rewritten"
    assert evaluation["reasoning_not_major_repetition"]
    assert evaluation["reasoning_not_rambling"]
    assert evaluation["reasoning_token_limit_ok"]


@pytest.mark.anyio
async def test_review_reasoning_allows_empty() -> None:
    judge = _QueueJudge([])
    client = _StubLLMClient(["rewritten"], reasonings=[""])
    builder = DPORecordBuilder(
        client=cast("LLMClient", client),
        judge=judge,
        model="test",
        model_name="test",
        temperature=0.1,
        reward_model="test",
    )
    builder.reward_scorer = cast("RewardScorer", _DummyRewardScorer())
    reviewed = await builder._review_reasoning_content(  # noqa: SLF001
        instruction="Do something.",
        response="Answer.",
        reasoning_content="",
        prompt_id="p1",
        attempt=1,
        origin="generation",
    )
    assert reviewed is not None
    reasoning_content, evaluation = reviewed
    assert reasoning_content == ""
    assert evaluation["reasoning_not_major_repetition"]
    assert evaluation["reasoning_not_rambling"]
    assert evaluation["reasoning_token_limit_ok"]
