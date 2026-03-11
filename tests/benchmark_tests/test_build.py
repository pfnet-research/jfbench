import asyncio
import logging
from typing import Any
from typing import cast
from typing import Sequence
from typing import TYPE_CHECKING

import pytest

import jfbench.benchmark.build as build
from jfbench.benchmark.build import BenchmarkData
from jfbench.benchmark.build import ConstraintCollections
from jfbench.benchmark.build import ConstraintSetName
from jfbench.benchmark.build import get_benchmark_data_with_multiple_constraints
from jfbench.benchmark.build import get_constraint_collections
from jfbench.benchmark.build import MetaData
from jfbench.constraints import ConjunctionCountIfbenchConstraint
from jfbench.constraints import PlainStyleConstraint
from jfbench.constraints import WordsLengthConstraint
from jfbench.protocol import Constraint


if TYPE_CHECKING:  # pragma: no cover - typing helper
    from jfbench.llm import LLMClient


class _StubPrompt:
    def text(self, constraints: list[Constraint], *, train_or_test: str = "train") -> str:
        return "prompt"

    @property
    def document(self) -> str:
        return "prompt"


class _NamedPrompt:
    def __init__(self, value: str) -> None:
        self._value = value

    def text(self, constraints: list[Constraint], *, train_or_test: str = "train") -> str:
        return self._value

    @property
    def document(self) -> str:
        return self._value


class _OrderSensitivePrompt:
    def __init__(self, value: str) -> None:
        self._value = value

    def text(self, constraints: list[Constraint], *, train_or_test: str = "train") -> str:
        constraint_names = "-".join(constraint.__class__.__name__ for constraint in constraints)
        return f"{self._value}-{constraint_names}"

    @property
    def document(self) -> str:
        return self._value


class _SyncConstraint:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def evaluate(self, value: str) -> tuple[bool, None]:
        self.seen.append(value)
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        return "stub"

    @property
    def group(self) -> str:
        return "Test"

    def rewrite_instructions(self) -> str:
        return "rewrite sync"

    @property
    def competitives(self) -> list[str]:
        return []

    def to_serializable_kwargs(self) -> dict[str, object]:
        return {}


class _AsyncConstraint:
    def __init__(self) -> None:
        self.seen: list[str] = []

    async def evaluate(self, value: str) -> tuple[bool, None]:
        self.seen.append(value)
        await asyncio.sleep(0)
        return False, None

    def instructions(self, train_or_test: str = "train") -> str:
        return "stub"

    @property
    def group(self) -> str:
        return "Test"

    def rewrite_instructions(self) -> str:
        return "rewrite async"

    @property
    def competitives(self) -> list[str]:
        return []

    def to_serializable_kwargs(self) -> dict[str, object]:
        return {}


class _AsyncVotingConstraint:
    def __init__(self, outcomes: list[bool]) -> None:
        self._outcomes = outcomes
        self.seen: list[str] = []
        self.calls = 0

    async def evaluate(self, value: str) -> tuple[bool, str | None]:
        self.seen.append(value)
        outcome = self._outcomes[self.calls]
        self.calls += 1
        await asyncio.sleep(0)
        return outcome, None if outcome else "failed"

    def instructions(self, train_or_test: str = "train") -> str:
        return "stub"

    @property
    def group(self) -> str:
        return "Test"

    def rewrite_instructions(self) -> str:
        return "rewrite async voting"

    @property
    def competitives(self) -> list[str]:
        return []

    def to_serializable_kwargs(self) -> dict[str, object]:
        return {}


class _AsyncParallelConstraint:
    def __init__(self) -> None:
        self.active = 0
        self.max_active = 0
        self.calls = 0

    async def evaluate(self, value: str) -> tuple[bool, None]:
        _ = value
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        return "stub"

    @property
    def group(self) -> str:
        return "Test"

    def rewrite_instructions(self) -> str:
        return "rewrite async parallel"

    @property
    def competitives(self) -> list[str]:
        return []

    def to_serializable_kwargs(self) -> dict[str, object]:
        return {}


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


class _RewriteValueConstraint:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, value: str) -> tuple[bool, None]:
        self.calls += 1
        return (value == "formatted"), None

    def instructions(self, train_or_test: str = "train") -> str:
        return "Format via mdformat"

    def rewrite_instructions(self) -> str:
        raise AssertionError("rewrite_instructions should not be called")

    @property
    def group(self) -> str:
        return "Test"

    def rewrite_value(self, value: str) -> str:
        assert value != "formatted"
        return "formatted"

    @property
    def competitives(self) -> list[str]:
        return []

    def to_serializable_kwargs(self) -> dict[str, object]:
        return {}


class _RewriteClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    async def async_ask(self, prompts: list[str]) -> tuple[list[str], list[None]]:
        self.prompts.extend(prompts)
        if not self._responses:
            raise RuntimeError("No more responses")
        response = self._responses.pop(0)
        return [response], [None]


class _UppercaseRewriteConstraint:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, value: str) -> tuple[bool, None]:
        return value.isupper(), None

    def instructions(self, train_or_test: str = "train") -> str:
        return "Use uppercase"

    def rewrite_instructions(self) -> str:
        return "すべて大文字で出力してください。"

    @property
    def group(self) -> str:
        return "Test"

    def rewrite_value(self, value: str) -> str:
        self.calls += 1
        return value.upper()

    @property
    def competitives(self) -> list[str]:
        return []

    def to_serializable_kwargs(self) -> dict[str, object]:
        return {}


class _FormatConstraint:
    def evaluate(self, value: str) -> tuple[bool, None]:
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        return "format"

    @property
    def group(self) -> str:
        return "Format"

    def rewrite_instructions(self) -> str:
        return "format"

    @property
    def competitives(self) -> list[str]:
        return ["_ConflictingConstraint"]

    def to_serializable_kwargs(self) -> dict[str, object]:
        return {}


class _AlternateFormatConstraint:
    def evaluate(self, value: str) -> tuple[bool, None]:
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        return "alternate format"

    @property
    def group(self) -> str:
        return "Format"

    def rewrite_instructions(self) -> str:
        return "alternate format"

    @property
    def competitives(self) -> list[str]:
        return []

    def to_serializable_kwargs(self) -> dict[str, object]:
        return {}


class _CharacterConstraint:
    def evaluate(self, value: str) -> tuple[bool, None]:
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        return "character"

    @property
    def group(self) -> str:
        return "Character"

    def rewrite_instructions(self) -> str:
        return "character"

    @property
    def competitives(self) -> list[str]:
        return []

    def to_serializable_kwargs(self) -> dict[str, object]:
        return {}


class _ConflictingConstraint:
    def evaluate(self, value: str) -> tuple[bool, None]:
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        return "conflict"

    @property
    def group(self) -> str:
        return "Content"

    def rewrite_instructions(self) -> str:
        return "conflict"

    @property
    def competitives(self) -> list[str]:
        return ["_FormatConstraint"]

    def to_serializable_kwargs(self) -> dict[str, object]:
        return {}


class _SafeConstraint:
    def evaluate(self, value: str) -> tuple[bool, None]:
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        return "safe"

    @property
    def group(self) -> str:
        return "Structure"

    def rewrite_instructions(self) -> str:
        return "safe"

    @property
    def competitives(self) -> list[str]:
        return []

    def to_serializable_kwargs(self) -> dict[str, object]:
        return {}


class _ContentConstraint:
    def evaluate(self, value: str) -> tuple[bool, None]:
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        return "content"

    @property
    def group(self) -> str:
        return "Content"

    def rewrite_instructions(self) -> str:
        return "content"

    @property
    def competitives(self) -> list[str]:
        return []

    def to_serializable_kwargs(self) -> dict[str, object]:
        return {}


_TEST_PROMPT_SOURCE = "test_source"
_ASYNC_EVAL_TRIALS = 3


def _make_benchmark(constraints: Sequence[Constraint]) -> BenchmarkData:
    constraint_list = list(constraints)
    return BenchmarkData(
        prompt=_StubPrompt(),
        constraints=constraint_list,
        meta_data=MetaData(
            prompt_source=_TEST_PROMPT_SOURCE,
            data_id="test-data",
            n_constraints=len(constraint_list),
            constraint_types=[constraint.__class__.__name__ for constraint in constraint_list],
            constraint_groups=[constraint.group for constraint in constraint_list],
            constraint_instructions=[constraint.instructions() for constraint in constraint_list],
            prompt="prompt",
        ),
    )


def test_benchmark_data_evaluate_handles_sync_constraints() -> None:
    constraint = _SyncConstraint()
    data = BenchmarkData(
        prompt=_StubPrompt(),
        constraints=[constraint],
        meta_data=MetaData(
            prompt_source=_TEST_PROMPT_SOURCE,
            data_id="test-sync",
            n_constraints=1,
            constraint_types=["_SyncConstraint"],
            constraint_groups=["Test"],
            constraint_instructions=["stub"],
            prompt="prompt",
        ),
    )

    results = asyncio.run(data.evaluate("value"))

    assert results == {"_SyncConstraint": True}
    assert constraint.seen == ["value"]


def test_benchmark_data_evaluate_handles_async_constraints() -> None:
    constraint = _AsyncConstraint()
    data = BenchmarkData(
        prompt=_StubPrompt(),
        constraints=[constraint],
        meta_data=MetaData(
            prompt_source=_TEST_PROMPT_SOURCE,
            data_id="test-async",
            n_constraints=1,
            constraint_types=["_AsyncConstraint"],
            constraint_groups=["Test"],
            constraint_instructions=["stub"],
            prompt="prompt",
        ),
    )

    results = asyncio.run(data.evaluate("value"))

    assert results == {"_AsyncConstraint": False}
    assert constraint.seen == ["value"] * _ASYNC_EVAL_TRIALS


def test_benchmark_data_evaluate_uses_majority_vote_for_awaitable_evaluations() -> None:
    constraint = _AsyncVotingConstraint([True, False, True])
    data = BenchmarkData(
        prompt=_StubPrompt(),
        constraints=[constraint],
        meta_data=MetaData(
            prompt_source=_TEST_PROMPT_SOURCE,
            data_id="test-async-majority",
            n_constraints=1,
            constraint_types=["_AsyncVotingConstraint"],
            constraint_groups=["Test"],
            constraint_instructions=["stub"],
            prompt="prompt",
        ),
    )

    results = asyncio.run(data.evaluate("value"))

    assert results == {"_AsyncVotingConstraint": True}
    assert constraint.calls == _ASYNC_EVAL_TRIALS
    assert constraint.seen == ["value"] * _ASYNC_EVAL_TRIALS


def test_benchmark_data_evaluate_uses_majority_vote_for_awaitable_failures() -> None:
    constraint = _AsyncVotingConstraint([False, True, False])
    data = BenchmarkData(
        prompt=_StubPrompt(),
        constraints=[constraint],
        meta_data=MetaData(
            prompt_source=_TEST_PROMPT_SOURCE,
            data_id="test-async-majority-failure",
            n_constraints=1,
            constraint_types=["_AsyncVotingConstraint"],
            constraint_groups=["Test"],
            constraint_instructions=["stub"],
            prompt="prompt",
        ),
    )

    results = asyncio.run(data.evaluate("value"))

    assert results == {"_AsyncVotingConstraint": False}
    assert constraint.calls == _ASYNC_EVAL_TRIALS
    assert constraint.seen == ["value"] * _ASYNC_EVAL_TRIALS


def test_benchmark_data_evaluate_runs_awaitable_trials_in_parallel() -> None:
    constraint = _AsyncParallelConstraint()
    data = BenchmarkData(
        prompt=_StubPrompt(),
        constraints=[constraint],
        meta_data=MetaData(
            prompt_source=_TEST_PROMPT_SOURCE,
            data_id="test-async-parallel",
            n_constraints=1,
            constraint_types=["_AsyncParallelConstraint"],
            constraint_groups=["Test"],
            constraint_instructions=["stub"],
            prompt="prompt",
        ),
    )

    results = asyncio.run(data.evaluate("value"))

    assert results == {"_AsyncParallelConstraint": True}
    assert constraint.calls == _ASYNC_EVAL_TRIALS
    assert constraint.max_active > 1


def test_benchmark_data_rewrite_returns_input_when_already_valid() -> None:
    constraint = _TokenConstraint("ok")
    benchmark = _make_benchmark([constraint])
    client = _RewriteClient(["unused"])

    value, evaluation = asyncio.run(benchmark.rewrite("ok", cast("LLMClient", client)))

    assert value == "ok"
    assert evaluation[constraint.__class__.__name__]
    assert client.prompts == []


def test_benchmark_data_rewrite_with_trace_returns_previous_failure() -> None:
    constraint = _TokenConstraint("token")
    benchmark = _make_benchmark([constraint])
    client = _RewriteClient(["token"])

    trace = asyncio.run(benchmark.rewrite_with_trace("", cast("LLMClient", client)))

    assert trace.value == "token"
    assert trace.previous_value == ""
    assert trace.steps == 1
    assert trace.previous_evaluation is not None


def test_benchmark_data_rewrite_with_trace_skips_history_when_valid() -> None:
    constraint = _TokenConstraint("token")
    benchmark = _make_benchmark([constraint])
    client = _RewriteClient(["unused"])

    trace = asyncio.run(benchmark.rewrite_with_trace("token", cast("LLMClient", client)))

    assert trace.value == "token"
    assert trace.previous_value is None
    assert trace.steps == 0
    assert trace.previous_evaluation is None
    assert client.prompts == []


def test_build_meta_data_records_prompt_and_instructions() -> None:
    constraint = _TokenConstraint("needle")
    meta = BenchmarkData.build_meta_data(
        prompt_source=_TEST_PROMPT_SOURCE,
        data_id="meta-test",
        prompt=_StubPrompt(),
        constraints=[constraint],
        constraint_set="train",
    )
    assert meta.prompt == "prompt"
    assert meta.constraint_instructions == ["Include needle"]


class _ReasonTokenConstraint:
    def __init__(self, token: str, reason: str) -> None:
        self.token = token
        self.reason = reason

    def evaluate(self, value: str) -> tuple[bool, str | None]:
        if self.token in value:
            return True, None
        return False, self.reason

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
        return {"token": self.token, "reason": self.reason}


class _NoRewriteInstructionsConstraint:
    def evaluate(self, value: str) -> tuple[bool, None]:
        return True, None

    def instructions(self, train_or_test: str = "train") -> str:
        return "Fallback instruction"

    def rewrite_instructions(self) -> str:
        raise ValueError("no rewrite instructions")

    @property
    def group(self) -> str:
        return "Test"

    @property
    def competitives(self) -> list[str]:
        return []

    def to_serializable_kwargs(self) -> dict[str, object]:
        return {}


def test_benchmark_data_rewrite_prompts_include_all_constraints() -> None:
    constraints: list[Constraint] = [_TokenConstraint("A"), _TokenConstraint("B")]
    benchmark = _make_benchmark(constraints)
    client = _RewriteClient(["A", "AB"])

    value, evaluation = asyncio.run(benchmark.rewrite("", cast("LLMClient", client)))

    assert value == "AB"
    assert all(evaluation.values())
    assert len(client.prompts) == 2
    for prompt in client.prompts:
        assert "Aを含めてください" in prompt
        assert "Bを含めてください" in prompt


def test_benchmark_data_rewrite_raises_after_attempt_limit() -> None:
    constraint = _TokenConstraint("X")
    benchmark = _make_benchmark([constraint])
    client = _RewriteClient([""] * 3)

    with pytest.raises(RuntimeError):
        asyncio.run(benchmark.rewrite("", cast("LLMClient", client)))


def test_benchmark_data_rewrite_uses_constraint_rewrite_value() -> None:
    constraint = _RewriteValueConstraint()
    benchmark = _make_benchmark([constraint])
    client = _RewriteClient(["should not be used"])

    value, evaluation = asyncio.run(benchmark.rewrite("raw", cast("LLMClient", client)))

    assert value == "formatted"
    assert evaluation["_RewriteValueConstraint"] is True
    assert constraint.calls >= 2
    assert client.prompts == []


def test_benchmark_data_rewrite_applies_rewrite_value_after_llm_pass() -> None:
    uppercase_constraint = _UppercaseRewriteConstraint()
    constraints: list[Constraint] = [_TokenConstraint("A"), uppercase_constraint]
    benchmark = _make_benchmark(constraints)
    client = _RewriteClient(["a"])

    value, evaluation = asyncio.run(benchmark.rewrite("", cast("LLMClient", client)))

    assert value == "A"
    assert all(evaluation.values())
    assert uppercase_constraint.calls >= 1
    assert len(client.prompts) == 1


def test_benchmark_data_adds_reason_and_logs_when_missing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    constraint = _AsyncConstraint()
    benchmark = _make_benchmark([constraint])

    with caplog.at_level(logging.INFO):
        evaluations, _ = asyncio.run(
            benchmark._evaluate_constraints_with_details("value")  # noqa: SLF001
        )

    assert evaluations[0][0] is False
    assert evaluations[0][1] == "[_AsyncConstraint] Constraint evaluation failed."
    assert "[_AsyncConstraint] Constraint evaluation failed." in caplog.text


def test_benchmark_data_rewrite_includes_failure_reasons() -> None:
    constraint = _ReasonTokenConstraint("A", "Reason for missing A")
    benchmark = _make_benchmark([constraint])
    client = _RewriteClient(["A"])

    value, evaluation = asyncio.run(benchmark.rewrite("", cast("LLMClient", client)))

    assert value == "A"
    assert evaluation["_ReasonTokenConstraint"] is True
    prompt = client.prompts[0]
    assert "- [_ReasonTokenConstraint] Aを含めてください。" in prompt
    assert "理由: Reason for missing A" in prompt


def test_benchmark_data_rewrite_falls_back_to_instruction_when_missing_rewrite() -> None:
    constraints: list[Constraint] = [
        _TokenConstraint("A"),
        _NoRewriteInstructionsConstraint(),
    ]
    benchmark = _make_benchmark(constraints)
    client = _RewriteClient(["A"])

    value, evaluation = asyncio.run(benchmark.rewrite("", cast("LLMClient", client)))

    assert value == "A"
    assert all(evaluation.values())
    assert "Fallback instruction" in client.prompts[0]


def test_get_constraint_collections_includes_training_sets(true_client: "LLMClient") -> None:
    collections = get_constraint_collections("train")

    rule_instance = None
    for factory in collections.rule_based:
        candidate = factory(seed=0, document="sample")
        if isinstance(candidate, WordsLengthConstraint):
            rule_instance = candidate
            break
    llm_instance = None
    for factory in collections.llm_based:
        candidate = factory(true_client, "sample", seed=0)
        if isinstance(candidate, PlainStyleConstraint):
            llm_instance = candidate
            break

    assert isinstance(rule_instance, WordsLengthConstraint)
    assert isinstance(llm_instance, PlainStyleConstraint)


def test_get_constraint_collections_includes_ifbench_test_sets(true_client: "LLMClient") -> None:
    collections = get_constraint_collections("test")

    rule_instance = None
    training_rule_instance = None
    for factory in collections.rule_based:
        candidate = factory(seed=0, document="sample")
        if isinstance(candidate, ConjunctionCountIfbenchConstraint):
            rule_instance = candidate
        if isinstance(candidate, WordsLengthConstraint):
            training_rule_instance = candidate

    assert isinstance(rule_instance, ConjunctionCountIfbenchConstraint)
    assert isinstance(training_rule_instance, WordsLengthConstraint)

    llm_instance = None
    for factory in collections.llm_based:
        candidate = factory(true_client, "sample", seed=0)
        if isinstance(candidate, PlainStyleConstraint):
            llm_instance = candidate
            break

    assert isinstance(llm_instance, PlainStyleConstraint)


def test_get_constraint_collections_rejects_unknown_set() -> None:
    with pytest.raises(ValueError):
        get_constraint_collections(cast("ConstraintSetName", "unknown"))


def test_rule_constraint_factory_exposes_kwargs() -> None:
    collections = get_constraint_collections("train")
    factory = next(
        factory
        for factory in collections.length
        if isinstance(factory(seed=0, document="sample"), WordsLengthConstraint)
    )

    assert getattr(factory, "kwargs", {}) == {"minimum": 10, "maximum": 20}


class _DeterministicRng:
    def __init__(self, choice_indexes: list[int]) -> None:
        self._choice_indexes = choice_indexes
        self._idx = 0

    def choice(self, seq: Sequence[Any], p: Any | None = None) -> Any:
        if not seq:
            raise ValueError("Sequence cannot be empty.")
        if self._idx < len(self._choice_indexes):
            index = self._choice_indexes[self._idx]
        else:
            index = 0
        self._idx += 1
        return seq[index % len(seq)]

    def integers(self, low: int, high: int | None = None, endpoint: bool = False) -> int:
        return int(low)

    def shuffle(self, seq: Sequence[Any]) -> None:
        return None


class _ProgressRecorder:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs
        self.updates: list[int] = []
        self.closed = False

    def update(self, value: int) -> None:
        self.updates.append(value)

    def close(self) -> None:
        self.closed = True


def test_get_benchmark_data_with_single_constraint_shuffles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts = [_NamedPrompt("first"), _NamedPrompt("second")]
    constraint_collections = ConstraintCollections(
        character=[lambda seed=None, document=None: _CharacterConstraint()],
        rule_content=[],
        llm_content=[],
        format=[lambda seed=None, document=None: _FormatConstraint()],
        length=[],
        logic=[],
        meta_output=[],
        notation=[],
        rule_processing=[],
        llm_processing=[],
        structure=[],
        rule_style=[],
        llm_style=[],
        ifbench_count=[],
        ifbench_format=[],
        ifbench_ratio=[],
        ifbench_repeat=[],
        ifbench_sentence=[],
        ifbench_words=[],
    )

    class _ShuffleRng:
        def shuffle(self, seq: list[Any]) -> None:
            seq.reverse()

    seed_values: list[int] = []

    def _rng_factory(seed: int) -> _ShuffleRng:
        seed_values.append(seed)
        return _ShuffleRng()

    monkeypatch.setattr(
        "jfbench.benchmark.build.get_constraint_collections",
        lambda constraint_set: constraint_collections,
    )
    monkeypatch.setattr(
        "jfbench.benchmark.build.np.random.default_rng",
        _rng_factory,
    )

    benchmark_data = build.get_benchmark_data_with_single_constraint(
        cast("LLMClient", object()),
        prompts,
        prompt_source="test_source",
        seed=123,
        constraint_set="train",
    )

    assert seed_values == [123]
    data_ids = [data.meta_data.data_id for data in benchmark_data]
    assert data_ids == [
        "1-_FormatConstraint",
        "1-_CharacterConstraint",
        "0-_FormatConstraint",
        "0-_CharacterConstraint",
    ]


def test_get_benchmark_data_with_multiple_constraints_skips_competitives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = _StubPrompt()
    constraint_collections = ConstraintCollections(
        character=[lambda seed=None, document=None: _CharacterConstraint()],
        rule_content=[lambda seed=None, document=None: _ConflictingConstraint()],
        llm_content=[],
        format=[lambda seed=None, document=None: _FormatConstraint()],
        length=[],
        logic=[],
        meta_output=[],
        notation=[],
        rule_processing=[],
        llm_processing=[],
        structure=[],
        rule_style=[lambda seed=None, document=None: _SafeConstraint()],
        llm_style=[],
        ifbench_count=[],
        ifbench_format=[],
        ifbench_ratio=[],
        ifbench_repeat=[],
        ifbench_sentence=[],
        ifbench_words=[],
    )
    monkeypatch.setitem(
        build.CONSTRAINT_COLLECTIONS,
        "custom",
        constraint_collections,
    )
    monkeypatch.setattr(
        "jfbench.benchmark.build.np.random.default_rng",
        lambda seed: _DeterministicRng([0, 0, 1, 3]),
    )

    benchmark_data = get_benchmark_data_with_multiple_constraints(
        cast("LLMClient", object()),
        [prompt],
        n_constraints=3,
        n_benchmark_data=1,
        seed=0,
        prompt_source="test_source",
        constraint_set=cast("ConstraintSetName", "custom"),
    )

    assert len(benchmark_data) == 1
    constraint_names = {
        constraint.__class__.__name__ for constraint in benchmark_data[0].constraints
    }
    assert "_ConflictingConstraint" not in constraint_names
    assert "_SafeConstraint" in constraint_names


def test_get_benchmark_data_with_multiple_constraints_does_not_force_character(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = _StubPrompt()
    constraint_collections = ConstraintCollections(
        character=[lambda seed=None, document=None: _CharacterConstraint()],
        rule_content=[lambda seed=None, document=None: _ContentConstraint()],
        llm_content=[],
        format=[lambda seed=None, document=None: _FormatConstraint()],
        length=[],
        logic=[],
        meta_output=[],
        notation=[],
        rule_processing=[],
        llm_processing=[],
        structure=[],
        rule_style=[],
        llm_style=[],
        ifbench_count=[],
        ifbench_format=[],
        ifbench_ratio=[],
        ifbench_repeat=[],
        ifbench_sentence=[],
        ifbench_words=[],
    )
    monkeypatch.setitem(
        build.CONSTRAINT_COLLECTIONS,
        "no_character_bias",
        constraint_collections,
    )
    monkeypatch.setattr(
        "jfbench.benchmark.build.np.random.default_rng",
        lambda seed: _DeterministicRng([0, 0, 1]),
    )

    benchmark_data = get_benchmark_data_with_multiple_constraints(
        cast("LLMClient", object()),
        [prompt],
        n_constraints=2,
        n_benchmark_data=1,
        seed=0,
        prompt_source="test_source",
        constraint_set=cast("ConstraintSetName", "no_character_bias"),
    )

    constraint_names = {
        constraint.__class__.__name__ for constraint in benchmark_data[0].constraints
    }
    assert "_CharacterConstraint" not in constraint_names
    assert "_ContentConstraint" in constraint_names


def test_get_benchmark_data_with_multiple_constraints_allows_same_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = _StubPrompt()
    constraint_collections = ConstraintCollections(
        character=[],
        rule_content=[],
        llm_content=[],
        format=[
            lambda seed=None, document=None: _FormatConstraint(),
            lambda seed=None, document=None: _AlternateFormatConstraint(),
        ],
        length=[],
        logic=[],
        meta_output=[],
        notation=[],
        rule_processing=[],
        llm_processing=[],
        structure=[],
        rule_style=[],
        llm_style=[],
        ifbench_count=[],
        ifbench_format=[],
        ifbench_ratio=[],
        ifbench_repeat=[],
        ifbench_sentence=[],
        ifbench_words=[],
    )
    monkeypatch.setitem(
        build.CONSTRAINT_COLLECTIONS,
        "same_group",
        constraint_collections,
    )
    monkeypatch.setattr(
        "jfbench.benchmark.build.np.random.default_rng",
        lambda seed: _DeterministicRng([0, 0, 1]),
    )

    benchmark_data = get_benchmark_data_with_multiple_constraints(
        cast("LLMClient", object()),
        [prompt],
        n_constraints=2,
        n_benchmark_data=1,
        seed=0,
        prompt_source="test_source",
        constraint_set=cast("ConstraintSetName", "same_group"),
    )

    constraint_names = {
        constraint.__class__.__name__ for constraint in benchmark_data[0].constraints
    }
    assert constraint_names == {"_AlternateFormatConstraint", "_FormatConstraint"}


def test_get_benchmark_data_with_multiple_constraints_updates_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts = [_NamedPrompt("first"), _NamedPrompt("second")]

    class _PromptCyclingRng(_DeterministicRng):
        def __init__(self, prompt_seq: Sequence[Any]) -> None:
            super().__init__([0, 0, 0, 0, 0, 0])
            self._prompt_seq = prompt_seq
            self._prompt_calls = 0

        def choice(self, seq: Sequence[Any], p: Any | None = None) -> Any:
            if seq is self._prompt_seq:
                prompt = seq[self._prompt_calls % len(seq)]
                self._prompt_calls += 1
                return prompt
            return super().choice(seq, p=p)

    constraint_collections = ConstraintCollections(
        character=[lambda seed=None, document=None: _CharacterConstraint()],
        rule_content=[],
        llm_content=[],
        format=[lambda seed=None, document=None: _FormatConstraint()],
        length=[],
        logic=[],
        meta_output=[],
        notation=[],
        rule_processing=[],
        llm_processing=[],
        structure=[],
        rule_style=[],
        llm_style=[],
        ifbench_count=[],
        ifbench_format=[],
        ifbench_ratio=[],
        ifbench_repeat=[],
        ifbench_sentence=[],
        ifbench_words=[],
    )
    monkeypatch.setitem(
        build.CONSTRAINT_COLLECTIONS,
        "progress",
        constraint_collections,
    )

    progress_instances: list[_ProgressRecorder] = []

    def _progress_factory(*args: Any, **kwargs: Any) -> _ProgressRecorder:
        progress = _ProgressRecorder(*args, **kwargs)
        progress_instances.append(progress)
        return progress

    monkeypatch.setattr("jfbench.benchmark.build.tqdm", _progress_factory)
    monkeypatch.setattr(
        "jfbench.benchmark.build.np.random.default_rng",
        lambda seed: _PromptCyclingRng(prompts),
    )

    benchmark_data = get_benchmark_data_with_multiple_constraints(
        cast("LLMClient", object()),
        prompts,
        n_constraints=2,
        n_benchmark_data=2,
        seed=0,
        prompt_source="test_source",
        constraint_set=cast("ConstraintSetName", "progress"),
    )

    assert len(benchmark_data) == 2
    assert {data.meta_data.prompt for data in benchmark_data} == {"first", "second"}
    assert len(progress_instances) == 1
    progress = progress_instances[0]
    assert progress.kwargs["total"] == 2
    assert progress.kwargs["desc"] == "Building benchmark data"
    assert progress.kwargs["leave"] is False
    assert progress.updates == [1, 1]
    assert progress.closed


def test_get_benchmark_data_with_multiple_constraints_includes_constraint_names_in_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = _StubPrompt()
    constraint_collections = ConstraintCollections(
        character=[],
        rule_content=[],
        llm_content=[],
        format=[lambda seed=None, document=None: _FormatConstraint()],
        length=[],
        logic=[],
        meta_output=[],
        notation=[],
        rule_processing=[],
        llm_processing=[],
        structure=[],
        rule_style=[lambda seed=None, document=None: _SafeConstraint()],
        llm_style=[],
        ifbench_count=[],
        ifbench_format=[],
        ifbench_ratio=[],
        ifbench_repeat=[],
        ifbench_sentence=[],
        ifbench_words=[],
    )
    monkeypatch.setitem(
        build.CONSTRAINT_COLLECTIONS,
        "id_check",
        constraint_collections,
    )
    monkeypatch.setattr(
        "jfbench.benchmark.build.np.random.default_rng",
        lambda seed: _DeterministicRng([0, 0, 1]),
    )

    benchmark_data = get_benchmark_data_with_multiple_constraints(
        cast("LLMClient", object()),
        [prompt],
        n_constraints=2,
        n_benchmark_data=1,
        seed=0,
        prompt_source="test_source",
        constraint_set=cast("ConstraintSetName", "id_check"),
    )

    assert benchmark_data[0].meta_data.data_id == "multi-_FormatConstraint-_SafeConstraint"


def test_get_benchmark_data_with_multiple_constraints_skips_duplicate_prompts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts = [_NamedPrompt("duplicate"), _NamedPrompt("unique")]
    constraint_collections = ConstraintCollections(
        character=[lambda seed=None, document=None: _CharacterConstraint()],
        rule_content=[],
        llm_content=[],
        format=[lambda seed=None, document=None: _FormatConstraint()],
        length=[],
        logic=[],
        meta_output=[],
        notation=[],
        rule_processing=[],
        llm_processing=[],
        structure=[],
        rule_style=[],
        llm_style=[],
        ifbench_count=[],
        ifbench_format=[],
        ifbench_ratio=[],
        ifbench_repeat=[],
        ifbench_sentence=[],
        ifbench_words=[],
    )
    monkeypatch.setitem(
        build.CONSTRAINT_COLLECTIONS,
        "dup_check",
        constraint_collections,
    )
    monkeypatch.setattr(
        "jfbench.benchmark.build.np.random.default_rng",
        lambda seed: _DeterministicRng([0, 0, 0, 1, 0, 0]),
    )

    benchmark_data = get_benchmark_data_with_multiple_constraints(
        cast("LLMClient", object()),
        prompts,
        n_constraints=2,
        n_benchmark_data=2,
        seed=0,
        prompt_source="test_source",
        constraint_set=cast("ConstraintSetName", "dup_check"),
    )

    prompts_seen = {data.meta_data.prompt for data in benchmark_data}
    assert len(benchmark_data) == 2
    assert prompts_seen == {"duplicate", "unique"}


def test_get_benchmark_data_with_multiple_constraints_skips_order_only_duplicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts = [_OrderSensitivePrompt("base"), _OrderSensitivePrompt("other")]

    class _OrderAwareRng(_DeterministicRng):
        def __init__(self, prompt_seq: Sequence[Any], prompt_indexes: list[int]) -> None:
            super().__init__([0, 0, 0, 0, 0, 0, 0])
            self._prompt_seq = prompt_seq
            self._prompt_indexes = prompt_indexes
            self._prompt_calls = 0
            self.shuffle_calls = 0

        def choice(self, seq: Sequence[Any], p: Any | None = None) -> Any:
            if seq is self._prompt_seq:
                if self._prompt_calls < len(self._prompt_indexes):
                    index = self._prompt_indexes[self._prompt_calls]
                else:
                    index = self._prompt_indexes[-1]
                prompt = seq[index % len(seq)]
                self._prompt_calls += 1
                return prompt
            return super().choice(seq, p=p)

        def shuffle(self, seq: Sequence[Any]) -> None:
            self.shuffle_calls += 1
            if self.shuffle_calls % 2 == 0:
                if isinstance(seq, list):
                    seq.reverse()

    constraint_collections = ConstraintCollections(
        character=[lambda seed=None, document=None: _CharacterConstraint()],
        rule_content=[],
        llm_content=[],
        format=[lambda seed=None, document=None: _FormatConstraint()],
        length=[],
        logic=[],
        meta_output=[],
        notation=[],
        rule_processing=[],
        llm_processing=[],
        structure=[],
        rule_style=[],
        llm_style=[],
        ifbench_count=[],
        ifbench_format=[],
        ifbench_ratio=[],
        ifbench_repeat=[],
        ifbench_sentence=[],
        ifbench_words=[],
    )
    monkeypatch.setitem(
        build.CONSTRAINT_COLLECTIONS,
        "order_check",
        constraint_collections,
    )
    rng = _OrderAwareRng(prompts, [0, 0, 1])
    monkeypatch.setattr(
        "jfbench.benchmark.build.np.random.default_rng",
        lambda seed: rng,
    )

    benchmark_data = get_benchmark_data_with_multiple_constraints(
        cast("LLMClient", object()),
        prompts,
        n_constraints=2,
        n_benchmark_data=2,
        seed=0,
        prompt_source="test_source",
        constraint_set=cast("ConstraintSetName", "order_check"),
    )

    assert len(benchmark_data) == 2
    prompt_texts = {data.meta_data.prompt for data in benchmark_data}
    assert any(text.startswith("base-") for text in prompt_texts)
    assert any(text.startswith("other-") for text in prompt_texts)


def test_get_benchmark_data_with_multiple_constraints_raises_when_all_conflict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompt = _StubPrompt()
    constraint_collections = ConstraintCollections(
        character=[],
        rule_content=[lambda seed=None, document=None: _ConflictingConstraint()],
        llm_content=[],
        format=[lambda seed=None, document=None: _FormatConstraint()],
        length=[],
        logic=[],
        meta_output=[],
        notation=[],
        rule_processing=[],
        llm_processing=[],
        structure=[],
        rule_style=[],
        llm_style=[],
        ifbench_count=[],
        ifbench_format=[],
        ifbench_ratio=[],
        ifbench_repeat=[],
        ifbench_sentence=[],
        ifbench_words=[],
    )
    monkeypatch.setitem(
        build.CONSTRAINT_COLLECTIONS,
        "custom",
        constraint_collections,
    )
    monkeypatch.setattr(
        "jfbench.benchmark.build.np.random.default_rng",
        lambda seed: _DeterministicRng([0, 0, 0, 0, 0]),
    )

    with pytest.raises(ValueError):
        get_benchmark_data_with_multiple_constraints(
            cast("LLMClient", object()),
            [prompt],
            n_constraints=3,
            n_benchmark_data=1,
            seed=0,
            prompt_source="test_source",
            constraint_set=cast("ConstraintSetName", "custom"),
        )
