import asyncio
from dataclasses import dataclass
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from typing import cast
from typing import ClassVar
from typing import TYPE_CHECKING

import pandas as pd
import pytest
from pytest import MonkeyPatch
import zstandard as zstd

from jfbench.benchmark import eval as eval_module
from jfbench.constraints._group import ConstraintGroupMixin
from jfbench.llm import LLMClient


if TYPE_CHECKING:
    from jfbench.benchmark.build import BenchmarkData


_DEFAULT_PROMPT_SOURCE = "test"
_DEFAULT_N_CONSTRAINTS = 1


class _DummyBenchmark:
    def __init__(
        self,
        index: int,
        prompt_source: str = _DEFAULT_PROMPT_SOURCE,
        n_constraints: int = _DEFAULT_N_CONSTRAINTS,
    ) -> None:
        self.index = index
        self.meta_data = SimpleNamespace(
            prompt_source=prompt_source,
            data_id=f"id-{index}",
            n_constraints=n_constraints,
            constraint_types=["TestConstraint"],
            constraint_groups=["Test"],
        )

    def text(self) -> str:
        return f"prompt-{self.index}"

    def evaluate(self, response: str) -> dict[str, int]:
        return {"length": len(response)}


@dataclass
class _StubAsyncClient:
    prompts: list[str]
    use_tqdm_calls: list[bool]

    def __init__(self) -> None:
        self.prompts = []
        self.use_tqdm_calls = []

    async def async_ask(
        self,
        prompts: list[str],
        *,
        use_tqdm: bool = False,
    ) -> tuple[list[str], list[str]]:
        self.prompts.extend(prompts)
        self.use_tqdm_calls.append(use_tqdm)
        responses = [f"response-for-{prompt}" for prompt in prompts]
        details = [f"detail-for-{prompt}" for prompt in prompts]
        return responses, details


class _ReasoningAsyncClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.use_tqdm_calls: list[bool] = []

    async def async_ask(
        self,
        prompts: list[str],
        *,
        use_tqdm: bool = False,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        self.prompts.extend(prompts)
        self.use_tqdm_calls.append(use_tqdm)
        responses = [f"response-for-{prompt}" for prompt in prompts]
        details = [
            {"choices": [{"message": {"reasoning_content": f"reasoning-for-{prompt}"}}]}
            for prompt in prompts
        ]
        return responses, details


class _SampleSerializableConstraint(ConstraintGroupMixin):
    def __init__(self, foo: str, *, document: str = "", seed: int | None = None) -> None:
        super().__init__(seed=seed)
        self.foo = foo
        self.document = document
        self.seed = seed

    def evaluate(self, value: str) -> tuple[bool, None]:
        return self.foo in value, None

    def instructions(self, train_or_test: str = "train") -> str:
        _ = train_or_test
        return f"Include {self.foo}"

    @property
    def group(self) -> str:
        return "Test"

    def rewrite_instructions(self) -> str:
        return f"Include {self.foo}"


class _SerializableLLMConstraint(ConstraintGroupMixin):
    def __init__(
        self,
        client: LLMClient,
        foo: str = "needle",
        *,
        seed: int | None = None,
    ) -> None:
        super().__init__(seed=seed)
        self.client = client
        self.foo = foo

    def evaluate(self, value: str) -> tuple[bool, None]:
        return self.foo in value, None

    def instructions(self, train_or_test: str = "train") -> str:
        _ = train_or_test
        return f"Include {self.foo}"

    @property
    def group(self) -> str:
        return "Test"

    def rewrite_instructions(self) -> str:
        return f"Include {self.foo}"


def _write_jsonl_zst(path: Path, rows: list[dict[str, Any]]) -> None:
    compressor = zstd.ZstdCompressor()
    with path.open("wb") as raw:
        with compressor.stream_writer(raw) as writer:
            with io.TextIOWrapper(writer, encoding="utf-8") as text:
                for row in rows:
                    text.write(json.dumps(row, ensure_ascii=False))
                    text.write("\n")


def test_evaluate_model_generates_and_evaluates_openrouter(tmp_path: Path) -> None:
    results_path = tmp_path / "results.jsonl"
    dataset = [_DummyBenchmark(i) for i in range(3)]
    client = _StubAsyncClient()
    config = eval_module.ModelConfig(
        provider="openrouter",
        model="test-model",
        model_short="TestModel",
    )

    _ = asyncio.run(
        eval_module.evaluate_model(
            cast("list[BenchmarkData]", dataset),
            config=config,
            with_generate=True,
            with_eval=True,
            override=False,
            results_path=results_path,
            client=cast("LLMClient", client),
        )
    )

    assert client.prompts == ["prompt-0", "prompt-1", "prompt-2"]
    assert client.use_tqdm_calls == [True]
    stored = (
        pd.read_json(results_path, lines=True).sort_values("prompt_index").reset_index(drop=True)
    )
    assert stored["prompt_index"].tolist() == [0, 1, 2]
    assert stored["model_short"].unique().tolist() == ["TestModel"]
    assert stored["response_details"].tolist() == [
        "detail-for-prompt-0",
        "detail-for-prompt-1",
        "detail-for-prompt-2",
    ]
    assert stored["reasoning_content"].tolist() == ["", "", ""]
    assert stored["results"].notna().all()


def test_evaluate_model_stores_reasoning_content(tmp_path: Path) -> None:
    results_path = tmp_path / "results.jsonl"
    dataset = [_DummyBenchmark(0)]
    client = _ReasoningAsyncClient()
    config = eval_module.ModelConfig(
        provider="vllm",
        model="test-model",
        model_short="ReasoningModel",
    )

    _ = asyncio.run(
        eval_module.evaluate_model(
            cast("list[BenchmarkData]", dataset),
            config=config,
            with_generate=True,
            with_eval=True,
            override=False,
            results_path=results_path,
            client=cast("LLMClient", client),
        )
    )

    stored = (
        pd.read_json(results_path, lines=True).sort_values("prompt_index").reset_index(drop=True)
    )
    assert stored["reasoning_content"].tolist() == ["reasoning-for-prompt-0"]
    detail = stored.loc[0, "response_details"]
    assert detail["choices"][0]["message"]["reasoning_content"] == "reasoning-for-prompt-0"


def test_evaluate_model_override_forces_regeneration_and_evaluation(tmp_path: Path) -> None:
    results_path = tmp_path / "results.jsonl"
    dataset = [_DummyBenchmark(i) for i in range(2)]
    existing_entries = [
        {
            "model": "test-model",
            "model_short": "TestModel",
            "prompt_index": i,
            "response": f"cached-{i}",
            "results": {"length": len(f"cached-{i}")},
            "data_id": f"id-{i}",
            "response_details": f"cached-detail-{i}",
            "prompt_source": _DEFAULT_PROMPT_SOURCE,
            "n_constraints": _DEFAULT_N_CONSTRAINTS,
        }
        for i in range(2)
    ]
    results_path.write_text(
        pd.DataFrame(existing_entries).to_json(orient="records", lines=True, index=False),
        encoding="utf-8",
    )

    client = _StubAsyncClient()
    config = eval_module.ModelConfig(
        provider="openrouter",
        model="test-model",
        model_short="TestModel",
    )

    _ = asyncio.run(
        eval_module.evaluate_model(
            cast("list[BenchmarkData]", dataset),
            config=config,
            with_generate=True,
            with_eval=True,
            override=True,
            results_path=results_path,
            client=cast("LLMClient", client),
        )
    )

    assert client.prompts == ["prompt-0", "prompt-1"]
    assert client.use_tqdm_calls == [True]
    assert client.use_tqdm_calls == [True]
    stored = (
        pd.read_json(results_path, lines=True).sort_values("prompt_index").reset_index(drop=True)
    )
    assert stored["response"].tolist() == ["response-for-prompt-0", "response-for-prompt-1"]
    assert stored["response_details"].tolist() == [
        "detail-for-prompt-0",
        "detail-for-prompt-1",
    ]
    assert stored["reasoning_content"].tolist() == ["", ""]
    assert stored["results"].notna().all()


def test_evaluate_model_skips_generation_when_complete(tmp_path: Path) -> None:
    results_path = tmp_path / "results.jsonl"
    dataset = [_DummyBenchmark(i) for i in range(2)]
    existing_entries = [
        {
            "model": "test-model",
            "model_short": "TestModel",
            "prompt_index": i,
            "response": f"response-{i}",
            "results": {"length": len(f"response-{i}")},
            "data_id": f"id-{i}",
            "response_details": f"detail-{i}",
            "prompt_source": _DEFAULT_PROMPT_SOURCE,
            "n_constraints": _DEFAULT_N_CONSTRAINTS,
        }
        for i in range(2)
    ]
    results_path.write_text(
        pd.DataFrame(existing_entries).to_json(orient="records", lines=True, index=False),
        encoding="utf-8",
    )

    config = eval_module.ModelConfig(
        provider="openrouter",
        model="test-model",
        model_short="TestModel",
    )

    _ = asyncio.run(
        eval_module.evaluate_model(
            cast("list[BenchmarkData]", dataset),
            config=config,
            with_generate=True,
            with_eval=True,
            override=False,
            results_path=results_path,
            client=None,
        )
    )


def test_evaluate_model_only_evaluates_existing(tmp_path: Path) -> None:
    results_path = tmp_path / "results.jsonl"
    dataset = [_DummyBenchmark(i) for i in range(2)]
    existing_entries = [
        {
            "model": "test-model",
            "model_short": "TestModel",
            "prompt_index": 0,
            "response": "response-for-prompt-0",
            "results": None,
            "data_id": "id-0",
            "response_details": "detail-for-prompt-0",
            "prompt_source": _DEFAULT_PROMPT_SOURCE,
            "n_constraints": _DEFAULT_N_CONSTRAINTS,
        },
        {
            "model": "test-model",
            "model_short": "TestModel",
            "prompt_index": 1,
            "response": "cached",
            "results": {"length": 6},
            "data_id": "id-1",
            "response_details": "detail-for-cached",
            "prompt_source": _DEFAULT_PROMPT_SOURCE,
            "n_constraints": _DEFAULT_N_CONSTRAINTS,
        },
    ]
    results_path.write_text(
        pd.DataFrame(existing_entries).to_json(orient="records", lines=True, index=False),
        encoding="utf-8",
    )

    config = eval_module.ModelConfig(
        provider="openrouter",
        model="test-model",
        model_short="TestModel",
    )

    _ = asyncio.run(
        eval_module.evaluate_model(
            cast("list[BenchmarkData]", dataset),
            config=config,
            with_generate=False,
            with_eval=True,
            override=False,
            results_path=results_path,
            client=None,
        )
    )

    stored = pd.read_json(results_path, lines=True)
    assert stored.loc[stored["prompt_index"] == 0, "results"].notna().all()
    assert stored.loc[stored["prompt_index"] == 0, "response_details"].notna().all()
    assert stored.loc[stored["prompt_index"] == 0, "reasoning_content"].notna().all()


def test_generate_responses_batches_async_requests() -> None:
    class _BatchRecordingClient:
        def __init__(self) -> None:
            self.batches: list[list[str]] = []
            self.use_tqdm_calls: list[bool] = []

        async def async_ask(
            self,
            prompts: list[str],
            *,
            use_tqdm: bool = False,
        ) -> tuple[list[str], list[str]]:
            self.batches.append(list(prompts))
            self.use_tqdm_calls.append(use_tqdm)
            responses = [f"response-for-{prompt}" for prompt in prompts]
            details = [f"detail-for-{prompt}" for prompt in prompts]
            return responses, details

    pending_items: list[tuple[int, _DummyBenchmark]] = [
        (0, _DummyBenchmark(0)),
        (1, _DummyBenchmark(1)),
        (2, _DummyBenchmark(2)),
        (3, _DummyBenchmark(3)),
        (4, _DummyBenchmark(4)),
    ]
    pending_items_typed = cast("list[tuple[int, BenchmarkData]]", pending_items)
    config = eval_module.ModelConfig(
        provider="openrouter",
        model="test-model",
        model_short="TestModel",
    )
    client = _BatchRecordingClient()

    entries = asyncio.run(
        eval_module._generate_responses(  # noqa: SLF001
            cast("LLMClient", client),
            config,
            pending_items_typed,
            n_concurrent_generations=2,
        )
    )

    assert client.batches == [
        ["prompt-0", "prompt-1"],
        ["prompt-2", "prompt-3"],
        ["prompt-4"],
    ]
    assert client.use_tqdm_calls == [True, True, True]
    assert [entry["response"] for entry in entries] == [
        "response-for-prompt-0",
        "response-for-prompt-1",
        "response-for-prompt-2",
        "response-for-prompt-3",
        "response-for-prompt-4",
    ]


def test_generate_does_not_skip_when_model_short_differs(tmp_path: Path) -> None:
    results_path = tmp_path / "results.jsonl"
    dataset = [_DummyBenchmark(0)]
    existing_entries = [
        {
            "model": "test-model",
            "model_short": "OtherModel",
            "prompt_index": 0,
            "response": "precomputed",
            "results": {"length": 10},
            "data_id": "id-0",
            "response_details": "detail-for-precomputed",
            "prompt_source": _DEFAULT_PROMPT_SOURCE,
            "n_constraints": _DEFAULT_N_CONSTRAINTS,
        }
    ]
    results_path.write_text(
        pd.DataFrame(existing_entries).to_json(orient="records", lines=True, index=False),
        encoding="utf-8",
    )

    client = _StubAsyncClient()
    config = eval_module.ModelConfig(
        provider="openrouter",
        model="test-model",
        model_short="TargetModel",
    )

    asyncio.run(
        eval_module.evaluate_model(
            cast("list[BenchmarkData]", dataset),
            config=config,
            with_generate=True,
            with_eval=False,
            override=False,
            results_path=results_path,
            client=cast("LLMClient", client),
        )
    )

    assert client.prompts == ["prompt-0"]
    assert client.use_tqdm_calls == [True]


def test_generate_skips_when_model_short_matches_even_if_model_differs(tmp_path: Path) -> None:
    results_path = tmp_path / "results.jsonl"
    dataset = [_DummyBenchmark(0)]
    existing_entries = [
        {
            "model": "legacy-model",
            "model_short": "TargetModel",
            "prompt_index": 0,
            "response": "cached-response",
            "results": {"length": 12},
            "data_id": "id-0",
            "response_details": "detail",
            "prompt_source": _DEFAULT_PROMPT_SOURCE,
            "n_constraints": _DEFAULT_N_CONSTRAINTS,
        }
    ]
    results_path.write_text(
        pd.DataFrame(existing_entries).to_json(orient="records", lines=True, index=False),
        encoding="utf-8",
    )

    client = _StubAsyncClient()
    config = eval_module.ModelConfig(
        provider="openrouter",
        model="new-model",
        model_short="TargetModel",
    )

    asyncio.run(
        eval_module.evaluate_model(
            cast("list[BenchmarkData]", dataset),
            config=config,
            with_generate=True,
            with_eval=False,
            override=False,
            results_path=results_path,
            client=cast("LLMClient", client),
        )
    )

    assert client.prompts == []
    assert client.use_tqdm_calls == []


def test_evaluate_model_lazily_instantiates_client_for_generation(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    results_path = tmp_path / "results.jsonl"
    dataset = [_DummyBenchmark(i) for i in range(2)]

    class _RecordingClient:
        instances: ClassVar[list["_RecordingClient"]] = []

        def __init__(
            self, provider: str, model: str, extra_body: dict[str, Any] | None = None
        ) -> None:
            self.provider = provider
            self.model = model
            self.extra_body = extra_body
            self.prompts: list[str] = []
            self.use_tqdm_calls: list[bool] = []
            type(self).instances.append(self)

        async def async_ask(
            self,
            prompts: list[str],
            *,
            use_tqdm: bool = False,
        ) -> tuple[list[str], list[str]]:
            self.prompts.extend(prompts)
            self.use_tqdm_calls.append(use_tqdm)
            responses = [f"response-for-{prompt}" for prompt in prompts]
            details = [f"detail-for-{prompt}" for prompt in prompts]
            return responses, details

    monkeypatch.setattr(eval_module, "LLMClient", _RecordingClient)
    config = eval_module.ModelConfig(provider="openrouter", model="auto-model", model_short="Auto")

    asyncio.run(
        eval_module.evaluate_model(
            cast("list[BenchmarkData]", dataset),
            config=config,
            with_generate=True,
            with_eval=False,
            override=False,
            results_path=results_path,
            client=None,
        )
    )

    assert len(_RecordingClient.instances) == 1
    instance = _RecordingClient.instances[0]
    assert instance.prompts == ["prompt-0", "prompt-1"]
    assert instance.use_tqdm_calls == [True]


def test_evaluate_model_skips_instantiation_when_no_pending(
    monkeypatch: MonkeyPatch, tmp_path: Path
) -> None:
    results_path = tmp_path / "results.jsonl"
    dataset = [_DummyBenchmark(i) for i in range(2)]
    existing_entries = [
        {
            "model": "auto-model",
            "model_short": "Auto",
            "prompt_index": i,
            "response": f"cached-{i}",
            "results": {"length": len(f"cached-{i}")},
            "data_id": f"id-{i}",
            "response_details": f"detail-{i}",
            "prompt_source": _DEFAULT_PROMPT_SOURCE,
            "n_constraints": _DEFAULT_N_CONSTRAINTS,
        }
        for i in range(2)
    ]
    results_path.write_text(
        pd.DataFrame(existing_entries).to_json(orient="records", lines=True, index=False),
        encoding="utf-8",
    )

    class _TrackingClient:
        instances: ClassVar[int] = 0

        def __init__(self, *args: object, **kwargs: object) -> None:
            type(self).instances += 1

        async def async_ask(self, *args: object, **kwargs: object) -> tuple[list[str], list[str]]:
            raise AssertionError("async_ask should not be called when no pending prompts.")

    monkeypatch.setattr(eval_module, "LLMClient", _TrackingClient)
    config = eval_module.ModelConfig(provider="openrouter", model="auto-model", model_short="Auto")

    asyncio.run(
        eval_module.evaluate_model(
            cast("list[BenchmarkData]", dataset),
            config=config,
            with_generate=True,
            with_eval=False,
            override=False,
            results_path=results_path,
            client=None,
        )
    )

    assert _TrackingClient.instances == 0


def test_evaluation_only_targets_matching_model_short(tmp_path: Path) -> None:
    results_path = tmp_path / "results.jsonl"
    dataset = [_DummyBenchmark(0)]
    existing_entries = [
        {
            "model": "test-model",
            "model_short": "OtherModel",
            "prompt_index": 0,
            "response": "existing-response",
            "results": None,
            "data_id": "id-0",
            "response_details": "detail-for-existing",
            "prompt_source": _DEFAULT_PROMPT_SOURCE,
            "n_constraints": _DEFAULT_N_CONSTRAINTS,
        }
    ]
    results_path.write_text(
        pd.DataFrame(existing_entries).to_json(orient="records", lines=True, index=False),
        encoding="utf-8",
    )

    config = eval_module.ModelConfig(
        provider="openrouter",
        model="test-model",
        model_short="TargetModel",
    )

    _ = asyncio.run(
        eval_module.evaluate_model(
            cast("list[BenchmarkData]", dataset),
            config=config,
            with_generate=False,
            with_eval=True,
            override=False,
            results_path=results_path,
            client=None,
        )
    )

    stored = pd.read_json(results_path, lines=True)
    assert stored.loc[stored["prompt_index"] == 0, "results"].isna().all()


def test_evaluation_runs_when_model_short_matches(tmp_path: Path) -> None:
    results_path = tmp_path / "results.jsonl"
    dataset = [_DummyBenchmark(0)]
    existing_entries = [
        {
            "model": "legacy-model",
            "model_short": "TargetModel",
            "prompt_index": 0,
            "response": "response-target",
            "results": None,
            "data_id": "id-0",
            "response_details": "details",
            "prompt_source": _DEFAULT_PROMPT_SOURCE,
            "n_constraints": _DEFAULT_N_CONSTRAINTS,
        }
    ]
    results_path.write_text(
        pd.DataFrame(existing_entries).to_json(orient="records", lines=True, index=False),
        encoding="utf-8",
    )

    config = eval_module.ModelConfig(
        provider="openrouter",
        model="new-model",
        model_short="TargetModel",
    )

    _ = asyncio.run(
        eval_module.evaluate_model(
            cast("list[BenchmarkData]", dataset),
            config=config,
            with_generate=False,
            with_eval=True,
            override=False,
            results_path=results_path,
            client=None,
        )
    )

    stored = pd.read_json(results_path, lines=True)
    assert stored.loc[stored["prompt_index"] == 0, "results"].notna().all()


def test_collect_pending_generation_filters_by_prompt_source(tmp_path: Path) -> None:
    results_path = tmp_path / "results.jsonl"
    existing_entries = [
        {
            "model": "test-model",
            "model_short": "TestModel",
            "prompt_index": 0,
            "prompt_source": "other",
            "n_constraints": 1,
            "response": "cached-response",
            "results": None,
        }
    ]
    results_path.write_text(
        pd.DataFrame(existing_entries).to_json(orient="records", lines=True, index=False),
        encoding="utf-8",
    )
    dataset = [
        _DummyBenchmark(0, prompt_source="ifbench"),
        _DummyBenchmark(1, prompt_source="ifbench"),
    ]
    config = eval_module.ModelConfig(
        provider="openrouter",
        model="test-model",
        model_short="TestModel",
    )

    pending = eval_module._collect_pending_generation(  # noqa: SLF001
        config,
        cast("list[BenchmarkData]", dataset),
        override=False,
        results_path=results_path,
    )

    assert [index for index, _ in pending] == [0, 1]


def test_collect_pending_evaluations_skips_other_prompt_sources(tmp_path: Path) -> None:
    results_path = tmp_path / "results.jsonl"
    existing_entries = [
        {
            "model": "test-model",
            "model_short": "TestModel",
            "prompt_index": 0,
            "prompt_source": "ifbench",
            "n_constraints": 1,
            "response": "response-ifbench",
            "results": None,
        },
        {
            "model": "test-model",
            "model_short": "TestModel",
            "prompt_index": 0,
            "prompt_source": "other",
            "n_constraints": 1,
            "response": "response-other",
            "results": None,
        },
    ]
    results_path.write_text(
        pd.DataFrame(existing_entries).to_json(orient="records", lines=True, index=False),
        encoding="utf-8",
    )
    dataset = [
        _DummyBenchmark(0, prompt_source="ifbench"),
        _DummyBenchmark(1, prompt_source="ifbench"),
    ]
    config = eval_module.ModelConfig(
        provider="openrouter",
        model="test-model",
        model_short="TestModel",
    )

    evaluation_items = eval_module._collect_pending_evaluations(  # noqa: SLF001
        config,
        cast("list[BenchmarkData]", dataset),
        override=False,
        results_path=results_path,
    )

    assert len(evaluation_items) == 1
    index, benchmark, response, _ = evaluation_items[0]
    assert index == 0
    assert isinstance(benchmark, _DummyBenchmark)
    assert benchmark.index == 0
    assert response == "response-ifbench"


def test_parse_args_requires_model_specs_json() -> None:
    with pytest.raises(SystemExit):
        eval_module.parse_args([])


def test_parse_args_accepts_custom_flags(tmp_path: Path) -> None:
    output_dir = tmp_path / "custom_results"
    args = eval_module.parse_args(
        [
            "--output-dir",
            str(output_dir),
            "--no-with-generate",
            "--no-with-eval",
            "--override",
            "--n-constraints",
            "3",
            "--n-benchmark-data",
            "15",
            "--seed",
            "99",
            "--n-concurrent-generations",
            "2",
            "--model-specs-json",
            json.dumps([{"provider": "openrouter", "model": "x", "model_short": "x"}]),
            "--judge-model-spec-json",
            json.dumps(
                {"provider": "openrouter", "model": "judge-model", "model_short": "JudgeModel"}
            ),
        ]
    )
    assert args.output_dir == output_dir
    assert args.with_generate is False
    assert args.with_eval is False
    assert args.override is True
    assert args.n_constraints == "3"
    assert args.n_benchmark_data == 15
    assert args.seed == 99
    assert args.n_concurrent_generations == 2
    assert isinstance(args.model_specs_json, str)
    assert isinstance(args.judge_model_spec_json, str)


def test_parse_args_accepts_dataset_path(tmp_path: Path) -> None:
    dataset_path = tmp_path / "golden_reasoning_dataset_postprocessed"
    args = eval_module.parse_args(
        [
            "--dataset-path",
            str(dataset_path),
            "--model-specs-json",
            json.dumps([{"provider": "openrouter", "model": "x", "model_short": "x"}]),
        ]
    )
    assert args.dataset_path == [str(dataset_path)]


def test_parse_args_accepts_multiple_dataset_paths(tmp_path: Path) -> None:
    first = tmp_path / "dataset_a"
    second = tmp_path / "dataset_b"
    args = eval_module.parse_args(
        [
            "--dataset-path",
            str(first),
            "--dataset-path",
            str(second),
            "--model-specs-json",
            json.dumps([{"provider": "openrouter", "model": "x", "model_short": "x"}]),
        ]
    )
    assert args.dataset_path == [str(first), str(second)]


def test_parse_args_supports_multiple_constraint_values() -> None:
    args = eval_module.parse_args(
        [
            "--benchmark",
            "ifbench",
            "--n-constraints",
            "1,3",
            "--model-specs-json",
            json.dumps([{"provider": "openrouter", "model": "x", "model_short": "x"}]),
        ]
    )
    assert args.benchmark == "ifbench"
    assert args.n_constraints == "1,3"


def test_select_judge_config_rejects_array() -> None:
    judge_specs = [
        {
            "provider": "vllm",
            "model": "judge-model",
            "model_short": "JudgeModel",
            "extra_body": {"base_url": "http://localhost:9000/v1"},
        }
    ]
    with pytest.raises(ValueError, match="JSON object"):
        eval_module._select_judge_config(json.dumps(judge_specs))  # noqa: SLF001


def test_normalize_benchmark_list_rejects_unknown_benchmark() -> None:
    with pytest.raises(ValueError, match="Unsupported benchmark"):
        eval_module._normalize_benchmark_list("other")  # noqa: SLF001


def test_parse_args_rejects_removed_favourable_flag() -> None:
    with pytest.raises(SystemExit):
        eval_module.parse_args(["--favourable-for-plamo"])


def test_shuffle_and_slice_dataset_is_deterministic() -> None:
    dataset = [_DummyBenchmark(i) for i in range(10)]

    subset = eval_module._shuffle_and_slice_dataset(dataset, limit=3, seed=123)  # noqa: SLF001
    repeat = eval_module._shuffle_and_slice_dataset(dataset, limit=3, seed=123)  # noqa: SLF001

    assert [item.index for item in subset] == [item.index for item in repeat]
    assert len(subset) == 3


def test_shuffle_and_slice_dataset_returns_full_copy_when_limit_missing() -> None:
    dataset = [_DummyBenchmark(i) for i in range(5)]

    subset = eval_module._shuffle_and_slice_dataset(dataset, limit=None, seed=0)  # noqa: SLF001

    assert [item.index for item in subset] == list(range(5))


def test_main_runs_all_benchmark_and_constraint_combinations(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_combos: list[tuple[str, int]] = []
    recorded_paths: list[Path] = []

    def fake_build_dataset(**kwargs: object) -> list["_DummyBenchmark"]:
        benchmark_name = str(kwargs["benchmark"])
        n_constraints_value = cast("int", kwargs["n_constraints"])
        captured_combos.append((benchmark_name, n_constraints_value))
        return [_DummyBenchmark(0)]

    async def fake_evaluate_model(**kwargs: object) -> object:
        results_path = kwargs["results_path"]
        assert isinstance(results_path, Path)
        recorded_paths.append(results_path)
        client = kwargs["client"]
        if client is None:
            client = object()
        return client

    def fake_select_configs(**kwargs: object) -> list[eval_module.ModelConfig]:
        raise AssertionError("should not select defaults")

    monkeypatch.setattr(eval_module, "_build_dataset", fake_build_dataset)
    monkeypatch.setattr(eval_module, "evaluate_model", fake_evaluate_model)
    asyncio.run(
        eval_module.main(
            benchmark=["ifbench"],
            output_dir=tmp_path,
            with_generate=False,
            with_eval=False,
            override=False,
            n_constraints=[1, 3],
            model_specs_json=json.dumps(
                [
                    {
                        "provider": "openrouter",
                        "model": "dummy",
                        "model_short": "Test Model/Alpha",
                    },
                ]
            ),
        )
    )

    assert sorted(captured_combos) == [
        ("ifbench", 1),
        ("ifbench", 3),
    ]
    expected_names = [
        "ifbench-train-1-Test_Model-Alpha.jsonl",
        "ifbench-train-3-Test_Model-Alpha.jsonl",
    ]
    assert sorted(path.name for path in recorded_paths) == sorted(expected_names)


def test_load_dataset_from_path_reconstructs_constraints(monkeypatch: MonkeyPatch) -> None:
    serialized = _SampleSerializableConstraint(
        foo="needle",
        document="document-text",
        seed=123,
    ).to_json()

    row = {
        "prompt_source": "ifbench",
        "prompt_id": "prompt-0",
        "prompt_document": "document-text",
        "prompt": "prompt-text",
        "response": "old-response",
        "constraint_types": ["_SampleSerializableConstraint"],
        "constraints": [serialized],
        "constraint_instructions": [],
        "seed": 123,
    }
    monkeypatch.setattr("jfbench.benchmark.eval.datasets.load_from_disk", lambda path: [row])

    dataset = eval_module._load_dataset_from_path(  # noqa: SLF001
        dataset_path="dummy-path",
        constraint_set="test",
    )

    assert len(dataset) == 1
    loaded = dataset[0]
    assert loaded.meta_data.prompt_source == "ifbench"
    assert loaded.meta_data.n_constraints == 1
    assert loaded.meta_data.constraint_types == ["_SampleSerializableConstraint"]
    assert loaded.meta_data.constraint_groups == ["Test"]
    assert loaded.meta_data.prompt == "prompt-text"
    constraint = loaded.constraints[0]
    assert isinstance(constraint, _SampleSerializableConstraint)
    assert constraint.foo == "needle"
    assert constraint.document == "document-text"
    assert constraint.seed is None


def test_load_dataset_from_path_uses_constraints_payload(monkeypatch: MonkeyPatch) -> None:
    serialized = _SampleSerializableConstraint(foo="from-constraints").to_json()
    row = {
        "prompt_source": "ifbench",
        "prompt_id": "prompt-0",
        "prompt_document": "document-text",
        "prompt": "prompt-text",
        "constraint_types": [],
        "constraints": [serialized],
        "constraint_instructions": ["stub"],
        "seed": 123,
    }
    monkeypatch.setattr("jfbench.benchmark.eval.datasets.load_from_disk", lambda path: [row])

    dataset = eval_module._load_dataset_from_path(  # noqa: SLF001
        dataset_path="dummy-path",
        constraint_set="test",
    )

    assert len(dataset) == 1
    constraint = dataset[0].constraints[0]
    assert isinstance(constraint, _SampleSerializableConstraint)
    assert constraint.foo == "from-constraints"


def test_load_dataset_from_path_overrides_restored_judge_client(
    monkeypatch: MonkeyPatch,
) -> None:
    dataset_client = LLMClient(
        provider="vllm",
        model="dataset-judge",
        extra_body={"base_url": "http://localhost:8009/v1"},
    )
    serialized = _SerializableLLMConstraint(
        client=dataset_client, foo="from-constraints"
    ).to_json()
    row = {
        "prompt_source": "ifbench",
        "prompt_id": "prompt-0",
        "prompt_document": "document-text",
        "prompt": "prompt-text",
        "constraint_types": ["_SerializableLLMConstraint"],
        "constraints": [serialized],
        "constraint_instructions": ["stub"],
        "seed": 123,
    }
    monkeypatch.setattr("jfbench.benchmark.eval.datasets.load_from_disk", lambda path: [row])

    judge_config = eval_module.ModelConfig(
        provider="vllm",
        model="override-judge",
        model_short="OverrideJudge",
        extra_body={"base_url": "http://localhost:8010/v1"},
    )
    dataset = eval_module._load_dataset_from_path(  # noqa: SLF001
        dataset_path="dummy-path",
        constraint_set="test",
        judge_config=judge_config,
    )

    assert len(dataset) == 1
    constraint = dataset[0].constraints[0]
    assert isinstance(constraint, _SerializableLLMConstraint)
    assert constraint.client.model == "override-judge"
    assert constraint.client.base_url == "http://localhost:8010/v1"


def test_load_dataset_from_path_falls_back_for_direct_jsonl_zst_file(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    serialized = _SampleSerializableConstraint(
        foo="needle-zst",
        document="document-zst",
        seed=222,
    ).to_json()
    path = tmp_path / "verified.jsonl.zst"
    _write_jsonl_zst(
        path,
        [
            {
                "prompt_source": "ifbench",
                "data_id": "data-zst-0",
                "prompt_document": "document-zst",
                "prompt": "prompt-zst",
                "constraints": [serialized],
                "reasoning_content": "reasoning-zst",
                "response": "response-zst",
            }
        ],
    )

    monkeypatch.setattr(
        "jfbench.benchmark.eval.datasets.load_from_disk",
        lambda _: [
            {
                "prompt_source": "ifbench",
                "data_id": "fallback-zst-0",
                "prompt_document": "document-zst",
                "prompt": "prompt-zst",
                "constraints": [serialized],
                "constraint_types": ["_SampleSerializableConstraint"],
                "constraint_instructions": [],
            }
        ],
    )
    dataset = eval_module._load_dataset_from_path(  # noqa: SLF001
        dataset_path=str(path),
        constraint_set="test",
    )
    assert len(dataset) == 1
    assert dataset[0].meta_data.data_id == "fallback-zst-0"


def test_load_dataset_from_path_reads_jsonl_zst_directory(tmp_path: Path) -> None:
    serialized = _SampleSerializableConstraint(
        foo="needle-dir-zst",
        document="document-dir-zst",
        seed=333,
    ).to_json()
    dataset_dir = tmp_path / "jfbench_verified_ifbench"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl_zst(
        dataset_dir / "test.jsonl.zst",
        [
            {
                "prompt_source": "ifbench",
                "data_id": "data-dir-zst-0",
                "prompt_document": "document-dir-zst",
                "prompt": "prompt-dir-zst",
                "constraints": [serialized],
                "reasoning_content": "reasoning-dir-zst",
                "response": "response-dir-zst",
            }
        ],
    )

    dataset = eval_module._load_dataset_from_path(  # noqa: SLF001
        dataset_path=str(dataset_dir),
        constraint_set="test",
    )

    assert len(dataset) == 1
    loaded = dataset[0]
    assert loaded.meta_data.prompt_source == "ifbench"
    assert loaded.meta_data.data_id == "data-dir-zst-0"
    assert loaded.meta_data.prompt == "prompt-dir-zst"
    constraint = loaded.constraints[0]
    assert isinstance(constraint, _SampleSerializableConstraint)
    assert constraint.foo == "needle-dir-zst"


def test_main_uses_loaded_dataset_when_dataset_path_is_set(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded = [
        _DummyBenchmark(0, prompt_source="ifbench", n_constraints=1),
        _DummyBenchmark(1, prompt_source="ifbench", n_constraints=1),
        _DummyBenchmark(2, prompt_source="ifbench", n_constraints=2),
        _DummyBenchmark(3, prompt_source="other", n_constraints=1),
    ]
    load_calls: list[str] = []
    evaluated: list[tuple[str, int, int]] = []

    def fake_load_dataset_from_path(**kwargs: object) -> list["_DummyBenchmark"]:
        path = kwargs.get("dataset_path")
        assert isinstance(path, str)
        load_calls.append(path)
        return loaded

    async def fake_evaluate_model(
        **kwargs: object,
    ) -> object | None:
        dataset = cast("list[_DummyBenchmark]", kwargs["dataset"])
        if dataset:
            source = str(dataset[0].meta_data.prompt_source)
            n_constraints = int(dataset[0].meta_data.n_constraints)
        else:
            source = ""
            n_constraints = 0
        evaluated.append((source, n_constraints, len(dataset)))
        return kwargs.get("client")

    monkeypatch.setattr(eval_module, "_load_dataset_from_path", fake_load_dataset_from_path)
    monkeypatch.setattr(
        eval_module,
        "_build_dataset",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not build dataset")),
    )
    monkeypatch.setattr(eval_module, "evaluate_model", fake_evaluate_model)

    asyncio.run(
        eval_module.main(
            benchmark="ifbench",
            output_dir=tmp_path,
            with_generate=False,
            with_eval=False,
            n_constraints=[1, 2],
            n_benchmark_data=1,
            seed=7,
            dataset_path="dummy-path",
            model_specs_json=json.dumps(
                [{"provider": "openrouter", "model": "x", "model_short": "x"}]
            ),
        )
    )

    assert load_calls == ["dummy-path"]
    assert sorted(evaluated) == [("ifbench", 1, 1), ("ifbench", 2, 1)]


def test_main_uses_loaded_dataset_when_multiple_dataset_paths_are_set(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
) -> None:
    loaded_by_path = {
        "dummy-path-a": [_DummyBenchmark(0, prompt_source="ifbench", n_constraints=1)],
        "dummy-path-b": [_DummyBenchmark(1, prompt_source="ifbench", n_constraints=1)],
    }
    load_calls: list[str] = []
    evaluated_sizes: list[int] = []

    def fake_load_dataset_from_path(**kwargs: object) -> list["_DummyBenchmark"]:
        path = kwargs.get("dataset_path")
        assert isinstance(path, str)
        load_calls.append(path)
        return loaded_by_path[path]

    async def fake_evaluate_model(
        **kwargs: object,
    ) -> object | None:
        dataset = cast("list[_DummyBenchmark]", kwargs["dataset"])
        evaluated_sizes.append(len(dataset))
        return kwargs.get("client")

    monkeypatch.setattr(eval_module, "_load_dataset_from_path", fake_load_dataset_from_path)
    monkeypatch.setattr(
        eval_module,
        "_build_dataset",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("must not build dataset")),
    )
    monkeypatch.setattr(eval_module, "evaluate_model", fake_evaluate_model)

    asyncio.run(
        eval_module.main(
            benchmark="ifbench",
            output_dir=tmp_path,
            with_generate=False,
            with_eval=False,
            n_constraints=[1],
            dataset_path=["dummy-path-a", "dummy-path-b"],
            model_specs_json=json.dumps(
                [{"provider": "openrouter", "model": "x", "model_short": "x"}]
            ),
        )
    )

    assert load_calls == ["dummy-path-a", "dummy-path-b"]
    assert evaluated_sizes == [2]


def test_load_dataset_from_path_raises_on_missing_required_kwargs(
    monkeypatch: MonkeyPatch,
) -> None:
    row = {
        "prompt_source": "ifbench",
        "prompt_id": "prompt-0",
        "prompt_document": "doc-text",
        "prompt": "prompt-text",
        "constraint_types": ["AbstractExcludedContentConstraint"],
        "constraints": [{"name": "AbstractExcludedContentConstraint"}],
        "constraint_instructions": ["内容「日本に関する説明」に触れないでください。"],
        "seed": 123,
    }
    monkeypatch.setattr("jfbench.benchmark.eval.datasets.load_from_disk", lambda path: [row])

    with pytest.raises(ValueError):
        eval_module._load_dataset_from_path(  # noqa: SLF001
            dataset_path="dummy-path",
            constraint_set="test",
        )


def test_main_uses_custom_model_specs(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    specs = [
        {"provider": "openrouter", "model": "custom-model", "model_short": "CustomModel"},
    ]
    recorded: list[eval_module.ModelConfig] = []

    monkeypatch.setattr(eval_module, "_build_dataset", lambda **kwargs: [_DummyBenchmark(0)])

    async def fake_evaluate_model(
        **kwargs: object,
    ) -> object | None:
        config = kwargs["config"]
        assert isinstance(config, eval_module.ModelConfig)
        recorded.append(config)
        return kwargs.get("client")

    monkeypatch.setattr(eval_module, "evaluate_model", fake_evaluate_model)

    asyncio.run(
        eval_module.main(
            benchmark="ifbench",
            output_dir=tmp_path,
            with_generate=False,
            with_eval=False,
            model_specs_json=json.dumps(specs),
        )
    )

    assert len(recorded) == 1
    assert recorded[0].model == "custom-model"
    assert recorded[0].model_short == "CustomModel"


def test_main_passes_judge_model_specs(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    judge_specs = {"provider": "vllm", "model": "judge-model", "model_short": "JudgeModel"}
    captured: list[eval_module.ModelConfig | None] = []

    def fake_build_dataset(**kwargs: object) -> list["_DummyBenchmark"]:
        captured.append(cast("eval_module.ModelConfig | None", kwargs.get("judge_config")))
        return [_DummyBenchmark(0)]

    async def fake_evaluate_model(
        **kwargs: object,
    ) -> object | None:
        return kwargs.get("client")

    monkeypatch.setattr(eval_module, "_build_dataset", fake_build_dataset)
    monkeypatch.setattr(eval_module, "evaluate_model", fake_evaluate_model)

    asyncio.run(
        eval_module.main(
            benchmark="ifbench",
            output_dir=tmp_path,
            with_generate=False,
            with_eval=False,
            model_specs_json=json.dumps(
                [{"provider": "openrouter", "model": "x", "model_short": "x"}]
            ),
            judge_model_spec_json=json.dumps(judge_specs),
        )
    )

    assert len(captured) == 1
    assert captured[0] is not None
    assert captured[0].model == "judge-model"


def test_parse_model_specs_json_validates_input() -> None:
    specs = [
        {"provider": "openrouter", "model": "custom", "model_short": "Custom"},
    ]
    parsed = eval_module._parse_model_specs_json(json.dumps(specs))  # noqa: SLF001
    assert parsed == specs

    with pytest.raises(ValueError):
        eval_module._parse_model_specs_json("{}")  # noqa: SLF001
    with pytest.raises(ValueError):
        eval_module._parse_model_specs_json("[]")  # noqa: SLF001


def test_empty_torch_cuda_cache_noop_without_torch(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(eval_module, "torch", None, raising=False)

    eval_module._empty_torch_cuda_cache()  # noqa: SLF001


def test_empty_torch_cuda_cache_calls_empty_cache(monkeypatch: MonkeyPatch) -> None:
    class _CudaStub:
        def __init__(self) -> None:
            self.called = False

        def is_available(self) -> bool:
            return True

        def empty_cache(self) -> None:
            self.called = True

    cuda_stub = _CudaStub()
    monkeypatch.setattr(eval_module, "torch", SimpleNamespace(cuda=cuda_stub), raising=False)

    eval_module._empty_torch_cuda_cache()  # noqa: SLF001

    assert cuda_stub.called is True
