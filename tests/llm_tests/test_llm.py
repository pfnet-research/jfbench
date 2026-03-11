import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from pytest import MonkeyPatch

from jfbench.llm import extract_reasoning_content
from jfbench.llm import LLMClient


class _OpenRouterStubClient(LLMClient):
    def __init__(self) -> None:
        self.provider = "openrouter"
        self.model = "stub"
        self.extra_body: dict[str, Any] = {}
        self.client = object()
        self.use_tqdm_calls: list[bool] = []
        self.semaphore = None

    async def _ask_openai(
        self,
        prompts: list[str],
        *,
        use_tqdm: bool = False,
    ) -> tuple[list[str], list[Any]]:
        assert prompts == ["prompt"]
        self.use_tqdm_calls.append(use_tqdm)
        return ["answer"], ["detail"]


def test_ask_returns_response_details() -> None:
    client = _OpenRouterStubClient()

    responses, details = client.ask(["prompt"])

    assert responses == ["answer"]
    assert details == ["detail"]
    assert client.use_tqdm_calls == [False]


def test_async_ask_returns_response_details_openrouter() -> None:
    client = _OpenRouterStubClient()

    responses, details = asyncio.run(client.async_ask(["prompt"]))

    assert responses == ["answer"]
    assert details == ["detail"]
    assert client.use_tqdm_calls == [False]


def test_ask_respects_use_tqdm_flag() -> None:
    client = _OpenRouterStubClient()

    client.ask(["prompt"], use_tqdm=True)

    assert client.use_tqdm_calls == [True]


def test_async_ask_respects_use_tqdm_flag_openrouter() -> None:
    client = _OpenRouterStubClient()

    asyncio.run(client.async_ask(["prompt"], use_tqdm=True))

    assert client.use_tqdm_calls == [True]


def test_vllm_client_accepts_custom_base_url(monkeypatch: MonkeyPatch) -> None:
    created: dict[str, Any] = {}

    class _DummyAsyncOpenAI:
        def __init__(self, *, base_url: str, api_key: str, timeout: int | None = None) -> None:
            _ = timeout
            created["base_url"] = base_url
            created["api_key"] = api_key

    monkeypatch.setattr("jfbench.llm.AsyncOpenAI", _DummyAsyncOpenAI)

    client = LLMClient(
        provider="vllm",
        model="remote-model",
        extra_body={"base_url": "http://remote-host:9000/v1", "temperature": 0.65},
    )

    assert created["base_url"] == "http://remote-host:9000/v1"
    assert created["api_key"] == "unsed"
    assert client.temperature == 0.65
    assert client.extra_body == {}


def test_openrouter_client_enables_reasoning_by_default(monkeypatch: MonkeyPatch) -> None:
    class _DummyAsyncOpenAI:
        def __init__(self, *, base_url: str, api_key: str) -> None:
            self.base_url = base_url
            self.api_key = api_key

    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    monkeypatch.setattr("jfbench.llm.AsyncOpenAI", _DummyAsyncOpenAI)

    client = LLMClient(provider="openrouter", model="remote-model")

    assert client.extra_body == {}
    assert client.temperature == 0.1


def test_llm_client_defaults_to_openrouter_gpt_oss_120b(monkeypatch: MonkeyPatch) -> None:
    class _DummyAsyncOpenAI:
        def __init__(self, *, base_url: str, api_key: str) -> None:
            self.base_url = base_url
            self.api_key = api_key

    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    monkeypatch.setattr("jfbench.llm.AsyncOpenAI", _DummyAsyncOpenAI)

    client = LLMClient()

    assert client.provider == "openrouter"
    assert client.model == "openai/gpt-oss-120b"


def test_vllm_client_requires_model_or_env(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.delenv("JFBENCH_LOCAL_MODEL", raising=False)

    with pytest.raises(ValueError, match="Model name is required"):
        _ = LLMClient(provider="vllm")


def test_extract_reasoning_content_supports_vllm_detail() -> None:
    detail = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(reasoning_content="steps"))]
    )

    reasoning = extract_reasoning_content("vllm", detail)

    assert reasoning == "steps"


def test_extract_reasoning_content_supports_openrouter_reasoning_details() -> None:
    detail = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    reasoning=None, reasoning_details=[SimpleNamespace(text="reasoned text")]
                )
            )
        ]
    )

    reasoning = extract_reasoning_content("openrouter", detail)

    assert reasoning == "reasoned text"


def test_llm_client_serializable_dict_round_trip_for_vllm(monkeypatch: MonkeyPatch) -> None:
    class _DummyAsyncOpenAI:
        def __init__(self, *, base_url: str, api_key: str, timeout: int | None = None) -> None:
            self.base_url = base_url
            self.api_key = api_key
            self.timeout = timeout

    monkeypatch.setattr("jfbench.llm.AsyncOpenAI", _DummyAsyncOpenAI)

    client = LLMClient(
        provider="vllm",
        model="model-a",
        extra_body={
            "base_url": "http://localhost:9000/v1",
            "api_key": "unused",
            "temperature": 0.4,
            "max_tokens": 512,
            "stop_token_ids": [1, 2, 3],
            "custom": "x",
        },
    )

    payload = client.to_serializable_dict()
    restored = LLMClient.from_serializable_dict(payload)

    assert restored.provider == "vllm"
    assert restored.model == "model-a"
    assert restored.temperature == 0.4
    assert restored.max_tokens == 512
    assert restored.stop_token_ids == [1, 2, 3]
    assert restored.extra_body == {"custom": "x"}
    assert restored.base_url == "http://localhost:9000/v1"


def test_llm_client_serializable_dict_round_trip_for_openrouter(monkeypatch: MonkeyPatch) -> None:
    class _DummyAsyncOpenAI:
        def __init__(self, *, base_url: str, api_key: str) -> None:
            self.base_url = base_url
            self.api_key = api_key

    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")
    monkeypatch.setattr("jfbench.llm.AsyncOpenAI", _DummyAsyncOpenAI)

    client = LLMClient(
        provider="openrouter",
        model="model-b",
        extra_body={"temperature": 0.2, "max_tokens": 256},
    )
    payload = client.to_serializable_dict()
    restored = LLMClient.from_serializable_dict(payload)

    assert restored.provider == "openrouter"
    assert restored.model == "model-b"
    assert restored.temperature == 0.2
    assert restored.max_tokens == 256
