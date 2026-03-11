import asyncio
import logging
import os
from typing import Any
from typing import cast
from typing import Literal
from typing import TYPE_CHECKING

from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion import Choice as ChatChoice
from openai.types.completion import Completion as LegacyCompletion
from openai.types.completion import CompletionChoice as LegacyChoice
from tqdm import tqdm


if TYPE_CHECKING:
    from collections.abc import Sequence

N_PARALLEL_REQUEST = 200
N_RETRIES_PER_REQUEST = 3
DEFAULT_OPENROUTER_MODEL = "openai/gpt-oss-120b"

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        provider: Literal["openrouter", "vllm"] = "openrouter",
        model: str | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        body_params = dict(extra_body) if extra_body is not None else {}
        self.client: AsyncOpenAI
        self.semaphore: asyncio.Semaphore | None = None
        self.base_url: str | None = None
        self.api_key: str | None = None
        self.timeout: int | None = None
        model_name: str | None
        if model is not None:
            model_name = model
        elif provider == "openrouter":
            model_name = DEFAULT_OPENROUTER_MODEL
        else:
            model_name = os.environ.get("JFBENCH_LOCAL_MODEL")
        if not model_name:
            raise ValueError("Model name is required. Set model or JFBENCH_LOCAL_MODEL for vllm.")
        if provider == "openrouter":
            self.client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
            )
            self.semaphore = asyncio.Semaphore(N_PARALLEL_REQUEST)
        elif provider == "vllm":
            base_url = body_params.pop("base_url", "http://localhost:8000/v1")
            api_key = body_params.pop("api_key", "unsed")
            timeout = body_params.pop("timeout", 600)
            self.base_url = str(base_url)
            self.api_key = str(api_key)
            self.timeout = int(timeout)
            self.client = AsyncOpenAI(
                base_url=base_url,
                api_key=api_key,
                timeout=timeout,
            )
            self.semaphore = asyncio.Semaphore(N_PARALLEL_REQUEST)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        self.provider = provider
        self.model = model_name
        self.temperature = body_params.pop("temperature", 0.1)
        self.max_tokens = body_params.pop("max_tokens", 4096)
        self.stop_token_ids = body_params.pop("stop_token_ids", None)
        self.extra_body = body_params

    def to_serializable_dict(self) -> dict[str, Any]:
        serialized_extra_body = dict(self.extra_body)
        serialized_extra_body["temperature"] = self.temperature
        serialized_extra_body["max_tokens"] = self.max_tokens
        if self.stop_token_ids is not None:
            serialized_extra_body["stop_token_ids"] = self.stop_token_ids
        if self.provider == "vllm":
            serialized_extra_body["base_url"] = self.base_url or "http://localhost:8000/v1"
            serialized_extra_body["api_key"] = self.api_key or "unsed"
            serialized_extra_body["timeout"] = self.timeout or 600
        return {
            "provider": self.provider,
            "model": self.model,
            "extra_body": serialized_extra_body,
        }

    @classmethod
    def from_serializable_dict(cls, payload: dict[str, Any]) -> "LLMClient":
        provider = payload.get("provider")
        if provider not in {"openrouter", "vllm"}:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        model = payload.get("model")
        if not isinstance(model, str):
            raise TypeError("LLM client payload model must be a string.")
        extra_body = payload.get("extra_body", {})
        if not isinstance(extra_body, dict):
            raise TypeError("LLM client payload extra_body must be a dict.")
        return cls(
            provider=cast('Literal["openrouter", "vllm"]', provider),
            model=model,
            extra_body=dict(extra_body),
        )

    async def async_ask(
        self,
        prompts: list[str],
        *,
        use_tqdm: bool = False,
    ) -> tuple[list[str], list[Any]]:
        if self.provider == "openrouter" or self.provider == "vllm":
            return await self._ask_openai(prompts, use_tqdm=use_tqdm)
        raise ValueError(f"Unsupported LLM provider for async_ask: {self.provider}")

    def ask(
        self,
        prompts: list[str],
        *,
        use_tqdm: bool = False,
    ) -> tuple[list[str], list[Any]]:
        if self.provider == "openrouter" or self.provider == "vllm":
            return asyncio.run(self._ask_openai(prompts, use_tqdm=use_tqdm))
        raise ValueError(f"Unsupported LLM provider for ask: {self.provider}")

    async def _ask_openai(
        self,
        prompts: list[str],
        *,
        use_tqdm: bool = False,
    ) -> tuple[list[str], list[Any]]:
        assert isinstance(self.client, AsyncOpenAI)
        assert self.semaphore is not None
        semaphore = self.semaphore

        async def _get_answer(index: int, prompt: str) -> tuple[int, str, Any]:
            messages = [{"role": "user", "content": prompt}]
            for i in range(N_RETRIES_PER_REQUEST):
                try:
                    async with semaphore:
                        response = await self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                            n=1,
                            extra_body=self.extra_body,
                        )
                    return index, to_string(response)[0].strip(), response
                except Exception as e:
                    logger.warning(
                        f"Error querying {self.provider} for prompt {index} (attempt {i + 1}): {e}"
                    )
                    if i == N_RETRIES_PER_REQUEST - 1:
                        raise e
            raise RuntimeError(
                f"Failed to obtain response for prompt {index} after {N_RETRIES_PER_REQUEST} retries."
            )

        tasks = [_get_answer(index, prompt) for index, prompt in enumerate(prompts)]
        results: list[str] = [""] * len(prompts)
        response_details: list[Any] = [None] * len(prompts)
        if tasks:
            iterator = asyncio.as_completed(tasks)
            progress = None
            if use_tqdm:
                progress = tqdm(
                    iterator,
                    total=len(tasks),
                    desc=f"Querying {self.provider}",
                    leave=False,
                )
                iterator = progress
            try:
                for coro in iterator:
                    idx, answer, response = await coro
                    results[idx] = answer
                    response_details[idx] = response
            finally:
                if progress is not None:
                    progress.close()
        return results, response_details


def to_string(obj: ChatCompletion | LegacyCompletion) -> list[str]:
    if isinstance(obj, ChatCompletion):
        raw_choices = getattr(obj, "choices", None)
        if not raw_choices:
            return [""]
        choices = cast("Sequence[ChatChoice]", raw_choices)
        return [(choice.message.content or "") for choice in choices]
    if isinstance(obj, LegacyCompletion):
        raw_choices = getattr(obj, "choices", None)
        if not raw_choices:
            return [""]
        choices = cast("Sequence[LegacyChoice]", raw_choices)
        return [(choice.text or "") for choice in choices]
    raise TypeError(f"Unsupported type: {type(obj)}")


def extract_reasoning_content(provider: str, response_detail: Any) -> str:
    if response_detail is None:
        return ""

    def _get_attr(obj: Any, name: str) -> Any:
        if hasattr(obj, name):
            return getattr(obj, name)
        if isinstance(obj, dict):
            return obj.get(name)
        return None

    try:
        choices = _get_attr(response_detail, "choices") or []
        if not choices:
            return ""
        first_choice = choices[0]
        message = _get_attr(first_choice, "message")
        if message is None:
            return ""
        if provider == "vllm":
            reasoning_content = _get_attr(message, "reasoning_content")
            if reasoning_content is not None:
                return str(reasoning_content)
            reasoning_content = _get_attr(_get_attr(message, "model_extra"), "reasoning_content")
            if reasoning_content is not None:
                return str(reasoning_content)
            return ""
        if provider == "openrouter":
            reasoning_content = _get_attr(message, "reasoning")
            if reasoning_content is not None:
                return str(reasoning_content)
            reasoning_details = _get_attr(message, "reasoning_details") or []
            if reasoning_details:
                first_detail = reasoning_details[0]
                return str(_get_attr(first_detail, "text") or "")
    except Exception:
        return ""
    return ""
