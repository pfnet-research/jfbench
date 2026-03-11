from pathlib import Path
import sys
from typing import cast
from typing import TYPE_CHECKING

import pytest

from jfbench.llm import LLMClient


if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from vllm import LLM


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


class StubLLMClient(LLMClient):
    def __init__(self, reply: str) -> None:
        # Avoid calling the base initializer to prevent real API clients.
        self.reply = reply
        self.captured_prompt: str | None = None
        self.client = cast("AsyncOpenAI | LLM", object())
        self.provider = "openrouter"
        self.model = "stub"

    def ask(
        self,
        prompts: list[str],
        *,
        use_tqdm: bool = False,
    ) -> tuple[list[str], list[str]]:
        assert isinstance(prompts, list)
        assert len(prompts) == 1
        self.captured_prompt = prompts[0]
        return [self.reply], [f"detail-for-{self.reply}"]

    async def async_ask(
        self,
        prompts: list[str],
        *,
        use_tqdm: bool = False,
    ) -> tuple[list[str], list[str]]:
        return self.ask(prompts, use_tqdm=use_tqdm)


@pytest.fixture
def true_client() -> StubLLMClient:
    return StubLLMClient("True")


@pytest.fixture
def false_client() -> StubLLMClient:
    return StubLLMClient("False")
