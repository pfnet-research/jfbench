from importlib import import_module
from types import ModuleType

from .protocol import Constraint
from .protocol import Prompt


_SUBMODULES = {"benchmark", "constraints", "prompts", "visualization"}

__all__ = [
    "Constraint",
    "Prompt",
    "benchmark",
    "constraints",
    "prompts",
    "visualization",
]


def __getattr__(name: str) -> ModuleType:
    if name in _SUBMODULES:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
