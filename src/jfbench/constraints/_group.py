import importlib
import inspect
import json
from pathlib import Path
import random
import secrets
import sys
from typing import Any
from typing import ClassVar
from typing import Sequence

from ._competitives import COMPETITIVE_CONSTRAINTS


class ConstraintGroupMixin:
    """Mixin that derives a constraint group name from the module path."""

    _group_value: ClassVar[str | None] = None

    def __init__(self, *, seed: int | None = None) -> None:
        instruction_seed = seed if seed is not None else secrets.randbits(64)
        self._instruction_rng = random.Random(instruction_seed)
        self._instruction_choice: str | None = None

    @property
    def group(self) -> str:
        cls = self.__class__
        cached = cls._group_value
        if cached is not None:
            return cached

        definition_path = self._resolve_definition_path(cls)
        group = self._extract_group_name(definition_path)
        if group is None:
            group = self._extract_group_from_module(cls.__module__)

        assert group is not None, f"Could not determine constraint group for class {cls.__name__}"
        cls._group_value = group
        return group

    @property
    def competitives(self) -> list[str]:
        constraints = COMPETITIVE_CONSTRAINTS.get(self.__class__.__name__, ())
        return list(constraints)

    @staticmethod
    def _resolve_definition_path(cls: type["ConstraintGroupMixin"]) -> Path | None:
        try:
            return Path(inspect.getfile(cls)).resolve()
        except TypeError:
            module = sys.modules.get(cls.__module__)
            module_path = getattr(module, "__file__", None) if module else None
            return Path(module_path).resolve() if module_path else None

    @staticmethod
    def _extract_group_name(path: Path | None) -> str | None:
        if path is None:
            return None

        parts = path.parts
        for index, part in enumerate(parts):
            if part == "constraints" and index + 1 < len(parts):
                return ConstraintGroupMixin._camelize(parts[index + 1])
        return None

    @staticmethod
    def _camelize(value: str) -> str:
        return "".join(segment.capitalize() for segment in value.split("_") if segment)

    @staticmethod
    def _extract_group_from_module(module_name: str) -> str | None:
        parts = module_name.split(".")
        try:
            idx = parts.index("constraints")
            raw_segment = parts[idx + 1]
        except (ValueError, IndexError):
            if len(parts) >= 2:
                raw_segment = parts[-2]
            elif parts:
                raw_segment = parts[-1]
            else:
                return None
        return ConstraintGroupMixin._camelize(raw_segment)

    def _random_instruction(self, options: Sequence[str]) -> str:
        if not options:
            raise ValueError("At least one instruction option is required.")
        if self._instruction_choice is None:
            self._instruction_choice = self._instruction_rng.choice(list(options))
        return self._instruction_choice

    def to_serializable_kwargs(self) -> dict[str, Any]:
        signature = inspect.signature(self.__class__.__init__)
        kwargs: dict[str, Any] = {}
        for name in signature.parameters:
            if name in {"self", "seed"}:
                continue
            if hasattr(self, name):
                kwargs[name] = self._serialize_value(getattr(self, name))
        return kwargs

    def to_serializable_dict(self) -> dict[str, Any]:
        return {
            "module": self.__class__.__module__,
            "name": self.__class__.__name__,
            "kwargs": self.to_serializable_kwargs(),
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_serializable_dict(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )

    @classmethod
    def from_serializable_dict(cls, payload: dict[str, Any]) -> "ConstraintGroupMixin":
        module_name = str(payload["module"])
        class_name = str(payload["name"])
        kwargs = payload.get("kwargs", {})
        if not isinstance(kwargs, dict):
            raise TypeError("Constraint payload kwargs must be a dict.")

        module = importlib.import_module(module_name)
        target_cls = getattr(module, class_name)
        if not isinstance(target_cls, type) or not issubclass(target_cls, ConstraintGroupMixin):
            raise TypeError(f"{module_name}.{class_name} is not a ConstraintGroupMixin subclass.")
        restored_kwargs = cls._deserialize_value(kwargs)
        if not isinstance(restored_kwargs, dict):
            raise TypeError("Restored constraint kwargs must be a dict.")
        return target_cls(**restored_kwargs)

    @classmethod
    def from_json(cls, payload: str) -> "ConstraintGroupMixin":
        data = json.loads(payload)
        if not isinstance(data, dict):
            raise TypeError("Constraint JSON payload must decode to an object.")
        return cls.from_serializable_dict(data)

    @classmethod
    def _serialize_value(cls, value: Any) -> Any:
        from jfbench.llm import LLMClient

        if isinstance(value, ConstraintGroupMixin):
            return {"__type__": "constraint", "value": value.to_serializable_dict()}
        if isinstance(value, LLMClient):
            return {"__type__": "llm_client", "value": value.to_serializable_dict()}
        if hasattr(value, "to_serializable_dict") and callable(value.to_serializable_dict):
            return {"__type__": "object", "value": value.to_serializable_dict()}
        if isinstance(value, list):
            return [cls._serialize_value(item) for item in value]
        if isinstance(value, tuple):
            return {"__type__": "tuple", "value": [cls._serialize_value(item) for item in value]}
        if isinstance(value, dict):
            return {str(key): cls._serialize_value(item) for key, item in value.items()}
        if isinstance(value, bool | int | float | str) or value is None:
            return value
        return {"__type__": "opaque", "value": repr(value)}

    @classmethod
    def _deserialize_value(cls, value: Any) -> Any:
        from jfbench.llm import LLMClient

        if isinstance(value, list):
            return [cls._deserialize_value(item) for item in value]
        if isinstance(value, dict):
            marker = value.get("__type__")
            if marker == "constraint":
                payload = value.get("value", {})
                if not isinstance(payload, dict):
                    raise TypeError("Invalid serialized constraint payload.")
                return cls.from_serializable_dict(payload)
            if marker == "llm_client":
                payload = value.get("value", {})
                if not isinstance(payload, dict):
                    raise TypeError("Invalid serialized LLM client payload.")
                return LLMClient.from_serializable_dict(payload)
            if marker == "tuple":
                members = value.get("value", [])
                if not isinstance(members, list):
                    raise TypeError("Invalid serialized tuple payload.")
                return tuple(cls._deserialize_value(item) for item in members)
            if marker == "opaque":
                return str(value.get("value", ""))
            return {key: cls._deserialize_value(item) for key, item in value.items()}
        return value
