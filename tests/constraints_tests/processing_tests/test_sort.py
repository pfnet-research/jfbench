import pytest
from pytest import MonkeyPatch

from jfbench.constraints._group import ConstraintGroupMixin
import jfbench.constraints._utils as utils
from jfbench.constraints.processing import DictionarySortProcessingConstraint
from jfbench.constraints.processing import LengthSortProcessingConstraint
from jfbench.constraints.processing import NumberSortProcessingConstraint


def test_dictionary_sort_processing_constraint_matches_sorted_output() -> None:
    document = "b. a. c."
    constraint = DictionarySortProcessingConstraint(document)
    expected = "[\na.,\nb.,\nc.\n]"
    print(constraint.evaluate(expected))
    assert constraint.evaluate(expected)[0] is True
    assert constraint.evaluate(f"prefix\n{expected}\nsuffix")[0] is True
    assert constraint.evaluate("[]")[0] is False
    assert constraint.evaluate("[\nb.,\na.,\nc.\n]")[0] is False


def test_length_and_number_sort_processing_constraint_sort_order() -> None:
    length_constraint = LengthSortProcessingConstraint("longer sentence. short.")
    number_constraint = NumberSortProcessingConstraint("10 cats. 2 dogs.")

    length_expected = "[\nshort.,\nlonger sentence.\n]"
    number_expected = "[\n2 dogs.,\n10 cats.\n]"

    assert length_constraint.evaluate(length_expected)[0] is True
    assert length_constraint.evaluate(f"meta\n{length_expected}")[0] is True
    assert length_constraint.evaluate("[\nlonger sentence.,\nshort.\n]")[0] is False

    assert number_constraint.evaluate(number_expected)[0] is True
    assert number_constraint.evaluate(f"{number_expected}\nextra context")[0] is True
    assert number_constraint.evaluate("[\n10 cats.,\n2 dogs.\n]")[0] is False


def _capture_options(monkeypatch: MonkeyPatch) -> dict[str, list[str]]:
    captured: dict[str, list[str]] = {}

    def fake_random_instruction(self: ConstraintGroupMixin, options: list[str]) -> str:
        captured["options"] = list(options)
        return options[0]

    monkeypatch.setattr(ConstraintGroupMixin, "_random_instruction", fake_random_instruction)
    return captured


@pytest.mark.parametrize(
    "constraint",
    [
        DictionarySortProcessingConstraint("b. a. c.", seed=0),
        LengthSortProcessingConstraint("b. a. c.", seed=0),
        NumberSortProcessingConstraint("1a. 2b. 3c.", seed=0),
    ],
)
def test_sort_instructions_templates(
    monkeypatch: MonkeyPatch,
    constraint: DictionarySortProcessingConstraint
    | LengthSortProcessingConstraint
    | NumberSortProcessingConstraint,
) -> None:
    captured = _capture_options(monkeypatch)
    instruction = constraint.instructions()
    options = captured["options"]

    assert instruction == options[0]
    assert len(options) == 5
    assert all("<文1>" in option for option in options)
    assert all("<文N>" in option for option in options)
    assert all(("含め" in option) or ("挿入" in option) for option in options)


def test_sort_constraints_use_segmenter(monkeypatch: MonkeyPatch) -> None:
    calls: list[str] = []

    class DummySegmenter:
        def segment(self, text: str) -> list[str]:
            calls.append(text)
            return ["b.", "a."]

    monkeypatch.setattr(utils, "_load_segmenter", lambda: DummySegmenter())

    constraint = DictionarySortProcessingConstraint("b. a.")
    expected_block = "[\na.,\nb.\n]"

    assert constraint.evaluate(expected_block)[0] is True
    assert calls == ["b. a."]
