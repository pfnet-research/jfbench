from __future__ import annotations

from typing import Any
from typing import Sequence
from typing import TYPE_CHECKING

from datasets import Dataset

from jfbench.dpo_dataset.postprocess import postprocess_datasets


if TYPE_CHECKING:
    from pathlib import Path


def _build_row(
    *,
    token_length: int,
    chosen_delta_logp: float,
    rejected_delta_logp: float,
    delta_margin: float,
    individual_sum_diff: float,
    chat_template_diff: float,
    pass_count_diff: float,
) -> dict[str, Any]:
    return {
        "chosen_token_length": token_length,
        "rejected_token_length": token_length,
        "chosen_reasoning_token_length": token_length,
        "rejected_reasoning_token_length": token_length,
        "sum_of_chosen_and_chosen_reasoning_reward": 10.0 + individual_sum_diff,
        "sum_of_rejected_and_rejected_reasoning_reward": 10.0,
        "concatenated_chosen_and_chosen_reasoning_reward": 20.0 + chat_template_diff,
        "concatenated_rejected_and_rejected_reasoning_reward": 20.0,
        "chosen_delta_logp": chosen_delta_logp,
        "rejected_delta_logp": rejected_delta_logp,
        "delta_margin": delta_margin,
        "chosen_all_constraints_pass_count": 20.0 + pass_count_diff,
        "rejected_al_constraints_pass_count": 20.0,
    }


def _write_dataset(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    Dataset.from_list(list(rows)).save_to_disk(str(path))


def _load_dataset(path: Path) -> Dataset:
    return Dataset.load_from_disk(str(path))


def test_postprocess_datasets_applies_all_filters(tmp_path: Path) -> None:
    input_path = tmp_path / "input_dataset"
    output_path = tmp_path / "output_dataset"
    rows: list[dict[str, Any]] = []
    for idx in range(14):
        rows.append(
            _build_row(
                token_length=idx + 1,
                chosen_delta_logp=idx + 1,
                rejected_delta_logp=-1.0,
                delta_margin=2.0,
                individual_sum_diff=1.0,
                chat_template_diff=1.0,
                pass_count_diff=8.0,
            )
        )
    rows[0]["chosen_token_length"] = 13
    rows[0]["rejected_token_length"] = 13
    rows[0]["chosen_reasoning_token_length"] = 13
    rows[0]["rejected_reasoning_token_length"] = 13
    rows[13]["chosen_token_length"] = 14
    rows[13]["rejected_token_length"] = 14
    rows[13]["chosen_reasoning_token_length"] = 14
    rows[13]["rejected_reasoning_token_length"] = 14
    rows[12]["chosen_token_length"] = 12
    rows[12]["rejected_token_length"] = 12
    rows[12]["chosen_reasoning_token_length"] = 12
    rows[12]["rejected_reasoning_token_length"] = 12
    rows[12]["sum_of_chosen_and_chosen_reasoning_reward"] = 9.0
    rows[11]["concatenated_chosen_and_chosen_reasoning_reward"] = 19.0
    rows[10]["rejected_delta_logp"] = 0.5
    rows[9]["delta_margin"] = 0.0
    rows[8]["chosen_all_constraints_pass_count"] = 27.0
    _write_dataset(input_path, rows)

    stats = postprocess_datasets([str(input_path)], output_path=output_path)

    assert stats.total_records == 14
    assert stats.removed_by_token_length_p95 == 2
    assert stats.removed_by_individual_sum_reward_difference == 1
    assert stats.removed_by_chat_template_reward_difference == 1
    assert stats.removed_by_chosen_delta_logp_p50 == 6
    assert stats.removed_by_rejected_delta_logp_non_negative == 1
    assert stats.removed_by_delta_margin_non_positive == 1
    assert stats.removed_by_pass_count_difference_threshold == 1
    assert stats.saved_records == 1

    filtered = _load_dataset(output_path)
    assert len(filtered) == 1
    assert filtered[0]["chosen_delta_logp"] == 8


def test_postprocess_datasets_can_disable_individual_filter(tmp_path: Path) -> None:
    input_path = tmp_path / "input_dataset"
    output_path = tmp_path / "output_dataset"
    rows = [
        _build_row(
            token_length=1,
            chosen_delta_logp=10.0,
            rejected_delta_logp=-1.0,
            delta_margin=1.0,
            individual_sum_diff=-1.0,
            chat_template_diff=1.0,
            pass_count_diff=8.0,
        ),
        _build_row(
            token_length=2,
            chosen_delta_logp=11.0,
            rejected_delta_logp=-1.0,
            delta_margin=1.0,
            individual_sum_diff=1.0,
            chat_template_diff=1.0,
            pass_count_diff=8.0,
        ),
    ]
    _write_dataset(input_path, rows)

    stats = postprocess_datasets(
        [str(input_path)],
        output_path=output_path,
        apply_token_length_p95_filter=False,
        apply_individual_sum_reward_difference_filter=False,
        apply_chat_template_reward_difference_filter=False,
        apply_chosen_delta_logp_p50_filter=False,
        apply_rejected_delta_logp_filter=False,
        apply_delta_margin_filter=False,
        apply_pass_count_difference_filter=False,
    )

    assert stats.removed_by_individual_sum_reward_difference == 0
    assert stats.saved_records == 2
    filtered = _load_dataset(output_path)
    assert len(filtered) == 2
