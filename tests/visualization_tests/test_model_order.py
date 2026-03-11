from __future__ import annotations

import pandas as pd

from jfbench.visualization.model_order import order_labels
from jfbench.visualization.model_order import preferred_model_labels


def test_order_labels_prefers_requested_order() -> None:
    existing = ["Model-A", "Model-B", "Model-C"]
    preferred = ["Model-B", "Model-A"]

    ordered = order_labels(existing, preferred)

    assert ordered == ["Model-B", "Model-A", "Model-C"]


def test_preferred_model_labels_matches_models() -> None:
    data = pd.DataFrame.from_records(
        [
            {"model": "model-a", "model_short": "A", "model_label": "Alpha"},
            {"model": "model-b", "model_short": "B", "model_label": "Beta"},
        ]
    )

    ordered = preferred_model_labels(data, ["model-b", "model-a"])

    assert ordered == ["Beta", "Alpha"]


def test_preferred_model_labels_keeps_duplicates_and_order() -> None:
    data = pd.DataFrame.from_records(
        [
            {"model": "model-a", "model_short": "A", "model_label": "Alpha"},
            {"model": "model-b", "model_short": "B", "model_label": "Beta"},
        ]
    )

    ordered = preferred_model_labels(data, ["model-b", "model-b", "model-a"])

    assert ordered == ["Beta", "Beta", "Alpha"]


def test_preferred_model_labels_handles_sanitized_names() -> None:
    data = pd.DataFrame.from_records(
        [
            {"model": "model-a", "model_short": "Model A", "model_label": "Alpha"},
            {"model": "model-b", "model_short": "Model B/Prime", "model_label": "Beta"},
        ]
    )

    ordered = preferred_model_labels(data, ["Model_B-Prime", "Model_A"])

    assert ordered == ["Beta", "Alpha"]
