"""`duration` must never reach the shipped model by accident."""

from __future__ import annotations

import pytest

from bank_marketing import features


def test_default_features_exclude_duration():
    numeric, categorical = features.feature_columns()
    assert "duration" not in numeric + categorical


def test_guard_trips_when_duration_is_added_to_the_contract(monkeypatch):
    monkeypatch.setattr(features, "NUMERIC_FEATURES", [*features.NUMERIC_FEATURES, "duration"])
    with pytest.raises(features.LeakageError, match="duration"):
        features.feature_columns()


def test_duration_requires_explicit_opt_in():
    numeric, _ = features.feature_columns(include_duration=True)
    assert "duration" in numeric


def test_target_cannot_be_a_feature(monkeypatch):
    monkeypatch.setattr(features, "CATEGORICAL_FEATURES", [*features.CATEGORICAL_FEATURES, "y"])
    with pytest.raises(features.LeakageError, match="target"):
        features.feature_columns()
