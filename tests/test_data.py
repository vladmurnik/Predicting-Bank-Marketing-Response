"""The raw dataset must stay intact and the time split must stay honest."""

from __future__ import annotations

import pytest

from bank_marketing import data


def test_split_sizes_and_boundaries(frame_factory):
    frame = frame_factory(100, ordered_months=True)
    train, test = data.split_by_time(frame)
    assert len(train) == 80
    assert len(test) == 20
    assert set(train.index).isdisjoint(test.index)
    assert train.index.max() < test.index.min()


def test_split_rejects_shuffled_rows(frame_factory):
    frame = frame_factory(200, seed=7, ordered_months=False)
    with pytest.raises(data.DataError, match="chronologically"):
        data.split_by_time(frame)


def test_validate_schema_rejects_missing_columns(frame_factory):
    frame = frame_factory(10).drop(columns=["euribor3m"])
    with pytest.raises(data.DataError, match="missing columns"):
        data.validate_schema(frame)


def test_validate_schema_rejects_unknown_target_values(frame_factory):
    frame = frame_factory(10)
    frame["y"] = "maybe"
    with pytest.raises(data.DataError, match="unexpected target values"):
        data.validate_schema(frame)


def test_validate_schema_accepts_the_synthetic_frame(frame_factory):
    data.validate_schema(frame_factory(10))
