"""Shared synthetic fixtures: tests must never depend on the downloaded dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

MONTHS = ["may", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "mar", "apr"]


def make_frame(n: int = 40, seed: int = 0, *, ordered_months: bool = False) -> pd.DataFrame:
    """Build a frame with the exact schema of the raw dataset."""
    rng = np.random.default_rng(seed)
    if ordered_months:
        block = max(1, n // 10)
        months = [MONTHS[(index // block) % len(MONTHS)] for index in range(n)]
    else:
        months = list(rng.choice(MONTHS, n))

    frame = pd.DataFrame(
        {
            "age": rng.integers(20, 70, n),
            "job": rng.choice(["admin.", "technician", "blue-collar", "unknown"], n),
            "marital": rng.choice(["married", "single", "divorced"], n),
            "education": rng.choice(["basic.4y", "high.school", "university.degree", "unknown"], n),
            "default": rng.choice(["no", "unknown"], n),
            "housing": rng.choice(["yes", "no", "unknown"], n),
            "loan": rng.choice(["yes", "no", "unknown"], n),
            "contact": rng.choice(["cellular", "telephone"], n),
            "month": months,
            "day_of_week": rng.choice(["mon", "tue", "wed", "thu", "fri"], n),
            "duration": rng.integers(0, 600, n),
            "campaign": rng.integers(1, 10, n),
            "pdays": rng.choice([999, 6], n),
            "previous": rng.integers(0, 3, n),
            "poutcome": rng.choice(["nonexistent", "failure", "success"], n),
            "emp.var.rate": rng.normal(1.0, 0.5, n),
            "cons.price.idx": rng.normal(93.5, 0.5, n),
            "cons.conf.idx": rng.normal(-40, 5, n),
            "euribor3m": rng.normal(3.5, 1.0, n),
            "nr.employed": rng.normal(5200, 50, n),
        }
    )
    frame["y"] = np.where(rng.random(n) < 0.2, "yes", "no")
    return frame


@pytest.fixture
def frame_factory():
    return make_frame


@pytest.fixture
def frame() -> pd.DataFrame:
    return make_frame(80, seed=1, ordered_months=True)
