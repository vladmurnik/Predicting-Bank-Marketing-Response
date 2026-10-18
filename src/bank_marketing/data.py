"""Dataset download, integrity checks and the time-based split."""

from __future__ import annotations

import hashlib
import io
import json
import shutil
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from . import config

REQUIRED_COLUMNS = (
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
    "y",
)


class DataError(ValueError):
    """Raised when the raw dataset does not match the expected schema."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "bank-marketing-case-study/0.1"})
    with urllib.request.urlopen(request, timeout=180) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def _matching_members(names: list[str]) -> list[str]:
    return [name for name in names if name.endswith("bank-additional-full.csv")]


def _extract_csv(archive: Path, destination: Path) -> None:
    """UCI ships a zip inside a zip, so look one level deep before giving up."""
    with zipfile.ZipFile(archive) as outer:
        members = _matching_members(outer.namelist())
        if len(members) == 1:
            with outer.open(members[0]) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            return
        for name in outer.namelist():
            if not name.endswith(".zip"):
                continue
            with zipfile.ZipFile(io.BytesIO(outer.read(name))) as inner:
                inner_members = _matching_members(inner.namelist())
                if len(inner_members) == 1:
                    with inner.open(inner_members[0]) as source, destination.open("wb") as target:
                        shutil.copyfileobj(source, target)
                    return
    raise DataError(f"bank-additional-full.csv not found inside {archive.name}")


def download_raw(*, force: bool = False) -> Path:
    """Download and unpack the raw CSV once; keep it byte-for-byte untouched."""
    config.ensure_dirs()
    if config.RAW_CSV.exists() and not force:
        return config.RAW_CSV

    archive = config.DATA_RAW / "bank+marketing.zip"
    _download(config.DATASET_URL, archive)
    try:
        _extract_csv(archive, config.RAW_CSV)
    finally:
        archive.unlink(missing_ok=True)

    digest = sha256_file(config.RAW_CSV)
    (config.DATA_RAW / "SHA256SUMS.txt").write_text(
        f"{digest}  bank-additional-full.csv\n", encoding="utf-8"
    )
    (config.DATA_RAW / "source.json").write_text(
        json.dumps(
            {
                "url": config.DATASET_URL,
                "page": config.DATASET_PAGE,
                "retrieved_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                "sha256": digest,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return config.RAW_CSV


def validate_schema(frame: pd.DataFrame) -> None:
    """Fail loudly if the CSV is not the dataset this project was built for."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise DataError(f"missing columns: {missing}")
    if frame[config.TARGET].isna().any():
        raise DataError("target column contains missing values")
    unexpected = set(frame[config.TARGET].unique()) - {"yes", "no"}
    if unexpected:
        raise DataError(f"unexpected target values: {sorted(unexpected)}")
    numeric = [
        "age",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed",
    ]
    not_numeric = [column for column in numeric if not pd.api.types.is_numeric_dtype(frame[column])]
    if not_numeric:
        raise DataError(f"columns expected to be numeric: {not_numeric}")


def load_raw() -> pd.DataFrame:
    """Load the raw CSV exactly as shipped (semicolon separated)."""
    if not config.RAW_CSV.exists():
        raise DataError(f"raw dataset is missing: {config.RAW_CSV}. Run `download` first.")
    frame = pd.read_csv(config.RAW_CSV, sep=";")
    validate_schema(frame)
    return frame


def assert_time_ordered(frame: pd.DataFrame, *, max_runs: int = 36) -> None:
    """Guard the assumption behind the time split.

    UCI documents the rows as ordered by campaign date. A shuffled frame would
    show a month change on nearly every row, so counting runs detects it without
    needing a full date column.
    """
    months = frame["month"].astype(str).str.lower()
    runs = int((months != months.shift()).sum())
    if runs > max_runs:
        raise DataError(
            f"month column changes {runs} times (expected <= {max_runs}); "
            "rows do not look chronologically ordered, the time split would leak the future"
        )


def split_by_time(frame: pd.DataFrame, *, test_size: float = config.TEST_SIZE):
    """Split by row order, keeping the newest rows for the final test."""
    cut = int(len(frame) * (1.0 - test_size))
    if cut <= 0 or cut >= len(frame):
        raise DataError("dataset is too small for the requested split")
    assert_time_ordered(frame)
    return frame.iloc[:cut].copy(), frame.iloc[cut:].copy()


def dataset_info(frame: pd.DataFrame) -> dict:
    """Summary facts quoted in the report."""
    unknown_columns = [
        column
        for column in ("job", "marital", "education", "default", "housing", "loan")
        if column in frame.columns
    ]
    return {
        "name": config.DATASET_NAME,
        "url": config.DATASET_PAGE,
        "sha256": sha256_file(config.RAW_CSV),
        "n_rows": int(len(frame)),
        "n_columns": int(frame.shape[1]),
        "positive_rate": float((frame[config.TARGET] == config.POSITIVE_LABEL).mean()),
        "n_duplicate_rows": int(frame.duplicated().sum()),
        "unknown_counts": {
            column: int((frame[column] == "unknown").sum()) for column in unknown_columns
        },
    }
