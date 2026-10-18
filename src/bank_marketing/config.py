"""Project-wide constants, paths and business assumptions."""

from __future__ import annotations

from pathlib import Path

# Reproducibility -----------------------------------------------------------
SEED = 42

# Paths ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_CSV = DATA_RAW / "bank-additional-full.csv"
METRICS_JSON = REPORTS_DIR / "metrics.json"
RUN_JSON = REPORTS_DIR / "run.json"
REPORT_MD = REPORTS_DIR / "report.md"
MODEL_PATH = MODELS_DIR / "model.joblib"

# Dataset -------------------------------------------------------------------
DATASET_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/222/bank+marketing"
DATASET_NAME = "UCI Bank Marketing (bank-additional-full)"

TARGET = "y"
POSITIVE_LABEL = "yes"

# `duration` is the length of the last phone call. It is known only after the
# call has happened, so it can never be used to decide whom to call.
LEAKY_COLUMNS = ("duration",)

# Validation ----------------------------------------------------------------
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2  # last fifth of the training window, used for selection
N_SPLITS = 5

# Business framing ----------------------------------------------------------
# One outbound call costs one unit; one successful conversion is worth ten.
CALL_COST = 1.0
CONVERSION_VALUE = 10.0
# The call centre can reach only this share of the base per campaign, so the
# model is evaluated as a ranking policy, not as an absolute probability cut.
CONTACT_BUDGET = 0.30

MODEL_VERSION = "bank-marketing-0.1.0"


def ensure_dirs() -> None:
    """Create every output directory the pipeline writes to."""
    for path in (DATA_RAW, REPORTS_DIR, FIGURES_DIR, MODELS_DIR):
        path.mkdir(parents=True, exist_ok=True)
