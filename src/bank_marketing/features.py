"""Feature contract and leakage-safe preprocessing."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


class LeakageError(ValueError):
    """Raised when a feature that is unknown at scoring time sneaks in."""


NUMERIC_FEATURES = [
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
]

CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]


def feature_columns(*, include_duration: bool = False) -> tuple[list[str], list[str]]:
    """Return the numeric and categorical feature lists.

    ``duration`` is opt-in and exists only for the leakage experiment: the model
    that is shipped never sees it.
    """
    numeric = list(NUMERIC_FEATURES)
    if include_duration:
        numeric.append("duration")
    categorical = list(CATEGORICAL_FEATURES)
    columns = numeric + categorical

    if not include_duration:
        leaked = set(columns) & set(config.LEAKY_COLUMNS)
        if leaked:
            raise LeakageError(
                f"features unknown at scoring time are not allowed: {sorted(leaked)}. "
                "Use include_duration=True only inside the leakage experiment."
            )
    if config.TARGET in columns:
        raise LeakageError("the target column must never be used as a feature")
    return numeric, categorical


def build_preprocessor(
    numeric: list[str], categorical: list[str], *, scale_numeric: bool
) -> ColumnTransformer:
    """One-hot encode categoricals and impute; scaling is only needed by linear models."""
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_pipeline(
    estimator, *, include_duration: bool = False, scale_numeric: bool = False
) -> Pipeline:
    """Wrap preprocessing and the estimator so both are fitted inside each fold."""
    numeric, categorical = feature_columns(include_duration=include_duration)
    return Pipeline(
        [
            ("preprocess", build_preprocessor(numeric, categorical, scale_numeric=scale_numeric)),
            ("model", estimator),
        ]
    )
