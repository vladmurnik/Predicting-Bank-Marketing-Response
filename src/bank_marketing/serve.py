"""FastAPI scoring service backed by the exported model bundle."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from . import config


class CustomerRecord(BaseModel):
    """One raw customer row, using the original dataset column names as aliases."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    age: int = Field(ge=18, le=100)
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    campaign: int = Field(ge=0, le=100)
    pdays: int = Field(ge=0, le=999)
    previous: int = Field(ge=0, le=100)
    poutcome: str
    emp_var_rate: float = Field(alias="emp.var.rate")
    cons_price_idx: float = Field(alias="cons.price.idx")
    cons_conf_idx: float = Field(alias="cons.conf.idx")
    euribor3m: float
    nr_employed: float = Field(alias="nr.employed")


class PredictionResponse(BaseModel):
    probability: float
    threshold: float
    prediction: int
    model_version: str


app = FastAPI(title="Bank Marketing Scoring API", version=config.MODEL_VERSION)

_CACHE: dict[str, dict] = {}


def _model_path() -> Path:
    return Path(os.environ.get("BANK_MODEL_PATH", str(config.MODEL_PATH))).resolve()


def get_bundle() -> dict:
    """Load the bundle once per path; tests point BANK_MODEL_PATH at a fixture."""
    path = _model_path()
    key = str(path)
    if key not in _CACHE:
        if not path.exists():
            raise FileNotFoundError(f"model bundle not found: {path}")
        _CACHE[key] = joblib.load(path)
    return _CACHE[key]


@app.get("/health")
def health() -> dict:
    try:
        bundle = get_bundle()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return {"status": "ok", "model_version": bundle["model_version"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(record: CustomerRecord) -> PredictionResponse:
    try:
        bundle = get_bundle()
    except FileNotFoundError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    row = pd.DataFrame([record.model_dump(by_alias=True)])
    missing = [column for column in bundle["feature_columns"] if column not in row.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"missing features: {missing}")

    probability = float(bundle["pipeline"].predict_proba(row[bundle["feature_columns"]])[0, 1])
    threshold = float(bundle["threshold"])
    return PredictionResponse(
        probability=probability,
        threshold=threshold,
        prediction=int(probability >= threshold),
        model_version=str(bundle["model_version"]),
    )
