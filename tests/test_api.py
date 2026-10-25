"""The HTTP contract is exercised without starting uvicorn."""

from __future__ import annotations

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier

from bank_marketing import config, features
from bank_marketing.serve import app


@pytest.fixture()
def client(tmp_path, monkeypatch, frame_factory):
    frame = frame_factory(80, seed=3, ordered_months=True)
    target = (frame[config.TARGET] == "yes").astype(int)
    inputs = frame.drop(columns=[config.TARGET])

    pipeline = features.make_pipeline(DummyClassifier(strategy="prior"))
    pipeline.fit(inputs, target)

    numeric, categorical = features.feature_columns()
    bundle = {
        "pipeline": pipeline,
        "threshold": 0.5,
        "feature_columns": numeric + categorical,
        "model_version": "test-0.0.1",
    }
    path = tmp_path / "model.joblib"
    joblib.dump(bundle, path)
    monkeypatch.setenv("BANK_MODEL_PATH", str(path))
    return TestClient(app)


def _payload(frame_factory) -> dict:
    frame = frame_factory(1, seed=11)
    row = frame.drop(columns=[config.TARGET, "duration"]).iloc[0]
    return {key: (value.item() if hasattr(value, "item") else value) for key, value in row.items()}


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_version"] == "test-0.0.1"


def test_predict_returns_probability(client, frame_factory):
    response = client.post("/predict", json=_payload(frame_factory))
    assert response.status_code == 200, response.text
    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["threshold"] == 0.5
    assert body["prediction"] in (0, 1)
    assert body["model_version"] == "test-0.0.1"


def test_predict_rejects_leaky_field(client, frame_factory):
    payload = _payload(frame_factory)
    payload["duration"] = 120
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_rejects_incomplete_payload(client):
    response = client.post("/predict", json={"age": 35})
    assert response.status_code == 422


def test_predict_accepts_snake_case_aliases(client, frame_factory):
    payload = _payload(frame_factory)
    payload["emp_var_rate"] = payload.pop("emp.var.rate")
    payload["cons_price_idx"] = payload.pop("cons.price.idx")
    payload["cons_conf_idx"] = payload.pop("cons.conf.idx")
    payload["nr_employed"] = payload.pop("nr.employed")
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text


def test_probability_matches_the_bundle(client, frame_factory):
    payload = _payload(frame_factory)
    expected = 0.0
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert np.isfinite(response.json()["probability"])
    assert response.json()["probability"] >= expected
