"""Preprocessing lives inside the pipeline and tolerates categories unseen in training."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from bank_marketing import config, features


def test_pipeline_is_unfitted_before_fit():
    pipeline = features.make_pipeline(LogisticRegression(max_iter=100), scale_numeric=True)
    assert not hasattr(pipeline, "n_features_in_")
    assert not hasattr(pipeline.named_steps["preprocess"], "transformers_")
    assert not hasattr(pipeline.named_steps["model"], "coef_")


def test_unseen_category_is_ignored(frame_factory):
    frame = frame_factory(120, ordered_months=True)
    target = (frame[config.TARGET] == "yes").astype(int)
    inputs = frame.drop(columns=[config.TARGET])

    pipeline = features.make_pipeline(LogisticRegression(max_iter=500), scale_numeric=True)
    pipeline.fit(inputs.iloc[:90], target.iloc[:90])

    row = inputs.iloc[[90]].copy()
    row["job"] = "astronaut"
    row["month"] = "smarch"
    proba = pipeline.predict_proba(row)
    assert proba.shape == (1, 2)
    assert 0.0 <= float(proba[0, 1]) <= 1.0


def test_feature_columns_exclude_target_and_duration():
    numeric, categorical = features.feature_columns()
    assert config.TARGET not in numeric + categorical
    assert "duration" not in numeric + categorical
    assert len(numeric) == 9
    assert len(categorical) == 10
