"""Training orchestration: CV comparison, leakage audit, final fit, artifacts."""

from __future__ import annotations

import json
import platform
import sys
from datetime import UTC, datetime

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit

from . import config, data, evaluate, features


def candidate_estimators() -> dict[str, tuple[object, bool]]:
    """Model zoo and whether the numeric block needs scaling for that model."""
    return {
        "dummy_prior": (DummyClassifier(strategy="prior"), False),
        "logreg": (LogisticRegression(max_iter=2000, random_state=config.SEED), True),
        "hist_gbdt": (
            HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.06,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=config.SEED,
            ),
            False,
        ),
        "lightgbm": (
            LGBMClassifier(
                n_estimators=400,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=40,
                subsample=0.9,
                subsample_freq=1,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=config.SEED,
                n_jobs=4,
                deterministic=True,
                force_col_wise=True,
                verbose=-1,
            ),
            False,
        ),
    }


def cross_validate(
    estimator,
    scale_numeric: bool,
    features_frame: pd.DataFrame,
    target: pd.Series,
    splitter,
    *,
    include_duration: bool = False,
    label: str = "",
) -> dict:
    """Fit the whole pipeline inside every fold and collect fold + OOF predictions."""
    folds: list[dict] = []
    oof_index: list[int] = []
    oof_target: list[int] = []
    oof_proba: list[float] = []

    for number, (train_idx, valid_idx) in enumerate(splitter.split(features_frame, target)):
        pipeline = features.make_pipeline(
            clone(estimator),
            include_duration=include_duration,
            scale_numeric=scale_numeric,
        )
        pipeline.fit(features_frame.iloc[train_idx], target.iloc[train_idx])
        proba = pipeline.predict_proba(features_frame.iloc[valid_idx])[:, 1]
        y_valid = target.iloc[valid_idx].to_numpy()

        folds.append(
            {
                "fold": number,
                "n_train": int(len(train_idx)),
                "n_valid": int(len(valid_idx)),
                "positive_rate": float(y_valid.mean()),
                "pr_auc": float(average_precision_score(y_valid, proba)),
                "roc_auc": float(roc_auc_score(y_valid, proba)),
                "brier": float(brier_score_loss(y_valid, proba)),
            }
        )
        oof_index.extend(int(i) for i in valid_idx)
        oof_target.extend(int(value) for value in y_valid)
        oof_proba.extend(float(value) for value in proba)

    summary = {}
    for key in ("pr_auc", "roc_auc", "brier"):
        values = np.array([fold[key] for fold in folds])
        summary[key] = {"mean": float(values.mean()), "std": float(values.std(ddof=0))}

    return {
        "label": label,
        "folds": folds,
        "summary": summary,
        "oof": {"index": oof_index, "y": oof_target, "proba": oof_proba},
    }


def split_train_validation(train_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Last fifth of the training window, kept for model selection and thresholds."""
    cut = int(len(train_frame) * (1.0 - config.VALIDATION_SIZE))
    return train_frame.iloc[:cut].copy(), train_frame.iloc[cut:].copy()


def _fit_and_score(estimator, scale_numeric, x_fit, y_fit, x_score) -> np.ndarray:
    pipeline = features.make_pipeline(clone(estimator), scale_numeric=scale_numeric)
    pipeline.fit(x_fit, y_fit)
    return pipeline.predict_proba(x_score)[:, 1]


def _write_run_metadata(raw_path, dataset_sha: str) -> None:
    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "lightgbm": lightgbm.__version__,
        "raw_sha256": dataset_sha,
        "raw_bytes": raw_path.stat().st_size,
        "seed": config.SEED,
    }
    config.RUN_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _baseline_metrics(y_test: pd.Series, *, budget: float) -> dict:
    prior = float(y_test.mean())
    n = len(y_test)
    contacts = int(n * budget)
    return {
        "random_at_budget": {
            "fraction": budget,
            "n_contacts": contacts,
            "precision": prior,
            "recall": budget,
            "f1": float(2 * prior * budget / (prior + budget)),
            "profit": float(
                contacts * prior * config.CONVERSION_VALUE - contacts * config.CALL_COST
            ),
        },
        "call_everyone": {
            "n_contacts": n,
            "precision": prior,
            "recall": 1.0,
            "f1": float(2 * prior / (prior + 1.0)),
            "profit": float(n * prior * config.CONVERSION_VALUE - n * config.CALL_COST),
        },
    }


def run(*, force_download: bool = False) -> dict:
    config.ensure_dirs()
    raw_path = data.download_raw(force=force_download)
    frame = data.load_raw()
    info = data.dataset_info(frame)
    train_frame, test_frame = data.split_by_time(frame)
    dev_frame, val_frame = split_train_validation(train_frame)

    x_train = train_frame.drop(columns=[config.TARGET])
    y_train = (train_frame[config.TARGET] == config.POSITIVE_LABEL).astype(int)
    x_test = test_frame.drop(columns=[config.TARGET])
    y_test = (test_frame[config.TARGET] == config.POSITIVE_LABEL).astype(int)
    x_dev = dev_frame.drop(columns=[config.TARGET])
    y_dev = (dev_frame[config.TARGET] == config.POSITIVE_LABEL).astype(int)
    x_val = val_frame.drop(columns=[config.TARGET])
    y_val = (val_frame[config.TARGET] == config.POSITIVE_LABEL).astype(int)

    candidates = candidate_estimators()

    # 1. Temporal CV over the whole training window: stability context.
    cv: dict[str, dict] = {}
    splitter = TimeSeriesSplit(n_splits=config.N_SPLITS)
    for name, (estimator, scale_numeric) in candidates.items():
        print(f"[cv] {name}", flush=True)
        result = cross_validate(estimator, scale_numeric, x_train, y_train, splitter, label=name)
        cv[name] = {"summary": result["summary"], "folds": result["folds"]}

    # 2. Recent holdout inside train: selection and operating point.
    print("[validation] fitting candidates on the recent window", flush=True)
    validation: dict[str, dict] = {}
    val_proba: dict[str, np.ndarray] = {}
    for name, (estimator, scale_numeric) in candidates.items():
        proba = _fit_and_score(estimator, scale_numeric, x_dev, y_dev, x_val)
        val_proba[name] = proba
        validation[name] = {
            **evaluate.ranking_metrics(y_val, proba),
            "n_train": int(len(dev_frame)),
            "n_valid": int(len(val_frame)),
            "positive_rate": float(y_val.mean()),
        }

    best_name = max(
        (name for name in validation if name != "dummy_prior"),
        key=lambda name: validation[name]["roc_auc"],
    )
    print(f"[validation] selected: {best_name}", flush=True)

    p_val = val_proba[best_name]
    budget_threshold_val = float(np.quantile(p_val, 1.0 - config.CONTACT_BUDGET))
    threshold_frame = evaluate.threshold_table(y_val, p_val)
    f1_threshold_val = float(threshold_frame.loc[threshold_frame["f1"].idxmax(), "threshold"])
    profit_threshold_val = float(
        threshold_frame.loc[threshold_frame["profit"].idxmax(), "threshold"]
    )

    # 3. Leakage audit: same folds, only `duration` added.
    print("[audit] duration leakage experiment", flush=True)
    leakage: dict[str, dict] = {}
    for name in ("logreg", "hist_gbdt"):
        estimator, scale_numeric = candidates[name]
        with_duration = cross_validate(
            estimator,
            scale_numeric,
            x_train,
            y_train,
            splitter,
            include_duration=True,
            label=f"{name}+duration",
        )
        without = cv[name]["summary"]["pr_auc"]["mean"]
        with_value = with_duration["summary"]["pr_auc"]["mean"]
        leakage[name] = {
            "without_duration": without,
            "with_duration": with_value,
            "delta": with_value - without,
        }

    # 4. Random CV as the naive counter-example.
    print("[audit] random split comparison", flush=True)
    stratified = cross_validate(
        candidates[best_name][0],
        candidates[best_name][1],
        x_train,
        y_train,
        StratifiedKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.SEED),
        label=f"{best_name}+stratified",
    )
    split_comparison = {
        "time_series_pr_auc": cv[best_name]["summary"]["pr_auc"]["mean"],
        "stratified_pr_auc": stratified["summary"]["pr_auc"]["mean"],
    }

    # 5. Final fit on the whole training window, one look at the test window.
    print("[fit] final model on the full training window", flush=True)
    final_pipeline = features.make_pipeline(
        clone(candidates[best_name][0]),
        scale_numeric=candidates[best_name][1],
    )
    final_pipeline.fit(x_train, y_train)
    test_proba = final_pipeline.predict_proba(x_test)[:, 1]

    budget_threshold_test = float(np.quantile(test_proba, 1.0 - config.CONTACT_BUDGET))
    budget_test = evaluate.threshold_metrics(y_test, test_proba, budget_threshold_test)
    f1_test = evaluate.threshold_metrics(y_test, test_proba, f1_threshold_val)
    profit_test = evaluate.threshold_metrics(y_test, test_proba, profit_threshold_val)

    dummy_pipeline = features.make_pipeline(DummyClassifier(strategy="prior"))
    dummy_pipeline.fit(x_train, y_train)
    dummy_proba = dummy_pipeline.predict_proba(x_test)[:, 1]

    baselines = _baseline_metrics(y_test, budget=config.CONTACT_BUDGET)
    baselines["dummy_prior"] = evaluate.ranking_metrics(y_test, dummy_proba)

    # 6. Diagnostics at the operational score cutoff.
    print("[diagnostics] slices, importance, worst cases", flush=True)
    slices = evaluate.slice_report(test_frame, y_test, test_proba, budget_threshold_test)
    importance = evaluate.compute_permutation_importance(final_pipeline, x_test, y_test)
    worst = evaluate.worst_cases(test_frame, y_test, test_proba, budget_threshold_test)

    numeric, categorical = features.feature_columns()
    bundle = {
        "pipeline": final_pipeline,
        "threshold": budget_threshold_test,
        "threshold_f1": f1_threshold_val,
        "threshold_profit": profit_threshold_val,
        "contact_budget": config.CONTACT_BUDGET,
        "feature_columns": numeric + categorical,
        "model_name": best_name,
        "model_version": config.MODEL_VERSION,
        "seed": config.SEED,
        "trained_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    joblib.dump(bundle, config.MODEL_PATH)

    metrics = {
        "dataset": {
            **info,
            "n_train": int(len(train_frame)),
            "n_test": int(len(test_frame)),
            "n_development": int(len(dev_frame)),
            "n_validation": int(len(val_frame)),
            "train_positive_rate": float(y_train.mean()),
            "validation_positive_rate": float(y_val.mean()),
            "test_positive_rate": float(y_test.mean()),
        },
        "cv": cv,
        "validation": validation,
        "selection_rule": "highest ROC-AUC on the last 20% of the training window",
        "selected_model": best_name,
        "leakage_experiment": leakage,
        "split_comparison": split_comparison,
        "operating_points": {
            "budget": {
                "fraction": config.CONTACT_BUDGET,
                "threshold_on_validation": budget_threshold_val,
                "threshold_on_test": budget_threshold_test,
                "test": budget_test,
            },
            "f1": {"threshold_on_validation": f1_threshold_val, "test": f1_test},
            "profit": {"threshold_on_validation": profit_threshold_val, "test": profit_test},
        },
        "test_ranking": {
            **evaluate.ranking_metrics(y_test, test_proba),
            "prior": float(y_test.mean()),
            "n": int(len(y_test)),
        },
        "baselines": baselines,
        "slices": slices,
        "importance": [
            {
                "feature": str(row.feature),
                "importance_mean": float(row.importance_mean),
                "importance_std": float(row.importance_std),
            }
            for row in importance.itertuples()
        ],
        "worst_cases": worst,
    }

    config.METRICS_JSON.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_run_metadata(raw_path, info["sha256"])

    print("[report] figures and markdown", flush=True)
    evaluate.plot_pr_curve(
        y_test, test_proba, float(y_test.mean()), config.FIGURES_DIR / "pr_curve.png"
    )
    evaluate.plot_calibration(y_test, test_proba, config.FIGURES_DIR / "calibration.png")
    evaluate.plot_thresholds(threshold_frame, config.FIGURES_DIR / "threshold.png")
    evaluate.plot_budget_curve(
        y_test, test_proba, config.CONTACT_BUDGET, config.FIGURES_DIR / "budget.png"
    )
    evaluate.plot_importance(importance, config.FIGURES_DIR / "importance.png")
    evaluate.write_report(metrics, config.REPORT_MD)

    print(f"[done] model: {best_name}", flush=True)
    print(f"[done] test PR-AUC: {metrics['test_ranking']['pr_auc']:.4f}", flush=True)
    print(f"[done] test ROC-AUC: {metrics['test_ranking']['roc_auc']:.4f}", flush=True)
    print(f"[done] precision @ budget: {budget_test['precision']:.4f}", flush=True)
    print(f"[done] recall @ budget: {budget_test['recall']:.4f}", flush=True)
    return metrics
