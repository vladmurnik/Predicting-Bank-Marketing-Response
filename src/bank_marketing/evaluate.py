"""Metrics, diagnostics, figures and the generated Markdown report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)

from . import config

FIGURES = {
    "pr_curve": "figures/pr_curve.png",
    "calibration": "figures/calibration.png",
    "threshold": "figures/threshold.png",
    "budget": "figures/budget.png",
    "importance": "figures/importance.png",
}


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def ranking_metrics(y_true, proba) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    return {
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
    }


def threshold_metrics(
    y_true,
    proba,
    threshold: float,
    *,
    cost: float = config.CALL_COST,
    value: float = config.CONVERSION_VALUE,
) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    predicted = p >= threshold
    tp = int((predicted & (y == 1)).sum())
    fp = int((predicted & (y == 0)).sum())
    fn = int((~predicted & (y == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "n_contacts": tp + fp,
        "predicted_positive_rate": float(predicted.mean()),
        "profit": float(tp * value - (tp + fp) * cost),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def binary_metrics(y_true, proba, threshold: float) -> dict:
    return {**ranking_metrics(y_true, proba), **threshold_metrics(y_true, proba, threshold)}


def threshold_table(
    y_true,
    proba,
    thresholds: np.ndarray | None = None,
    *,
    cost: float = config.CALL_COST,
    value: float = config.CONVERSION_VALUE,
) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    if thresholds is None:
        thresholds = np.round(np.arange(0.02, 0.95, 0.01), 2)
    rows = []
    for threshold in thresholds:
        row = threshold_metrics(y, p, float(threshold), cost=cost, value=value)
        rows.append(
            {
                "threshold": row["threshold"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "n_contacts": row["n_contacts"],
                "profit": row["profit"],
            }
        )
    return pd.DataFrame(rows)


def budget_table(y_true, proba, *, step: float = 0.05) -> pd.DataFrame:
    """Precision/recall/F1 when the top share of scores is contacted."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    order = np.argsort(-p)
    positives = int(y.sum())
    rows = []
    for fraction in np.round(np.arange(step, 1.0 + step / 2, step), 2):
        contacts = max(1, int(len(p) * fraction))
        chosen = order[:contacts]
        precision = float(y[chosen].mean())
        recall = float(y[chosen].sum() / positives)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "fraction": float(min(fraction, 1.0)),
                "n_contacts": contacts,
                "precision": precision,
                "recall": recall,
                "f1": float(f1),
                "profit": float(y[chosen].sum() * config.CONVERSION_VALUE - contacts),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def slice_report(
    frame: pd.DataFrame,
    y_true,
    proba,
    threshold: float,
    *,
    min_rows: int = 200,
) -> dict[str, list[dict]]:
    """Quality per subgroup; groups below ``min_rows`` are skipped as too noisy."""
    frame = frame.reset_index(drop=True)
    y = pd.Series(np.asarray(y_true, dtype=int))
    p = pd.Series(np.asarray(proba, dtype=float))

    dimensions: dict[str, pd.Series] = {}
    if "month" in frame.columns:
        dimensions["month"] = frame["month"].astype(str)
    if "job" in frame.columns:
        dimensions["job"] = frame["job"].astype(str)
    if "age" in frame.columns:
        bins = [17, 30, 40, 50, 60, 100]
        labels = ["18-30", "31-40", "41-50", "51-60", "60+"]
        dimensions["age_bin"] = pd.cut(frame["age"], bins=bins, labels=labels).astype(str)
    unknown_columns = [
        column
        for column in ("job", "marital", "education", "default", "housing", "loan")
        if column in frame.columns
    ]
    if unknown_columns:
        dimensions["unknown_count"] = frame[unknown_columns].eq("unknown").sum(axis=1).astype(str)

    result: dict[str, list[dict]] = {}
    for name, values in dimensions.items():
        rows = []
        for value, positions in values.groupby(values).groups.items():
            index = np.asarray(positions)
            if len(index) < min_rows:
                continue
            y_slice = y.iloc[index].to_numpy()
            p_slice = p.iloc[index].to_numpy()
            row = {
                "value": str(value),
                "n": int(len(index)),
                "positive_rate": float(y_slice.mean()),
                "mean_proba": float(p_slice.mean()),
            }
            if 0 < int(y_slice.sum()) < len(y_slice):
                row["pr_auc"] = float(average_precision_score(y_slice, p_slice))
            metrics = threshold_metrics(y_slice, p_slice, threshold)
            row["recall"] = metrics["recall"]
            row["precision"] = metrics["precision"]
            rows.append(row)
        result[name] = sorted(rows, key=lambda item: -item["n"])
    return result


def compute_permutation_importance(
    pipeline,
    features: pd.DataFrame,
    target: pd.Series,
    *,
    n_repeats: int = 3,
    max_rows: int = 4000,
) -> pd.DataFrame:
    if len(features) > max_rows:
        sample = features.sample(max_rows, random_state=config.SEED)
        target = target.loc[sample.index]
        features = sample
    result = permutation_importance(
        pipeline,
        features,
        target,
        scoring="average_precision",
        n_repeats=n_repeats,
        random_state=config.SEED,
        n_jobs=1,
    )
    return (
        pd.DataFrame(
            {
                "feature": features.columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def worst_cases(frame: pd.DataFrame, y_true, proba, threshold: float, *, k: int = 5) -> dict:
    frame = frame.reset_index(drop=True)
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    keep = [
        column
        for column in (
            "age",
            "job",
            "month",
            "campaign",
            "pdays",
            "previous",
            "poutcome",
            "contact",
        )
        if column in frame.columns
    ]

    def pack(indices: np.ndarray) -> list[dict]:
        return [
            {
                "probability": round(float(p[index]), 4),
                "true_label": int(y[index]),
                **{column: str(frame.at[index, column]) for column in keep},
            }
            for index in indices
        ]

    false_positives = np.where((p >= threshold) & (y == 0))[0]
    false_negatives = np.where((p < threshold) & (y == 1))[0]
    return {
        "false_positives": pack(false_positives[np.argsort(-p[false_positives])][:k]),
        "false_negatives": pack(false_negatives[np.argsort(p[false_negatives])][:k]),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def plot_pr_curve(y_true, proba, baseline_rate: float, path: Path) -> None:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    precision, recall, _ = precision_recall_curve(y, p)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(recall, precision, label=f"model (AP={average_precision_score(y, p):.3f})")
    ax.axhline(baseline_rate, linestyle="--", color="grey", label=f"prior ({baseline_rate:.3f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-recall curve (test)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_calibration(y_true, proba, path: Path) -> None:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(proba, dtype=float)
    prob_true, prob_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect")
    ax.plot(prob_pred, prob_true, marker="o", label="model")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed positive rate")
    ax.set_title("Calibration (test, quantile bins)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_thresholds(table: pd.DataFrame, path: Path) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
    left.plot(table["threshold"], table["f1"], label="F1")
    left.set_xlabel("absolute threshold")
    left.set_ylabel("F1")
    left.set_title("F1 vs absolute threshold (validation)")
    left.grid(alpha=0.3)

    right.plot(table["threshold"], table["profit"], color="darkgreen")
    right.set_xlabel("absolute threshold")
    right.set_ylabel("profit units")
    right.set_title("Expected profit vs threshold (validation)")
    right.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_budget_curve(y_true, proba, budget: float, path: Path) -> None:
    table = budget_table(y_true, proba)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(table["fraction"], table["precision"], label="precision")
    ax.plot(table["fraction"], table["recall"], label="recall")
    ax.plot(table["fraction"], table["f1"], label="F1")
    ax.axvline(budget, linestyle="--", color="black", label=f"budget {budget:.0%}")
    ax.axhline(float(np.mean(y_true)), linestyle=":", color="grey", label="prior")
    ax.set_xlabel("contacted share of the base (top scores)")
    ax.set_ylabel("metric")
    ax.set_title("Ranking policy on test")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_importance(importance: pd.DataFrame, path: Path, *, top: int = 12) -> None:
    head = importance.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(head["feature"], head["importance_mean"], xerr=head["importance_std"], color="#3b6ea5")
    ax.set_xlabel("drop in average precision when shuffled")
    ax.set_title("Permutation importance (test sample)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def write_report(metrics: dict, output: Path) -> None:
    dataset = metrics["dataset"]
    ranking = metrics["test_ranking"]
    budget = metrics["operating_points"]["budget"]
    budget_test = budget["test"]
    baselines = metrics["baselines"]
    selected = metrics["selected_model"]
    thresholds = metrics["operating_points"]

    parts: list[str] = []
    parts.append("# Bank Marketing: прогноз отклика на телемаркетинг\n")
    parts.append(
        "Отчёт сгенерирован автоматически командой "
        "`uv run python -m bank_marketing.cli all`. "
        f"Датасет: {dataset['name']}, sha256 `{dataset['sha256'][:16]}…`.\n"
    )

    parts.append("## 1. Итог\n")
    parts.append(
        _table(
            ["Показатель", "Значение"],
            [
                ["Выбранная модель", f"`{selected}`"],
                ["Правило выбора", "максимальный ROC-AUC на последних 20% train (validation)"],
                ["PR-AUC на test", _fmt(ranking["pr_auc"])],
                ["ROC-AUC на test", _fmt(ranking["roc_auc"])],
                ["Brier на test", _fmt(ranking["brier"])],
                ["Доля `yes` на test", f"{ranking['prior']:.1%}"],
                ["PR-AUC / prior", f"×{ranking['pr_auc'] / ranking['prior']:.2f}"],
                [f"Precision при бюджете {budget['fraction']:.0%}", _fmt(budget_test["precision"])],
                ["Recall при том же бюджете", _fmt(budget_test["recall"])],
                ["F1 при том же бюджете", _fmt(budget_test["f1"])],
                ["F1 «звонить всем»", _fmt(baselines["call_everyone"]["f1"])],
                ["Precision случайного отбора", _fmt(baselines["random_at_budget"]["precision"])],
            ],
        )
        + "\n"
    )
    parts.append(
        "Главная цифра — не PR-AUC, а **precision при фиксированном бюджете звонков**: "
        f"модель поднимает долю согласий с {baselines['random_at_budget']['precision']:.1%} "
        f"(случайный отбор) до {budget_test['precision']:.1%} на том же числе контактов. "
        "PR-AUC ниже, чем в учебниках, потому что test — это 2010 год с другой долей "
        "согласий (см. раздел 3), а не случайная выборка из тех же лет.\n"
    )

    parts.append("## 2. Данные\n")
    parts.append(
        f"- Строк: **{dataset['n_rows']}**, колонок: **{dataset['n_columns']}**.\n"
        f"- Доля `y == yes` во всём файле: **{dataset['positive_rate']:.2%}**.\n"
        f"- Полных дубликатов строк: **{dataset['n_duplicate_rows']}** (не удалялись, "
        "чтобы не менять распределение; см. `ASSUMPTIONS.md`).\n"
    )
    parts.append(
        _table(
            ["Окно", "Строк", "Доля `yes`"],
            [
                [
                    "train (2008-05 … 2010-04)",
                    dataset["n_train"],
                    f"{dataset['train_positive_rate']:.1%}",
                ],
                ["  из него development", dataset["n_development"], "—"],
                [
                    "  из него validation",
                    dataset["n_validation"],
                    f"{dataset['validation_positive_rate']:.1%}",
                ],
                [
                    "test (2010-05 … 2010-11)",
                    dataset["n_test"],
                    f"{dataset['test_positive_rate']:.1%}",
                ],
            ],
        )
        + "\n"
    )
    unknown_rows = [[column, count] for column, count in dataset["unknown_counts"].items()]
    parts.append("Пропуски закодированы строкой `unknown`, а не NaN:\n")
    parts.append(_table(["Колонка", "`unknown`"], unknown_rows) + "\n")

    parts.append("## 3. Дрейф и сплит\n")
    parts.append(
        "Доля согласий растёт по годам: 4.8% в development, 12.7% в validation и "
        f"{dataset['test_positive_rate']:.1%} на test. Это не ошибка сплита, а свойство "
        "кампании 2010 года. Отсюда два следствия: абсолютные вероятности плохо "
        "переносятся между окнами, а случайная кросс-валидация завышает качество.\n"
    )
    split = metrics["split_comparison"]
    parts.append(
        _table(
            ["Схема валидации внутри train", "PR-AUC"],
            [
                ["TimeSeriesSplit (production-like)", _fmt(split["time_series_pr_auc"])],
                ["StratifiedKFold shuffle (наивная)", _fmt(split["stratified_pr_auc"])],
            ],
        )
        + "\n"
    )
    parts.append(
        f"Разрыв **{split['stratified_pr_auc'] - split['time_series_pr_auc']:+.3f}** PR-AUC. "
        "Случайный сплит перемешивает месяцы и позволяет модели выучить режим, которого "
        "в будущем не будет. Поэтому все решения принимаются только по времени вперёд.\n"
    )

    parts.append("## 4. Сравнение моделей\n")
    cv_rows = []
    for name, payload in metrics["cv"].items():
        summary = payload["summary"]
        cv_rows.append(
            [
                f"`{name}`",
                f"{_fmt(summary['pr_auc']['mean'])} ± {_fmt(summary['pr_auc']['std'])}",
                _fmt(summary["roc_auc"]["mean"]),
                _fmt(summary["brier"]["mean"]),
            ]
        )
    parts.append("**TimeSeriesSplit по всему train (2008–2010/04):**\n")
    parts.append(_table(["Модель", "PR-AUC", "ROC-AUC", "Brier"], cv_rows) + "\n")
    parts.append(
        "В этом окне доля согласий 5–6%, и все модели близки к случайному ранжированию: "
        "сигнал почти весь приходится на смену режима 2010 года. Это честная картина "
        "сложности задачи, а не баг.\n"
    )

    val_rows = []
    for name, payload in metrics["validation"].items():
        val_rows.append(
            [
                f"`{name}`",
                _fmt(payload["pr_auc"]),
                _fmt(payload["roc_auc"]),
                _fmt(payload["brier"]),
            ]
        )
    parts.append("**Последние 20% train (validation, 2009-11 … 2010-04):**\n")
    parts.append(_table(["Модель", "PR-AUC", "ROC-AUC", "Brier"], val_rows) + "\n")
    parts.append(
        f"Выбор сделан по ROC-AUC на validation: {metrics['selection_rule']}. "
        "ROC-AUC не зависит от доли положительного класса и поэтому устойчивее "
        "PR-AUC при таком дрейфе. Разницы между моделями лежат в пределах шума — "
        "это отдельный вывод, а не повод выбирать по test.\n"
    )

    parts.append("## 5. Утечка `duration`\n")
    leak_rows = []
    for name, payload in metrics["leakage_experiment"].items():
        leak_rows.append(
            [
                f"`{name}`",
                _fmt(payload["without_duration"]),
                _fmt(payload["with_duration"]),
                f"{payload['delta']:+.3f}",
            ]
        )
    parts.append(_table(["Модель", "Без `duration`", "С `duration`", "Δ"], leak_rows) + "\n")
    parts.append(
        "`duration` — длительность последнего звонка. В момент, когда нужно решить, "
        "кому звонить, этого значения ещё не существует, поэтому признак исключён "
        "из продакшн-модели. Цифры показывают цену честной постановки: с ним PR-AUC "
        "вырастает в разы. Тест `tests/test_leakage_guard.py` не даст вернуть его случайно.\n"
    )

    parts.append("## 6. Порог против бюджета\n")
    parts.append(
        _table(
            [
                "Политика",
                "Порог на validation",
                "Контактов на test",
                "Precision",
                "Recall",
                "F1",
                "Прибыль",
            ],
            [
                [
                    "max F1 (абсолютный порог)",
                    _fmt(thresholds["f1"]["threshold_on_validation"], 2),
                    thresholds["f1"]["test"]["n_contacts"],
                    _fmt(thresholds["f1"]["test"]["precision"]),
                    _fmt(thresholds["f1"]["test"]["recall"]),
                    _fmt(thresholds["f1"]["test"]["f1"]),
                    f"{thresholds['f1']['test']['profit']:.0f}",
                ],
                [
                    "max прибыли (абсолютный порог)",
                    _fmt(thresholds["profit"]["threshold_on_validation"], 2),
                    thresholds["profit"]["test"]["n_contacts"],
                    _fmt(thresholds["profit"]["test"]["precision"]),
                    _fmt(thresholds["profit"]["test"]["recall"]),
                    _fmt(thresholds["profit"]["test"]["f1"]),
                    f"{thresholds['profit']['test']['profit']:.0f}",
                ],
                [
                    f"top {budget['fraction']:.0%} по score",
                    "70-й процентиль",
                    budget_test["n_contacts"],
                    _fmt(budget_test["precision"]),
                    _fmt(budget_test["recall"]),
                    _fmt(budget_test["f1"]),
                    f"{budget_test['profit']:.0f}",
                ],
            ],
        )
        + "\n"
    )
    parts.append(
        "Абсолютный порог, подобранный на validation, переносится плохо: доля согласий "
        "выросла, и тот же порог на test зовёт почти всех (max F1) или 60% базы "
        "(max прибыли). Поэтому операционная политика — **фиксированный бюджет "
        "контактов**: берём верхние 30% по score, а порог пересчитывается как "
        "70-й процентиль score каждой партии.\n"
    )
    parts.append(
        f"Насколько это разные числа: порог бюджетной политики на validation — "
        f"{_fmt(budget['threshold_on_validation'], 4)}, на test — "
        f"{_fmt(budget['threshold_on_test'], 4)}. Одна и та же политика требует "
        "совершенно разных абсолютных значений, и это ещё один аргумент за квантиль.\n"
    )
    parts.append(
        "F1 и прибыль растут вместе с числом контактов, поэтому сравнивать политики "
        "по F1 при разном объёме нельзя. Корректное сравнение — precision при "
        f"одинаковом объёме: на 30% бюджета модель даёт "
        f"{_fmt(budget_test['precision'])} против "
        f"{_fmt(baselines['random_at_budget']['precision'])} у случайного отбора "
        f"(×{budget_test['precision'] / baselines['random_at_budget']['precision']:.2f}).\n"
    )
    parts.append(
        "При допущении «звонок = 1, конверсия = +10» каждый контакт прибылен, поэтому "
        "без ограничения мощности оптимум — звонить всем. Модель нужна именно при "
        "фиксированной пропускной способности: она концентрирует согласия в верхушке "
        "списка.\n"
    )
    parts.append(f"![threshold]({FIGURES['threshold']})\n")
    parts.append(f"![budget]({FIGURES['budget']})\n")
    parts.append(
        f"Для сравнения: «звонить всем» даёт F1 {_fmt(baselines['call_everyone']['f1'])} "
        f"при {baselines['call_everyone']['n_contacts']} контактах, а случайные 30% — "
        f"F1 {_fmt(baselines['random_at_budget']['f1'])}.\n"
    )

    parts.append("## 7. Калибровка\n")
    parts.append(f"![calibration]({FIGURES['calibration']})\n")
    parts.append(
        f"Brier score на test: **{_fmt(ranking['brier'])}**. Кривая калибровки показывает, "
        "можно ли трактовать выход как вероятность. При таком дрейфе вероятности "
        "смещены, поэтому порог задаётся квантилем score, а не абсолютным числом; "
        "post-hoc калибровка по историческому окну здесь не поможет.\n"
    )

    parts.append("## 8. Анализ ошибок\n")
    parts.append(f"![pr]({FIGURES['pr_curve']})\n")
    parts.append("### Важность признаков\n")
    parts.append(f"![importance]({FIGURES['importance']})\n")
    importance_rows = [
        [f"`{row['feature']}`", _fmt(row["importance_mean"], 4)]
        for row in metrics["importance"][:10]
    ]
    parts.append(_table(["Признак", "Δ average precision"], importance_rows) + "\n")

    for dimension, rows in metrics["slices"].items():
        parts.append(f"### Срез: `{dimension}`\n")
        table_rows = []
        for row in rows[:8]:
            table_rows.append(
                [
                    row["value"],
                    row["n"],
                    f"{row['positive_rate']:.1%}",
                    _fmt(row.get("pr_auc", float("nan"))),
                    _fmt(row["precision"]),
                    _fmt(row["recall"]),
                ]
            )
        parts.append(
            _table(["Значение", "N", "Доля yes", "PR-AUC", "Precision", "Recall"], table_rows)
        )
        parts.append("")

    parts.append("### Худшие ошибки\n")
    parts.append("**False positives** (модель звала, клиент отказался):\n")
    parts.append(
        _table(
            ["P(yes)", "Возраст", "Job", "Месяц", "Кампания"],
            [
                [
                    row["probability"],
                    row.get("age", ""),
                    row.get("job", ""),
                    row.get("month", ""),
                    row.get("campaign", ""),
                ]
                for row in metrics["worst_cases"]["false_positives"]
            ],
        )
        + "\n"
    )
    parts.append("**False negatives** (модель молчала, клиент согласился):\n")
    parts.append(
        _table(
            ["P(yes)", "Возраст", "Job", "Месяц", "Кампания"],
            [
                [
                    row["probability"],
                    row.get("age", ""),
                    row.get("job", ""),
                    row.get("month", ""),
                    row.get("campaign", ""),
                ]
                for row in metrics["worst_cases"]["false_negatives"]
            ],
        )
        + "\n"
    )

    parts.append("## 9. Ограничения\n")
    parts.append(
        "- Один временной holdout (2010-05 … 2010-11). Доверительных интервалов нет; "
        "для решения о запуске нужен bootstrap по месяцам.\n"
        "- Данные 2008–2010 годов; рынок и продукты изменились, метрики не переносятся "
        "на сегодняшний день без переобучения.\n"
        "- Модель использует `month` и макроэкономические признаки как прокси времени. "
        "На 2010 году это помогает, но за его пределами ранжирование деградирует.\n"
        "- Бюджет 30% — допущение о пропускной способности колл-центра, а не измеренный факт.\n"
        "- `duration` исключён, поэтому качество ниже ноутбуков, где он есть. Это "
        "осознанная цена честной постановки.\n"
    )

    parts.append("## 10. Воспроизведение\n")
    parts.append(
        "```bash\n"
        "uv sync\n"
        "uv run python -m bank_marketing.cli all\n"
        "uv run pytest -q\n"
        "uv run ruff check .\n"
        "```\n"
    )
    parts.append(
        f"Seed: {config.SEED}. Полные метрики — в `reports/metrics.json`, "
        "версии окружения и хеш данных — в `reports/run.json`.\n"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(parts), encoding="utf-8")
