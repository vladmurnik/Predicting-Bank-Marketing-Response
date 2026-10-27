# Model Card: Bank Marketing Term-Deposit Propensity

## 1. Model

| Field | Value |
| --- | --- |
| Version | `bank-marketing-0.1.0` |
| Type | logistic regression inside a `Pipeline` (one-hot + median imputer + StandardScaler) |
| Task | binary classification: will the client subscribe to a term deposit |
| Output | `P(yes)` and a decision `probability >= threshold` |
| Artifact | `models/model.joblib` (pipeline + threshold + feature list) |
| Seed | 42 |
| Training | `uv run python -m bank_marketing.cli all` |

The model was selected by a rule fixed before the holdout was scored: highest
ROC-AUC on the last 20% of the training window. Logistic regression beat
HistGradientBoosting and LightGBM by a margin within noise (0.575 versus 0.544
and 0.560). On this data a more complex model does not buy a stable advantage,
and the linear model transfers the distribution shift better.

## 2. Intended Use

**Use for:** prioritising a client list when call-centre capacity is limited.
The model ranks the base and the business contacts the top 30% (configurable in
`config.CONTACT_BUDGET`).

**Do not use for:**

* the sole reason for a call — a business rule makes the decision;
* clients outside the training distribution (other products, regions, years);
* automated denial of service;
* absolute probability estimates without recalibration on the current window:
  drift shifts the probabilities, so the cutoff is a score quantile.

## 3. Data

| Field | Value |
| --- | --- |
| Source | UCI Bank Marketing, `bank-additional-full.csv` (id 222) |
| Rows | 41,188 |
| Period | May 2008 – November 2010 |
| Features | 19 (9 numeric, 10 categorical) |
| Target | `y` (`yes` / `no`), 11.27% positive |
| Duplicates | 12 exact duplicates, kept |
| Missing values | encoded as `unknown`, treated as a category |

**Excluded feature:** `duration` (length of the last call). It is known only
after the call and does not exist when the client is chosen. With it, CV PR-AUC
rises from 0.086 to 0.402 (logreg) and from 0.076 to 0.456
(HistGradientBoosting) — that is the cost of leakage. Re-introducing it is
blocked by `tests/test_leakage_guard.py`.

## 4. Validation

| Window | Rows | Positive rate | Purpose |
| --- | --- | --- | --- |
| development | 26,360 | 4.8% | fit candidates |
| validation | 6,590 | 12.7% | model selection and operating point |
| test | 8,238 | 30.8% | one final measurement |

The split is strictly temporal. Random cross-validation gives PR-AUC 0.206
versus 0.086 for the time-series scheme — not progress, but leakage of future
months through shuffling. It is reported only as a counter-example.

## 5. Test Metrics

| Metric | Value |
| --- | --- |
| PR-AUC | 0.527 |
| ROC-AUC | 0.748 |
| Brier | 0.191 |
| Positive rate (prior) | 0.308 |
| PR-AUC / prior | ×1.71 |
| Precision @ 30% budget | 0.551 |
| Recall @ 30% budget | 0.536 |
| F1 @ 30% budget | 0.543 |
| Precision, random 30% | 0.308 |
| F1, call everyone | 0.471 |

PR-AUC looks modest only next to textbook examples: the test window is 2010,
with a 31% positive rate, not a random sample across all years. Against its own
prior the model gives ×1.71, and at a fixed contact budget it lifts precision
from 30.8% to 55.1%.

## 6. Operating Point

The threshold in the bundle (`threshold ≈ 0.298`) is the 70th percentile of the
score on the test batch, not a magic number. In production the cutoff is
recomputed for every batch: take the top 30% by score. An absolute threshold
tuned on validation transfers poorly, because the positive rate grew 2.4×
between windows.

## 7. Calibration

Brier 0.191; the calibration curve is in `reports/figures/calibration.png`.
Post-hoc calibration (isotonic/Platt) was deliberately not applied: it would be
fitted on a historical window with a different positive rate and would be worse
in the new regime. A quantile cutoff is used instead.

## 8. Limitations and Risks

1. **Drift.** The positive rate grew from 4.8% to 30.8% between windows. The
   model uses `month` and macro variables as time proxies; ranking degrades
   outside 2010.
2. **One holdout.** No confidence intervals. A launch decision needs a monthly
   bootstrap and a second temporal cut.
3. **Fairness was not audited.** `age`, `job`, `marital` and `education` may be
   proxies for protected groups. Slice metrics are in the report, but no legal
   review was performed.
4. **The economics are assumed.** Call cost 1 and conversion value 10 are a
   teaching assumption, not a measurement.
5. **The 30% budget** is an assumption about call-centre capacity.
6. **No production monitoring.** The repository has no drift detector;
   recommended: PSI on features and precision@budget on a labelled batch.

## 9. Retraining and Monitoring

* Retrain when the campaign changes, or quarterly on the last 12 months.
* Compare precision@budget on a new labelled batch with 0.551.
* Track PSI for `month`, `pdays`, `euribor3m`, `nr.employed`.
* If precision@budget falls below random selection, stop using the model.

## 10. Reproduction

```bash
uv sync
uv run python -m bank_marketing.cli all
uv run pytest -q
```

Metrics: `reports/metrics.json`. Report: `reports/report.md`.
Environment versions and the dataset sha256: `reports/run.json`.
