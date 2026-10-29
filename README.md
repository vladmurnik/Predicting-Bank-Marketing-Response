**English** · [Русский](README.ru.md)

# Predicting Bank Marketing Response

Binary classification on the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing): given a client profile and campaign context, predict whether the client will subscribe to a term deposit.

The pipeline uses a chronological split, excludes the `duration` feature (unknown before the call), compares four models on identical folds, selects the operating point by call budget instead of a fixed probability threshold, and exposes the model through a FastAPI service.

## Results

Holdout: May–November 2010, the last 20% of the file. It was evaluated once, after model selection and threshold logic were fixed.

| Metric | Value |
| --- | --- |
| PR-AUC | **0.527** (prior 0.308, ×1.71) |
| ROC-AUC | **0.748** |
| Brier score | **0.191** |
| Precision @ 30% budget | **0.551** |
| Recall @ 30% budget | 0.536 |
| F1 @ 30% budget | 0.543 |
| Precision, random 30% | 0.308 |
| F1, call everyone | 0.471 |

PR-AUC is lower than results that include `duration`. That feature is not available at scoring time and is excluded.

## Dataset

| Property | Value |
| --- | --- |
| Source | UCI Bank Marketing, `bank-additional-full.csv` (id 222) |
| Rows | 41,188 |
| Period | May 2008 – November 2010 |
| Features | 19 (9 numeric, 10 categorical) |
| Target | `y` (`yes` / `no`), 11.27% positive |
| Missing values | encoded as `unknown`, treated as a category |
| Duplicates | 12 exact duplicates, kept |

## Pipeline

### Time-based split

Rows are chronological, so the split is by time: the first 80% for training, the last 20% for testing. No shuffling.

A random split was also measured for comparison: PR-AUC rose from 0.086 to 0.206. The difference comes from leaking future months into training, not from a better model. The number is kept in the report as a reference.

### Leakage guard

`duration` is the length of the last call and the strongest predictor in the dataset. Including it raises cross-validated PR-AUC from 0.086 to 0.402 for logistic regression and from 0.076 to 0.456 for HistGradientBoosting.

The feature is excluded because the call duration is unknown before the call is made. `tests/test_leakage_guard.py` fails if `duration` is added back to the feature contract or the API schema.

### Models and selection

Four candidates are compared on identical folds: a prior-only `DummyClassifier`, logistic regression, HistGradientBoosting and LightGBM. The selection rule was fixed before the holdout was touched: highest ROC-AUC on the last 20% of the training window.

Logistic regression was selected (0.575 vs 0.544 and 0.560). The gap is within noise: on this dataset the larger models do not provide a stable advantage, and the linear model transfers better across the distribution shift.

### Operating point

The F1-optimal threshold on validation was 0.03. On the test window it would call 99.2% of the base: the positive rate grows from 12.7% to 30.8%, so the same probability threshold selects a different volume.

The operating point is therefore defined as a budget: call the top 30% by score and recompute the cutoff as the 70th percentile of each batch. On the holdout this yields precision 0.551 vs 0.308 for a random 30% sample at the same call volume.

### Error analysis

`reports/report.md` contains calibration curves, permutation importance, slices by month, job, age and number of `unknown` fields, and the highest-confidence false positives and false negatives. Performance is stronger in the 2010 regime and weaker in 2008–2009; the report states this explicitly.

## Tech stack

| Tech | Purpose |
| --- | --- |
| Python 3.12 | pipeline and service |
| uv | environment and lockfile |
| pandas / NumPy | loading, splitting, slicing |
| scikit-learn | `Pipeline`, logistic regression, HistGradientBoosting, metrics, calibration |
| LightGBM | gradient-boosting candidate |
| matplotlib | PR curve, calibration, threshold and budget plots |
| joblib | model bundle with the pipeline and threshold |
| FastAPI + Pydantic | `/predict` and `/health`; `duration` rejected by the schema |
| Uvicorn | API server |
| pytest | 18 tests, including the leakage guard |
| Ruff | lint and formatting |
| Git + GitHub | version control |

## Project structure

```text
.
├── src/bank_marketing/
│   ├── config.py      # paths, seed, business constants
│   ├── data.py        # download, schema checks, time split
│   ├── features.py    # feature contract and preprocessing
│   ├── train.py       # CV, model selection, final fit, artifacts
│   ├── evaluate.py    # metrics, slices, figures, report
│   ├── serve.py       # FastAPI service
│   └── cli.py         # download / train / all
├── tests/             # 18 tests, no dataset needed
├── reports/           # report.md, metrics.json, figures
├── models/model.joblib
├── MODEL_CARD.md
├── LEARN.md           # beginner walkthrough (RU)
├── ASSUMPTIONS.md     # design decisions and trade-offs (RU)
├── make.py            # task runner for Windows
└── Makefile
```

## Quick start

```bash
uv sync
uv run python -m bank_marketing.cli all
```

On Windows without GNU make:

```powershell
uv run python make.py repro
```

The command downloads the dataset (~1 MB, a zip inside a zip), trains the models and writes `reports/report.md`, `reports/metrics.json` and `models/model.joblib`.

Run the tests:

```bash
uv run pytest
```

The suite does not require the dataset. Two consecutive `cli all` runs produce identical `reports/metrics.json` (seed 42).

## API

```bash
uv run uvicorn bank_marketing.serve:app --host 127.0.0.1 --port 8000
```

**Live demo:** not deployed yet.

The service accepts one raw client row and returns `probability`, `threshold`, `prediction` and `model_version`. `duration` is rejected by the request schema.

## License

MIT — see [LICENSE](LICENSE).
