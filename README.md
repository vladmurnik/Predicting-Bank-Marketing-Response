# Bank Marketing: Will They Say Yes?

**English** · [Русский](README.ru.md)

It's a binary classifier on the UCI Bank Marketing dataset. Given a client and
the campaign context, will they subscribe to a term deposit?

---

## The numbers

Test window: May–November 2010, the last 20% of the file. I looked at it once,
after every decision was already fixed.

| Metric | Value |
| --- | --- |
| PR-AUC | **0.527** (prior 0.308, ×1.71) |
| ROC-AUC | **0.748** |
| Brier | **0.191** |
| Precision @ 30% budget | **0.551** |
| Recall @ 30% budget | 0.536 |
| F1 @ 30% budget | 0.543 |
| Precision, random 30% | 0.308 |
| F1, call everyone | 0.471 |

Are these numbers impressive? Not next to notebooks that use `duration`. But
those notebooks are cheating, and I'll show you exactly how.

## The data

| Property | Value |
| --- | --- |
| Source | UCI Bank Marketing, `bank-additional-full.csv` (id 222) |
| Rows | 41,188 |
| Period | May 2008 – November 2010 |
| Features | 19 (9 numeric, 10 categorical) |
| Target | `y` (`yes` / `no`), 11.27% positive |
| Missing values | encoded as `unknown`, treated as a category |
| Duplicates | 12 exact duplicates, kept on purpose |

## Tech stack

| Tech | What I used it for |
| --- | --- |
| Python 3.12 | the whole pipeline |
| uv | environment and lockfile, so runs are reproducible |
| pandas / NumPy | loading, splitting, slicing, the boring parts |
| scikit-learn | `Pipeline`, logistic regression, HistGradientBoosting, metrics, calibration |
| LightGBM | the gradient-boosting candidate |
| matplotlib | PR curve, calibration, threshold and budget plots |
| joblib | one model bundle with the pipeline and the threshold |
| FastAPI + Pydantic | `/predict` and `/health`, with `duration` rejected by the schema |
| Uvicorn | serving the API |
| pytest | 18 tests, including the leakage guard |
| Ruff | lint and formatting |
| Git + GitHub | this history |

## How it works

### 1. Time goes one way

The rows are chronological, so I split by time: first 80% train, last 20% test.
No shuffling, no "just this once".

For comparison I also ran a random split. PR-AUC jumped from 0.086 to 0.206.
That's not a better model — that's the model peeking at future months. I left
the number in the report as a reminder to myself.

### 2. The leakage trap

`duration` is the length of the last call. It's also the most predictive column
in the dataset. With it, CV PR-AUC goes from 0.086 to 0.402 for logistic
regression and from 0.076 to 0.456 for HistGradientBoosting.

One small problem: you don't know how long the call lasted before you decide to
call.

So I dropped it and wrote `tests/test_leakage_guard.py`, which fails if anyone —
including future me — adds it back.

### 3. Four models, one rule

I compared a prior-only `DummyClassifier`, logistic regression,
HistGradientBoosting and LightGBM on identical folds. The selection rule was
fixed before I touched the holdout: highest ROC-AUC on the last 20% of the
training window.

Logistic regression won (0.575 vs 0.544 and 0.560). The gap is within noise,
which is its own lesson: on this data a bigger model doesn't buy a stable
advantage, and the linear one handles the distribution shift better.

### 4. The threshold problem

The F1-optimal threshold on validation was 0.03. On test, 0.03 called 99.2% of
the base. Nothing was broken — the positive rate grew from 12.7% to 30.8%, so
the same number means something completely different in a new regime.

So I stopped pretending that probabilities transfer. The operating point is a
budget: call the top 30% by score, and recompute the cutoff as the 70th
percentile of each batch. On test that gives precision 0.551 against 0.308 for
random 30% — same number of calls, almost twice the hit rate.

### 5. Where it breaks

The report has calibration, permutation importance, slices by month, job, age
and number of `unknown` fields, plus the worst false positives and false
negatives. Short version: the model is useful in the 2010 regime and much weaker
in 2008–2009. Time is doing a lot of the work, and I say that out loud instead
of hiding it.

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
├── LEARN.md           # my beginner notes (RU)
├── ASSUMPTIONS.md     # decisions and trade-offs (RU)
├── make.py            # task runner for Windows
└── Makefile
```

## Run it

```bash
uv sync
uv run python -m bank_marketing.cli all
```

On Windows without GNU make:

```powershell
uv run python make.py repro
```

It downloads the dataset (~1 MB, a zip inside a zip — yes, really), trains the
models and writes `reports/report.md`, `reports/metrics.json` and
`models/model.joblib`.

## API

```bash
uv run uvicorn bank_marketing.serve:app --host 127.0.0.1 --port 8000
```

**Live demo:** not deployed yet. I'll drop the link here when it's up.

The service takes one raw client row and returns `probability`, `threshold`,
`prediction` and `model_version`. `duration` is rejected by the request schema,
so the leak can't sneak in through the API either.

## What I learned

* A random split can make you feel like a genius. It's lying to you.
* The strongest feature is often the one you won't have in production.
* A threshold is not a business decision. A budget is.
* If two windows have different positive rates, compare precision at the same
  volume — not F1 at different volumes.
* Writing the test that blocks the leak was more valuable than another 0.01 of
  PR-AUC.

## What's next

* Walk-forward backtest month by month through 2010.
* Bootstrap confidence intervals for precision@budget.
* A fairness pass over age and job slices.
* A small EDA notebook, because I like looking at data before modelling it.

## License

MIT — see [LICENSE](LICENSE).
