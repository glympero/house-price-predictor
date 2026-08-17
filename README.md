<!-- @format -->

# House Price Predictor

End-to-end regression solution for an analyst who needs a consistent estimate of
house price per unit area from age, accessibility and location. The repository
covers source validation, training-only EDA, model comparison and bounded tuning,
a protected final evaluation, a FastAPI service, demo UI, Docker image, CI checks,
operating notes and an interview deck.

The selected model is a boundedly tuned histogram gradient-boosting regressor. It
won on location-grouped cross-validation; linear regression remains the
interpretable benchmark and the learning reference, not the model selected because
it was more familiar.

Deeper write-ups live in `docs/`:

| document                                | purpose                                                       |
| --------------------------------------- | ------------------------------------------------------------- |
| [architecture.md](docs/architecture.md) | components, dependency direction, training and request flows  |
| [decisions.md](docs/decisions.md)       | dataset, validation, model, feature and uncertainty decisions |
| [monitoring.md](docs/monitoring.md)     | production signals, thresholds and retraining response        |

## Dataset and source decision

The project uses the [UCI Real Estate Valuation dataset](https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set):
414 transactions from Sindian District, New Taipei City, Taiwan, recorded in
2012-2013.

The assignment names a "Real Estate Price Prediction Dataset" without a link.
Choosing the Kaggle dataset with that matching name was a reasonable way to resolve
the ambiguity: it is a republication of these records. For the reproducible
implementation, the code downloads from UCI, the original publisher, without an
account. The exact expected XLSX is pinned by SHA-256:

```text
597d72fcc6c0539e6035a033ddb387db48fff3fb1f3c98fee31fe081c64a9059
```

Both a cached file and a new download are rejected if the checksum differs. This
makes the source assumption explicit and prevents a silently changed file from
altering the evidence.

The target is already price **per unit area**, so the dataset has no floor-area
feature. It provides coordinates, MRT distance and nearby convenience stores. If
the brief intended a different dataset containing size, the same workflow would
apply, but the input schema and trained artifact would have to change.

## Quick start

### Docker

```bash
docker compose up --build
```

Open <http://localhost:8000>. The image downloads the pinned dataset and trains
during the build, so the first build needs network access; the running service does
not.

### Local development

Requires [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-groups
uv run python -m ipykernel install --user --name house-prices --display-name "Python (house-prices)"
```

The second command registers the project virtual environment as the
`Python (house-prices)` Jupyter kernel requested by all three notebooks. Run it
once after cloning, after moving the repository, or after recreating `.venv`.
It works on Windows, macOS and Linux; the registration made on one computer is
local to that computer and is deliberately not committed.

Open the notebooks from the project environment with:

```bash
uv run jupyter lab
```

For the application and repository checks:

```bash
uv run python -m house_prices.train
uv run python -m house_prices.evaluate
uv run uvicorn house_prices.api.main:app
uv run python -m pytest
uv run ruff check .
uv run ruff format --check .
```

The artifact is a generated build output and is not committed. Run training before
starting the API on a fresh clone. Equivalent `make setup`, `make notebook`,
`make train`, `make evaluate`, `make test`, `make lint`, `make slides` and
`make package` targets are provided.
`make package` creates `dist/house-price-predictor-submission.zip` and excludes
caches, local data, models, environment files and repository metadata.

## Project structure

```text
src/house_prices/
  config.py          paths, seed, split and checksum constants
  data.py            download, cache, schema and checksum validation
  features.py        leak-safe feature transformer for linear candidates
  train.py           grouped split/CV, bounded search, diagnostics, persistence
  evaluate.py        protected-holdout metrics, bootstrap interval and figures
  package.py         clean submission archive
  api/               FastAPI lifecycle, schemas and endpoints
notebooks/
  01_eda.ipynb       detailed EDA on training rows only
  02_modeling.ipynb  reproducible comparison, selection and final evaluation
  03_gradient_descent_reference.ipynb
                     manual one-feature fit checked against scikit-learn
tests/               unit, integration, API, UI and packaging checks
ui/index.html        single-file analyst demo
docs/                architecture, decisions, monitoring and generated figures
presentation/        reproducible deck generator and slides.pptx
```

## Evidence protocol

This is a supervised batch regression task: historical rows contain inputs and a
continuous target, and the application predicts that target for a new property.

The protocol is fixed in code:

1. Split 414 rows into 331 training rows and 83 protected holdout rows.
2. Keep every exact latitude/longitude pair in one partition. This yields 207
   training locations, 52 holdout locations and zero coordinate overlap.
3. Restrict detailed, target-aware EDA and every feature/model decision to the 331
   training rows.
4. Compare candidates on the same shuffled 5-fold `GroupKFold`, again grouping
   exact coordinates.
5. Run small, predefined grids for Ridge, random forest and histogram gradient
   boosting. The grids and winning parameters are stored in `metadata.json`.
6. Select the non-baseline candidate with the lowest mean grouped-CV RMSE, refit it
   on all 331 training rows, and freeze it.
7. Score that exact artifact once on the protected 83-row, 52-location holdout.

This is an approximately 80/20 split. Grouping makes it slightly harder and more
realistic than a random row split: repeated transactions at the same coordinates
cannot make validation look easier by appearing on both sides.

### Model comparison

All values below come from the same grouped training folds. Training RMSE is shown
to expose the generalization gap rather than rewarding a model for fitting its own
rows.

| candidate                       | train RMSE | grouped-CV RMSE | grouped-CV MAE | reason tested                                   |
| ------------------------------- | ---------: | --------------: | -------------: | ----------------------------------------------- |
| Mean baseline                   |      12.89 |           12.90 |          10.41 | minimum useful bar                              |
| Raw linear regression           |       8.27 |            8.53 |           6.36 | simplest interpretable model                    |
| Engineered linear regression    |       7.36 |            7.74 |           5.66 | log-distance and age curve hypotheses           |
| Ridge, alpha 10                 |       7.49 |            7.64 |           5.45 | shrink correlated location coefficients         |
| Random forest                   |       4.54 |            7.17 |           5.11 | nonlinear interactions                          |
| **Histogram gradient boosting** |   **4.36** |        **6.94** |       **5.07** | second nonlinear family; lowest grouped-CV RMSE |

Selected parameters are `learning_rate=0.10`, `max_leaf_nodes=7`,
`min_samples_leaf=20`, and `l2_regularization=0`. The 4.36-to-6.94 training/CV
gap is an overfitting diagnostic and is disclosed, even though this candidate has
the best validation result.

#### Selection caveat: the tuned scores carry search optimism

The grouped-CV column above is valid for the predefined development-time selection
rule, but it is not an unbiased estimate of each selected tuning procedure's future
error.

Each tuned candidate is scored at its best grid cell, chosen on the same folds
that produced the score. A larger search has more opportunities to fit fold-specific
noise, although grid cells are correlated and the nominal cell count is not a count
of independent attempts. This is selection optimism inside development data, not
protected-holdout leakage.

A deterministic grouped nested-CV diagnostic estimates its size. The grid search
runs inside each outer training fold and the score comes from an outer fold that
search never saw. The protected holdout is not involved.

| candidate                    | grid cells | reported CV | nested CV | optimism |
| ---------------------------- | ---------: | ----------: | --------: | -------: |
| Mean baseline                |          1 |       12.90 |     12.90 |    +0.00 |
| Raw linear regression        |          1 |        8.53 |      8.53 |    +0.00 |
| Engineered linear regression |          1 |        7.74 |      7.74 |    +0.00 |
| Ridge                        |          6 |        7.64 |      7.74 |    +0.11 |
| Random forest                |         18 |        7.17 |      7.26 |    +0.09 |
| Histogram gradient boosting  |         36 |        6.94 |      7.40 |    +0.46 |

The untuned candidates are unchanged by construction. The 36-cell boosting search
has the largest observed gap, but the gaps are not strictly monotonic in grid size:
Ridge's +0.11 is slightly larger than the forest's +0.09. Grid breadth is therefore
a plausible contributor, not a proven single cause. Ridge's apparent edge over the
engineered linear model also disappears at the displayed precision.

**In this one five-fold nested run, the point ranking changes.** Random forest has
mean outer-fold RMSE 7.26 and histogram boosting 7.40. The difference is only 0.14;
the forest is lower on three outer folds and boosting on two. This does not establish
that random forest is intrinsically better. It shows that the original reported lead
from 6.94 to 7.17 is not robust enough to separate the nonlinear families.

The shipped model was not changed. This diagnostic was run after the original rule
had selected the model and after its holdout result was known, so it is recorded as
post-selection sensitivity analysis rather than used for retroactive reselection. A
fresh iteration should predeclare the grouped nested comparison, select a final
procedure, and reserve new protected evidence for its final estimate.

The exact folds, per-fold parameters and sensitivity calculations are persisted in
the [post-selection diagnostic JSON](docs/post_selection_diagnostics.json) and can be
reproduced with [`scripts/post_selection_diagnostics.py`](scripts/post_selection_diagnostics.py)
or `make diagnostics`.

### Feature engineering and scaling

Training-only EDA showed that raw MRT distance has a curved relationship with
price. For the linear benchmark, `log10(MRT distance)` improves grouped-CV RMSE
from 8.53 to 7.76. Adding age squared changes it only from 7.76 to 7.74, so that
second feature has weak evidence. Ridge then improves the engineered linear model
to 7.64. A distance-from-centre feature was tested and rejected because its 0.02
RMSE change was negligible beside fold variation.

`StandardScaler` is fitted **inside** each linear/Ridge pipeline and therefore only
on each fold's training portion. Scaling is important for regularization and stable
coefficient optimization. It does not make a tree ensemble better: trees split on
ordered thresholds, so monotonic log/square transforms add no split information.
The selected boosting pipeline consequently uses the five raw serving inputs and no
scaler; its transform ablation produced identical scores.

The correlations in the EDA are Pearson associations, not causal effects. For
example, price correlates negatively with MRT distance and positively with stores
and latitude, while MRT distance and longitude also correlate strongly with each
other. That overlap is why a coefficient cannot be interpreted as the isolated
effect of changing one location variable. Correlation guides hypotheses; grouped
validation decides whether a transformation predicts better.

### Protected holdout result

After model and parameters were frozen:

| metric                                       |        result |
| -------------------------------------------- | ------------: |
| RMSE                                         |         10.48 |
| 2,000-sample bootstrap 95% interval for RMSE | 5.54 to 16.15 |
| MAE                                          |          5.90 |
| R²                                           |         0.569 |
| observed 90% interval coverage               |         91.6% |

RMSE and MAE use the target unit, where one unit is 10,000 TWD per ping. RMSE
10.48 is therefore a large-error-weighted scale of **104,800 TWD per ping**; it is
not the average miss. MAE 5.90 says the average absolute miss is **59,000 TWD per
ping**. R² 0.569 is not 56.9% accuracy: it means 56.9% less squared error than
predicting the protected-set mean for every row.

The final RMSE is worse than grouped CV. One rare holdout sale has an actual value
of 117.5 (1,175,000 TWD per ping) and a prediction near 40.1 (401,000 TWD per
ping), an absolute error around 77.4 (774,000 TWD per ping). This makes the RMSE/MAE
difference large and exposes weak performance at the expensive tail. Training-only
out-of-fold diagnostics already show the same pattern: high-target RMSE is 9.32
versus 5.58 and 5.40 in the lower bands.

That single row dominates the estimate:

| holdout RMSE                | value |
| --------------------------- | ----: |
| all 83 rows                 | 10.48 |
| excluding the 117.5 sale    |  6.17 |
| excluding the worst 3 rows  |  5.77 |

One row of 83 accounts for **65.7% of the total squared error** (about 66%). Without
it, the holdout sensitivity RMSE is lower than the reported grouped-CV result. This
is not an argument that the model is fine and it is not a replacement score. It is a
principal reason the row-bootstrap interval runs from 5.54 to 16.15; without that
row, the same sensitivity calculation is approximately 5.30 to 7.04. More
representative expensive-property data is needed to estimate tail performance
precisely.

Two further facts belong with it. The training rows top out at 78.3, so this sale
is above anything the model saw in the target. And under the earlier random-row
split that transaction fell in training while its holdout topped out at 63.3. Those
facts are consistent with the coordinate-grouped holdout being harder and more
honest. They do not causally decompose the change from the historical 6.8 result,
because the partition and selected model both changed.

The holdout result is reported as a limitation; it is not used to reopen tuning or
choose another model. With only 83 rows, the wide bootstrap interval is the honest
statement of uncertainty.

## Gradient-descent implementation check

[`03_gradient_descent_reference.ipynb`](notebooks/03_gradient_descent_reference.ipynb)
fits the same standardized log-MRT one-feature problem in two ways on the 331
training rows. Manual gradient descent returns `w=-9.2234, b=37.6697`; scikit-learn
returns `w=-9.2234, b=37.6698`. That agreement checks the hand-written cost,
gradients and update loop against a trusted least-squares implementation. The
figure also shows how the learning rate changes convergence on this objective.

This is a deliberately simplified implementation check, not another model-selection
result. The complete candidates are compared with grouped cross-validation in
`02_modeling.ipynb`; manual gradient descent did not train the deployed histogram
boosting model.

## API and analyst demo

The demo is at `/` and OpenAPI documentation at `/docs`.

| endpoint          | purpose                                                     |
| ----------------- | ----------------------------------------------------------- |
| `POST /predict`   | point estimate, residual-based interval and warnings        |
| `GET /health`     | liveness only; succeeds if the process is running           |
| `GET /ready`      | readiness; returns 503 until the model is loaded            |
| `GET /model/info` | selected model, parameters, evidence protocol and data hash |

Example:

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"house_age_years":10,"mrt_distance_m":250,"n_convenience_stores":6,"latitude":24.975,"longitude":121.540}'
```

The price unit is **10,000 New Taiwan Dollars per ping**; one ping is about 3.3
m². The API returns the unit explicitly and the UI also provides a per-square-metre
conversion. It predicts price density, not total property price.

Inputs outside the ranges fitted on the 331 training rows receive HTTP 422 with an
`outside_model_support` error. Values are not silently clipped.

### Prediction interval

The 5th and 95th percentiles of the selected model's location-grouped out-of-fold
residuals are stored as offsets (currently about -9.16 and +11.91). This keeps the
point and interval tied to one model. The protected holdout's 91.6% observed
coverage is a finite-sample final diagnostic, not a guarantee for an individual
property or a different market. The fixed-width interval is a known limitation.

## Deployment and verification

The locked Python 3.12 slim image trains from the checksum-verified source, runs as
a non-root user and checks `/ready`. Runtime model and data paths are configurable
with `HOUSE_PRICES_MODELS_DIR`, `HOUSE_PRICES_DATA_DIR`,
`HOUSE_PRICES_REPORTS_DIR`, `HOUSE_PRICES_UI_FILE`, or the common
`HOUSE_PRICES_ROOT`.

CI performs Ruff checks, training, the complete test suite, an image build, a
readiness check and a real prediction. A committed `.joblib` could silently drift
from the code that created it, so the artifact is rebuilt instead and records code,
data and parameter evidence in metadata.

## Limitations

- Only 414 historical transactions from one Taiwanese district and period.
- The 83-row protected holdout gives an imprecise final estimate and contains one
  influential expensive sale that is 65.7% of the total squared error.
- Candidates were compared at their best grid cell on shared folds, so the tuned
  scores carry search optimism. One post-selection nested run gives random forest a
  0.14 lower mean RMSE, but the two nonlinear families are not separated by this
  evidence.
- No mechanism for current market inflation or temporal drift; transaction date is
  excluded because its 2012-2013 meaning does not transfer safely.
- Exact-coordinate grouping is a practical leakage guard, not proof of geographic
  generalization to another district.
- Prediction intervals have fixed width and observed, not guaranteed, coverage.
- This prototype has documented monitoring and rollback inputs but no live feedback
  store, model registry, authentication, rate limiting or multi-version rollout.

## Use of AI assistance

AI coding assistants supported scaffolding, implementation, tests, documentation,
Docker/CI review and critique of the validation design. They also helped explain
data-leakage and cross-validation concepts and identify claims that needed stronger
support.

The final workflow, dataset assumption, protected-split policy, bounded search,
model-selection rule, serving behaviour and limitations were checked through
executed notebooks and automated tests. Reported values come from the persisted
metadata and evaluation files, the raw dataset is checksum-verified, and the tests
exercise the split, folds, API and packaging path.
