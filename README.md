# House Price Predictor

End-to-end regression solution that estimates price per unit area from location,
age, and accessibility features. It covers data acquisition, exploratory
analysis, model training and evaluation, and a production-style serving layer
(FastAPI and a demo UI).

The shipped model is a linear regression on features derived from the
exploratory analysis. A random forest scored slightly better on cross-validation
and was not selected; the reasoning is in "Modeling decisions" below.

> **Status:** modeling and serving are complete. Containerization, CI, and the
> presentation are the remaining pieces.

## Dataset

The project uses the [UCI Real Estate Valuation dataset](https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set):
414 housing transactions from Sindian District, New Taipei City, Taiwan,
recorded in 2012 and 2013.

The dataset is not stored in the repo. The first time you run the code,
`house_prices.data` downloads it from UCI and caches it under `data/`.

### Assumption: which dataset this is

The brief specifies "Real Estate Price Prediction Dataset" without a link, so
identifying it is an assumption and is recorded here rather than left implicit.

There is a Kaggle dataset published under exactly that title, and it is a
re-publication of the UCI "Real Estate Valuation" dataset used here. The brief
names the other case's data the same way ("Telco Customer Churn Dataset"),
referring to a well-known public dataset by its common title, so the same
pattern applies. UCI was chosen as the source because it is the original
publisher and can be downloaded without an account, which keeps the project
reproducible for anyone cloning it.

Two consequences of that identification are worth stating up front.

The data is Taiwanese, so it uses local conventions: prices in New Taiwan
Dollars and floor area in ping. See the price unit note below.

The brief describes predicting price "based on features like location, size,
and amenities". This dataset provides location (coordinates and distance to the
nearest MRT station) and amenities (nearby convenience stores), but no size
feature, because the target is already expressed per unit of floor area. If a
different dataset was intended, the modelling approach would carry over but
floor area would likely become the strongest single predictor.

## Quick start

Requires [uv](https://docs.astral.sh/uv/). These commands work on any platform:

```bash
uv sync                                   # create venv and install dependencies
uv run python -m house_prices.train       # download data, compare models, save the artifact
uv run python -m house_prices.evaluate    # score the selected model on the holdout set
uv run uvicorn house_prices.api.main:app  # serve the API and demo UI on port 8000
uv run pytest                             # run the tests
```

There is a `Makefile` wrapping the same commands (`make train`, `make serve`,
`make test`, `make lint`). It is a convenience for machines that have `make`,
which does not include a default Windows install. Nothing in the project
depends on it.

## Modeling decisions

The dataset holds 414 transactions: 331 were used to fit the model and 83 were
held out and used once, at the end. Every step below was measured with the same
5-fold cross-validation on the 331 training rows.

| Step | CV RMSE | Why |
|---|---:|---|
| Mean baseline | 13.642 | minimum useful bar |
| Raw linear regression | 9.161 | simple interpretable benchmark |
| + log(MRT distance) | 8.435 | EDA showed a curved distance effect |
| + age² | 8.276 | EDA showed a U-shaped age effect |
| Ridge | 8.269 | test regularization on the engineered features |
| Random Forest | 7.995 | nonlinear benchmark |
| Gradient Boosting | 8.147 | second nonlinear benchmark |

1. The mean baseline established the floor.
2. Raw linear regression gave the interpretable benchmark.
3. `log(MRT distance)` improved CV RMSE after the EDA showed a curved distance
   effect.
4. `age²` improved it further after the EDA showed a U-shaped age relationship.
5. Ridge and Lasso added little predictive value. Lasso set `longitude` to zero,
   which is consistent with the redundancy between location features seen in the
   EDA.
6. A distance-from-center feature was tested and rejected: 8.276 to 8.278 for the
   linear model, and 7.995 to 8.038 for the forest. It is a deterministic
   function of latitude and longitude, which the model already has.
7. Random Forest achieved the best mean CV RMSE. The engineered linear model was
   selected for this prototype because its performance was close and it is
   simpler and more interpretable.
8. `transaction_date` slightly improved validation performance, by about 0.16
   RMSE, but was excluded because its historical meaning does not transfer safely
   to present-day inference.

The largest gains came from the EDA-driven feature engineering, not from
regularization or from changing model family.

Holdout result, reported after selection and not used to compare models:
RMSE 6.80, MAE 4.84, R² 0.724.

### Findings from building the service

- API boundary testing exposed unsafe polynomial extrapolation from the `age²`
  term.
- Serving now rejects inputs outside the fitted model support instead of
  returning an implausible number.
- Uncertainty was simplified to out-of-fold residual quantiles from the selected
  model, replacing two separate quantile models.
- The 2012 to 2013 historical limitation is shown to users in the demo UI.

## API

Start the service with the `uvicorn` command above. The demo page is at `/` and
the generated OpenAPI documentation at `/docs`.

| endpoint           | purpose                                                    |
| ------------------ | ---------------------------------------------------------- |
| `POST /predict`    | price estimate, prediction interval, and any input warnings |
| `GET /health`      | liveness, and whether a model is loaded                     |
| `GET /model/info`  | which model is serving, its metrics, and its data hash      |

Linux and macOS:

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"house_age_years":10,"mrt_distance_m":250,"n_convenience_stores":6,"latitude":24.975,"longitude":121.540}'
```

Windows PowerShell. Use `curl.exe`, because `curl` there is an alias for
`Invoke-WebRequest` and takes different arguments:

```powershell
curl.exe -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{"house_age_years":10,"mrt_distance_m":250,"n_convenience_stores":6,"latitude":24.975,"longitude":121.540}'
```

### A note on the price unit

The target is not the price of a house. It is a price per unit of floor area,
and the unit is the one the dataset uses: **10,000 New Taiwan Dollars per
ping**. A ping (坪) is the traditional property measure in Taiwan and equals
about 3.3 m².

A predicted value of `47.2` therefore reads as:

| interpretation                        | value                    |
| ------------------------------------- | ------------------------ |
| in the dataset's unit                 | 47.2                     |
| New Taiwan Dollars per ping           | 472,000 TWD              |
| New Taiwan Dollars per square metre   | about 143,000 TWD        |
| a 30 ping apartment (about 99 m²)     | about 14.2 million TWD   |

The unit is kept as the dataset defines it, because that is what the model
predicts and what every metric in this repository is expressed in. The API
returns it with an explicit `price_unit` field, and the demo UI converts it to
per square metre so a reader unfamiliar with the unit is not left guessing.

This also explains why there is no feature for the size of the property. The
target has already been divided by floor area, so the model estimates how
expensive a square of space is at a given location, age and accessibility,
rather than what a particular house costs.

### Inputs outside the model's support

The service does not extrapolate. If any input falls outside the range the model
was fitted on, the request is refused with 422 and an `outside_model_support`
error naming each offending field and its supported range. The input is not
clipped to the nearest supported value, because that would answer a question
about a different property.

Values that are impossible, or outside the district the model describes, are
rejected by schema validation before the model is reached.

### The prediction interval

The range is derived from the residual distribution of the selected model,
measured out of fold on the training data. The 5th and 95th percentiles of those
residuals are stored as offsets and applied to every prediction. They are not
symmetric (-10.6 and +11.8), so the range is not centred on the estimate.

The response reports the nominal 90% target and the coverage measured on the
holdout set, which was 92%. That measurement is **exploratory**: the same 83-row
holdout was inspected during development, so it is not an untouched sample and
the figure is not a validated coverage guarantee. Establishing one would require
a separate set that has never been examined.

Known limitation: the offsets are constant, so every property receives the same
interval width, and the range does not widen for inputs the model finds unusual.
