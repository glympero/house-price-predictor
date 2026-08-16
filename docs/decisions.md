# Decisions and limitations

This document records the decisions that materially change the evidence or serving
behaviour. It is meant to make the project discussable in an interview, including
where an earlier prototype was revised.

## 1. Frame the problem before choosing an algorithm

This is a supervised, offline regression task. Each historical transaction has
known inputs and a continuous target; the service estimates that target for one new
property. The intended user is an analyst who needs a repeatable benchmark, not an
automated valuation decision.

The business objective is therefore broader than a single score: useful predictive
accuracy, a visible uncertainty caveat, reproducibility, and a service whose failure
modes are explicit. RMSE is the primary selection metric and MAE/R² are secondary
diagnostics.

## 2. The matching Kaggle dataset was a reasonable assumption; UCI is the source

The brief gives a dataset name but no link. The Kaggle "Real Estate Price
Prediction" data matching that name republishes the UCI Real Estate Valuation
records, so selecting it was not bad judgement. The ambiguity is documented because
another dataset would change both features and target.

The implementation downloads the original UCI XLSX because it is authoritative and
does not require credentials. It verifies SHA-256
`597d72fcc6c0539e6035a033ddb387db48fff3fb1f3c98fee31fe081c64a9059`
before parsing. A matching title alone is not the reproducibility control; the
source, schema, row count and digest together are.

The target is price per unit area, which explains the absence of a size feature.
This is a limitation relative to any interpretation of the brief that expects total
house price.

## 3. Split before detailed EDA, and split locations rather than rows

The first notebook may load all 414 rows to validate columns and row count. Once the
schema is known, the code separates 331 training rows from an 83-row protected
holdout. All target-aware plots, correlations, feature choices, model comparisons and
tuning use only the training side.

A random row split would put repeated transactions at identical coordinates on both
sides. The implemented `GroupShuffleSplit` keeps coordinate pairs together and
selects, without looking at the target, the seeded draw closest to 20% of rows. The
result is 207 training locations and 52 holdout locations with zero overlap.

The same grouping is used in shuffled 5-fold `GroupKFold`. This estimates transfer
to unseen exact locations within the historical district. It is more demanding than
the original random-row diagnostic and explains why old and current metrics should
not be mixed.

## 4. Detailed EDA stays on training rows

The original notebook explored all rows, an approach seen in some educational
end-to-end examples. That is acceptable for schema/quality inspection but not for
target-aware exploration if the same rows will later be called an untouched test
set. The revised notebook shows full-data shape only, then computes distributions,
correlations, transformations and outlier views on 331 training rows.

Pearson correlations are treated as hypothesis generators:

- MRT distance has a strong negative association with price, strengthened by a log
  transform for a linear equation.
- Stores and coordinates carry location signal.
- Location variables also correlate with one another, so individual coefficients
  are unstable descriptions of overlapping signals.
- A low correlation does not prove a feature is useless to a nonlinear model, and a
  high correlation does not establish causation.

The prediction experiment, not the correlation table, decides whether a feature is
retained.

## 5. Linear regression is a benchmark, not a familiarity-based choice

Linear regression belongs early in the sequence because it is inexpensive,
inspectable and provides a strong diagnostic baseline. It also makes feature scaling,
coefficients, cost and gradient descent easy to explain. Those are valid reasons to
include it, but familiarity with a course is not a valid reason to ship it over a
candidate with better validation evidence.

The predefined selection rule is simple: exclude the mean baseline and choose the
lowest mean location-grouped CV RMSE. On the current training data:

| model | train RMSE | grouped-CV RMSE | grouped-CV MAE |
|---|---:|---:|---:|
| raw linear | 8.27 | 8.53 | 6.36 |
| engineered linear | 7.36 | 7.74 | 5.66 |
| Ridge | 7.49 | 7.64 | 5.45 |
| random forest | 4.54 | 7.17 | 5.11 |
| **histogram gradient boosting** | **4.36** | **6.94** | **5.07** |

Histogram gradient boosting therefore ships. Interpretability is addressed through
training-only error slices, permutation importance with caveats, model metadata and a
linear learning reference; it does not override the primary evidence.

## 6. Hyperparameters are tuned, but the search is bounded

Avoiding all tuning would leave credible performance unexplored. An exhaustive or
adaptive search over hundreds of combinations would be equally hard to defend on
331 rows. The compromise is a small, predefined `GridSearchCV` inside the grouped
training folds:

- Ridge: 6 alpha values from 0.01 to 1,000.
- Random forest: depth `{None, 4, 8}`, minimum leaf `{1, 3, 5}`, and feature fraction
  `{0.7, 1.0}` with 300 trees.
- Histogram gradient boosting: learning rate `{0.03, 0.10}`, leaf nodes
  `{7, 15, 31}`, minimum leaf `{10, 20, 30}`, and L2 `{0, 1}`.

Every configuration uses identical grouped folds and the final holdout is not passed
to `GridSearchCV`. Best parameters, configuration counts, training score, validation
score and fold variation are persisted. The selected boosting configuration has 7
leaf nodes, learning rate 0.10, minimum 20 samples per leaf and L2 0.

This is tuning with an explicit capacity budget, not opposition to tuning.

## 7. Feature engineering and scaling are model-specific

For the linear candidate, log MRT distance reduces grouped-CV RMSE from 8.53 to
7.76. Age squared changes it only from 7.76 to 7.74, weak evidence. A learned
distance-to-centre feature changes 7.741 to 7.721, negligible beside fold variation,
and is rejected.

Linear and Ridge pipelines use `StandardScaler`, fitted inside each training fold.
This prevents leakage and puts Ridge's penalty on comparable numeric scales.

Tree models do not need scaling. Log and square are monotonic over supported inputs,
so they do not add ordering information to tree thresholds; the selected histogram-
boosting ablation produced identical scores. Its pipeline uses raw serving features
and avoids redundant engineered columns.

## 8. Historical transaction date is excluded

Adding `transaction_date` improves grouped-CV RMSE only about 0.12 for the linear
model and 0.04 for the forest. The data spans 2012-2013, so a present-day date is far
outside the fitted period and would not carry the same market meaning. A weak
historical gain does not justify an unsafe production input.

A real deployment needs current sales plus a market/time feature defined at
prediction time and a time-based validation design.

## 9. The expensive sale is kept and exposed

No row is removed merely because its target is large. The protected holdout contains
a 117.5 transaction predicted near 40.1. That roughly 77.4 absolute error drives the
final RMSE to 10.48 while MAE is 5.90. It may be a legitimate rare sale, and deleting
it after seeing the error would contaminate the test.

Training-only grouped OOF error was already higher in the expensive target band
(RMSE 9.32 versus 5.58 and 5.40 in the other bands). The final row confirms a known
weak region. The correct response is to disclose it and seek more high-value data,
not tune against the protected example.

## 10. The holdout is final evidence, not another development set

The selected artifact scores RMSE 10.48, MAE 5.90 and R² 0.569 on 83 rows from 52
unseen coordinate locations. A deterministic 2,000-resample bootstrap gives an RMSE
95% interval of 5.54-16.15.

The gap from grouped-CV RMSE 6.94 is material. With one small holdout and an
influential tail observation, the wide interval is more informative than declaring
the CV estimate wrong or choosing a model after the fact. Any subsequent model
iteration should protect new data or use nested evaluation rather than repeatedly
consulting these 83 rows.

## 11. One point model and residual-based uncertainty

An earlier prototype used separate quantile models for the lower and upper bounds.
That could produce a point estimate and range based on different model opinions. The
current artifact stores location-grouped out-of-fold residual quantiles from the
selected point model and applies them as offsets.

Observed coverage on the protected holdout is 91.6% for a nominal 90% interval. This
is a finite-sample diagnostic, not a per-property guarantee. Fixed residual offsets
also fail to widen for unusual inputs. Conditional/conformal intervals would be a
future improvement when more calibration data exists.

## 12. Refuse unsupported inputs

The API validates general plausibility and then checks each value against support
recorded from the 331 fitting rows. Unsupported requests receive HTTP 422 and name
the offending fields. Silent clipping would answer a different question; unchecked
extrapolation would manufacture confidence where the data provides none.

## 13. Build the model with the code that serves it

The joblib artifact is gitignored. Docker installs the locked Python environment,
copies package/UI source, downloads the checksum-verified data, trains, evaluates,
and starts as a non-root user. `/health` is liveness; `/ready` requires a loaded
artifact and is the container healthcheck.

Building the model in the image prevents an old binary from silently disagreeing
with current feature code. It does require UCI during the build. At production
scale, an approved training job would publish an immutable model digest to a
registry, and the serving image would verify and load that version.

# Limitations and next evidence

- 414 transactions, one district, one historical period.
- An imprecise 83-row final test and weak expensive-tail performance.
- Exact-coordinate grouping does not prove wider geographic transfer.
- No current time/market signal and no realised-sale feedback loop.
- Fixed-width uncertainty intervals.
- No authentication, rate limits, registry, audit store or champion/challenger
  deployment in this interview prototype.

The most valuable next step is more recent, representative transaction data,
especially expensive properties and additional locations. That would justify a
time/geography-aware validation design, conditional uncertainty, and a broader but
still nested tuning search.
