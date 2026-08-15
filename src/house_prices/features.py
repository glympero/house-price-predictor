"""Feature engineering shared by training, evaluation, and inference.

Each transform traces back to a finding in notebooks/01_eda.ipynb. The
transforms are individually switchable so the modeling notebook can measure
what each one contributes instead of asserting it.

- ``log_mrt_distance``: the relationship between distance and price is curved,
  so a linear model fits it poorly on the raw scale. On by default.
- ``age_squared``: price falls with age and rises again after roughly two
  decades. A linear model with one coefficient per feature cannot represent
  that; a squared term lets it curve once. On by default.
- ``include_transaction_date``: the training data covers 2012 and 2013 only, so
  a current date would fall outside the range the model was fitted on and would
  not carry the same meaning as the historical feature. Off by default.

A distance-from-center feature was also tested. Cross-validation showed no
improvement for either model family, so it is not part of the pipeline. The
experiment is kept in notebooks/02_modeling.ipynb.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Turns the raw feature columns into a model-ready feature set."""

    def __init__(
        self,
        *,
        log_mrt_distance: bool = True,
        age_squared: bool = True,
        include_transaction_date: bool = False,
    ):
        self.log_mrt_distance = log_mrt_distance
        self.age_squared = age_squared
        self.include_transaction_date = include_transaction_date

    def fit(self, X: pd.DataFrame, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        if self.include_transaction_date:
            out["transaction_date"] = X["transaction_date"]
        out["house_age_years"] = X["house_age_years"]
        if self.age_squared:
            out["house_age_squared"] = X["house_age_years"] ** 2
        out["n_convenience_stores"] = X["n_convenience_stores"]
        out["latitude"] = X["latitude"]
        out["longitude"] = X["longitude"]
        if self.log_mrt_distance:
            out["log10_mrt_distance"] = np.log10(X["mrt_distance_m"])
        else:
            out["mrt_distance_m"] = X["mrt_distance_m"]
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        names = []
        if self.include_transaction_date:
            names.append("transaction_date")
        names.append("house_age_years")
        if self.age_squared:
            names.append("house_age_squared")
        names += ["n_convenience_stores", "latitude", "longitude"]
        names.append("log10_mrt_distance" if self.log_mrt_distance else "mrt_distance_m")
        return np.asarray(names)


def build_pipeline(model, *, scale: bool, **feature_options) -> Pipeline:
    """Assemble the preprocessing and model pipeline.

    ``scale`` should be True for linear models, whose regularization penalty
    treats all coefficients equally and therefore needs comparable feature
    scales, and False for tree models, whose splits are scale-invariant.
    """
    steps: list[tuple] = [("features", FeatureEngineer(**feature_options))]
    if scale:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)
