# House Price Predictor

End-to-end regression solution that predicts real estate prices from location,
age, and accessibility features — covering data acquisition, exploratory
analysis, model training and evaluation, and a production-style serving layer
(FastAPI + demo UI).

> **Status:** work in progress. Full documentation (setup, architecture,
> assumptions, results) will land as the project develops.

## Dataset

[UCI Real Estate Valuation](https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set)
(also published on Kaggle as "Real Estate Price Prediction") — 414 transactions
from Sindian District, New Taipei City, Taiwan. Downloaded automatically; not
committed to the repo.

## Quick start

```bash
uv sync            # create venv + install dependencies (requires uv)
uv run pytest      # run tests
```
