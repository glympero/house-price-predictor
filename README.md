# House Price Predictor

End-to-end regression solution that predicts real estate prices from location,
age, and accessibility features — covering data acquisition, exploratory
analysis, model training and evaluation, and a production-style serving layer
(FastAPI + demo UI).

> **Status:** work in progress. Full documentation (setup, architecture,
> assumptions, results) will land as the project develops.

## Dataset

The project uses the [UCI Real Estate Valuation dataset](https://archive.ics.uci.edu/dataset/477/real+estate+valuation+data+set)
(also published on Kaggle as "Real Estate Price Prediction"): 414 housing
transactions from Sindian District, New Taipei City, Taiwan, recorded in
2012–2013. The raw file is fetched on first run by `house_prices.data` and
cached under `data/`, which is why you won't find it in the repo — anyone
cloning the project gets the same data by running the code.

## Quick start

```bash
uv sync            # create venv + install dependencies (requires uv)
uv run pytest      # run tests
```
