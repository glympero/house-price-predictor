.PHONY: setup data train evaluate serve test lint slides package

setup:            ## Create venv and install all dependency groups
	uv sync --all-groups

data:             ## Download and cache the raw dataset
	uv run python -m house_prices.data

train:            ## Train candidate models, select winner, persist artifact
	uv run python -m house_prices.train

evaluate:         ## Evaluate persisted model on the holdout set
	uv run python -m house_prices.evaluate

serve:            ## Run the API + demo UI locally
	uv run uvicorn house_prices.api.main:app --reload --port 8000

test:             ## Run the test suite
	uv run pytest

lint:             ## Lint and format-check
	uv run ruff check .
	uv run ruff format --check .

slides:           ## Build the presentation deck
	uv run --group slides python presentation/build_slides.py

package:          ## Build the shareable zip
	uv run python scripts/package.py
