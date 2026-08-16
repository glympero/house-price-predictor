.PHONY: setup data train evaluate diagnostics serve test lint slides package docker docker-run

setup:            ## Create venv and install all dependency groups
	uv sync --all-groups

data:             ## Download and cache the raw dataset
	uv run python -m house_prices.data

train:            ## Train candidate models, select winner, persist artifact
	uv run python -m house_prices.train

evaluate:         ## Evaluate persisted model on the holdout set
	uv run python -m house_prices.evaluate

diagnostics:        ## Reproduce post-selection nested-CV and holdout sensitivity evidence
	uv run python scripts/post_selection_diagnostics.py

serve:            ## Run the API + demo UI locally
	uv run uvicorn house_prices.api.main:app --reload --port 8000

test:             ## Run the test suite
	uv run pytest

lint:             ## Lint and format-check
	uv run ruff check .
	uv run ruff format --check .

docker:           ## Build the serving image (trains the model during the build)
	docker build -t house-price-predictor .

docker-run:       ## Run the serving image on port 8000
	docker run --rm -p 8000:8000 house-price-predictor

slides: train evaluate  ## Rebuild evidence and the presentation deck
	uv run --group slides python presentation/build_slides.py

package: slides   ## Build a source-and-presentation submission zip
	uv run python scripts/package.py
