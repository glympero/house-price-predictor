# syntax=docker/dockerfile:1

# Serving image for the house price API.
#
# Build:  docker build -t house-price-predictor .
# Run:    docker run --rm -p 8000:8000 house-price-predictor
#
# The build needs network access, because it downloads the UCI dataset and
# trains the model. The running container does not.

FROM python:3.12-slim

# uv is pinned. An unpinned installer is the one dependency that can change the
# build without anything in this repository changing.
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /bin/

# Copy instead of hardlink, because the uv cache below is a separate mount.
ENV UV_LINK_MODE=copy
# Use the interpreter already in the base image.
ENV UV_PYTHON_DOWNLOADS=0
ENV PYTHONUNBUFFERED=1

# The user is created before anything is installed, so every file is written
# with the right owner from the start. Doing it the other way round and running
# `chown -R` at the end rewrites the whole virtual environment into a second
# layer, which silently doubles the size of the image.
RUN useradd --create-home --uid 10001 service \
    && mkdir -p /app && chown service:service /app
WORKDIR /app
USER service

# Dependencies go in their own layer, installed from the lockfile before any
# source is copied. Editing a Python file then rebuilds in seconds instead of
# reinstalling scikit-learn and pandas.
RUN --mount=type=cache,target=/home/service/.cache/uv,uid=10001,gid=10001 \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=README.md,target=README.md \
    uv sync --locked --no-install-project --no-dev

COPY --chown=service:service pyproject.toml uv.lock README.md ./
COPY --chown=service:service src/ src/
COPY --chown=service:service ui/ ui/

RUN --mount=type=cache,target=/home/service/.cache/uv,uid=10001,gid=10001 \
    uv sync --locked --no-dev

# Use the environment directly from here on. `uv run` re-syncs before every
# command, which would pull the dev dependencies back into the image and make
# the container depend on uv at startup.
ENV PATH="/app/.venv/bin:$PATH"

# The model is a build output, not source, so the repository does not carry it.
# Training here means the image can only ever contain a model built from the
# exact code in the same image, which a committed artifact cannot guarantee.
# Cost: the build needs to reach UCI, and adds about seven seconds.
RUN python -m house_prices.train

# The holdout score is part of what the service reports about itself. Without
# this, /model/info returns null for the holdout metrics and the demo page
# shows "n/a" where the coverage figure belongs.
RUN python -m house_prices.evaluate

# The image keeps the repository layout, so the defaults in config.py apply and
# nothing has to be configured to run it. To serve a model that was not built
# into the image, mount one and point HOUSE_PRICES_MODELS_DIR at it:
#
#   docker run -v ./models:/srv/models -e HOUSE_PRICES_MODELS_DIR=/srv/models ...

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD ["python", "-c", \
         "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready')"]

CMD ["uvicorn", "house_prices.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
