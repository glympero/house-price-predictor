"""Tests for path configuration.

The paths default to the repository layout, which is what every other test and
the notebooks rely on. They can also be pointed elsewhere, which is what lets a
container serve a model that was not built into it.
"""

import importlib

import pytest

from house_prices import config


@pytest.fixture
def reloaded_config(monkeypatch):
    """Reload config with patched environment, then restore the real one."""

    def _reload(**env):
        for name, value in env.items():
            monkeypatch.setenv(name, value)
        return importlib.reload(config)

    yield _reload
    monkeypatch.undo()
    importlib.reload(config)


def test_paths_default_to_the_repository_layout():
    assert config.DATA_DIR == config.REPO_ROOT / "data"
    assert config.MODELS_DIR == config.REPO_ROOT / "models"
    assert config.REPORTS_DIR == config.REPO_ROOT / "docs" / "figures"
    assert config.UI_FILE == config.REPO_ROOT / "ui" / "index.html"


def test_each_path_can_be_overridden(reloaded_config, tmp_path):
    models = tmp_path / "served-models"
    ui = tmp_path / "page.html"

    reloaded = reloaded_config(
        HOUSE_PRICES_MODELS_DIR=str(models),
        HOUSE_PRICES_UI_FILE=str(ui),
    )

    assert models.resolve() == reloaded.MODELS_DIR
    assert ui.resolve() == reloaded.UI_FILE
    # Paths that were not overridden still follow the repository root.
    assert reloaded.DATA_DIR == reloaded.REPO_ROOT / "data"


def test_overriding_the_root_moves_the_paths_that_depend_on_it(reloaded_config, tmp_path):
    reloaded = reloaded_config(HOUSE_PRICES_ROOT=str(tmp_path))

    assert tmp_path.resolve() == reloaded.REPO_ROOT
    assert tmp_path.resolve() / "data" == reloaded.DATA_DIR
    assert tmp_path.resolve() / "data" / "raw" == reloaded.RAW_DATA_DIR
    assert tmp_path.resolve() / "models" == reloaded.MODELS_DIR


def test_a_specific_path_wins_over_the_root(reloaded_config, tmp_path):
    models = tmp_path / "elsewhere"

    reloaded = reloaded_config(
        HOUSE_PRICES_ROOT=str(tmp_path),
        HOUSE_PRICES_MODELS_DIR=str(models),
    )

    assert models.resolve() == reloaded.MODELS_DIR
