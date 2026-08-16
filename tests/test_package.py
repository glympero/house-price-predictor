"""Tests for the submission-packaging boundary."""

import zipfile

from house_prices.package import ARCHIVE_ROOT, build_package


def test_package_includes_source_and_deck_but_excludes_local_artifacts(tmp_path):
    repo = tmp_path / "repo"
    for relative in (
        "README.md",
        "Dockerfile",
        "compose.yaml",
        "pyproject.toml",
        "uv.lock",
        "presentation/slides.pptx",
        "src/house_prices/train.py",
        "tests/test_train.py",
        "models/model.joblib",
        "data/raw/source.xlsx",
        ".venv/installed.txt",
    ):
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"content")

    output = build_package(repo, tmp_path / "submission.zip")
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())

    assert f"{ARCHIVE_ROOT}/README.md" in names
    assert f"{ARCHIVE_ROOT}/presentation/slides.pptx" in names
    assert f"{ARCHIVE_ROOT}/src/house_prices/train.py" in names
    assert not any("/models/" in name for name in names)
    assert not any("/data/" in name for name in names)
    assert not any("/.venv/" in name for name in names)
