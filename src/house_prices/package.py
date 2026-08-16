"""Create the reviewer-facing source-and-presentation ZIP."""

from __future__ import annotations

import zipfile
from pathlib import Path

PACKAGE_NAME = "house-price-predictor-submission.zip"
ARCHIVE_ROOT = "house-price-predictor"

EXCLUDED_DIRECTORIES = {
    ".git",
    ".idea",
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".vscode",
    "__pycache__",
    "data",
    "dist",
    "models",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
EXCLUDED_NAMES = {".coverage", ".DS_Store", ".env", "Thumbs.db"}

REQUIRED_FILES = {
    "README.md",
    "Dockerfile",
    "compose.yaml",
    "pyproject.toml",
    "uv.lock",
    "presentation/slides.pptx",
}


def _included(path: Path, repo_root: Path) -> bool:
    relative = path.relative_to(repo_root)
    if any(part in EXCLUDED_DIRECTORIES for part in relative.parts[:-1]):
        return False
    return path.name not in EXCLUDED_NAMES and path.suffix not in EXCLUDED_SUFFIXES


def build_package(repo_root: Path, output: Path | None = None) -> Path:
    """Write a clean ZIP containing source, tests, docs, notebooks, and deck."""
    repo_root = repo_root.resolve()
    missing = sorted(name for name in REQUIRED_FILES if not (repo_root / name).is_file())
    if missing:
        raise FileNotFoundError(
            "Cannot package the submission; required files are missing: " + ", ".join(missing)
        )

    output = output or repo_root / "dist" / PACKAGE_NAME
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(
        (path for path in repo_root.rglob("*") if path.is_file() and _included(path, repo_root)),
        key=lambda path: path.as_posix(),
    )
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            relative = path.relative_to(repo_root)
            archive.write(path, Path(ARCHIVE_ROOT) / relative)
    return output


def main(repo_root: Path | None = None) -> None:
    root = repo_root or Path(__file__).resolve().parents[2]
    output = build_package(root)
    size_mb = output.stat().st_size / 1_000_000
    print(f"wrote {output} ({size_mb:.1f} MB)")
