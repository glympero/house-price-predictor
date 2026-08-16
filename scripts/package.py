"""Command-line wrapper for the installed packaging implementation."""

from pathlib import Path

from house_prices.package import main

if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1])
