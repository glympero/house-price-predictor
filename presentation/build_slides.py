"""Build the case study deck from the trained artifact.

Run with ``uv run --group slides python presentation/build_slides.py``.

Every metric on the slides is read from ``models/metadata.json`` and
``models/evaluation.json`` rather than typed in, so the deck cannot drift away
from the model that was actually trained. Run training and evaluation first.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt
from sklearn.linear_model import LinearRegression

from house_prices import config
from house_prices.data import load_dataset
from house_prices.train import Candidate, run_cv_comparison, split_data

OUTPUT = Path(__file__).parent / "slides.pptx"

INK = RGBColor(0x1C, 0x24, 0x30)
MUTED = RGBColor(0x66, 0x70, 0x85)
ACCENT = RGBColor(0x2F, 0x5D, 0x9E)


def load_results() -> tuple[dict, dict]:
    metadata = json.loads((config.MODELS_DIR / config.METADATA_FILENAME).read_text())
    evaluation = json.loads((config.MODELS_DIR / config.EVALUATION_FILENAME).read_text())
    return metadata, evaluation


def cv_rmse(metadata: dict, model: str) -> float:
    for row in metadata["cv_comparison"]:
        if row["model"] == model:
            return float(row["cv_rmse"])
    raise KeyError(model)


def log_only_cv_rmse() -> float:
    """Score the intermediate step: log distance added, age squared not yet.

    The saved comparison holds the endpoints of the progression but not this
    middle step, so it is measured here with the same protocol rather than
    written in by hand.
    """
    X_train, _, y_train, _ = split_data(load_dataset())
    candidate = Candidate(
        name="log_only",
        estimator=LinearRegression(),
        scale=True,
        complexity=1,
        purpose="",
        features={"log_mrt_distance": True, "age_squared": False},
    )
    comparison = run_cv_comparison(X_train, y_train, [candidate])
    return float(comparison["cv_rmse"].iloc[0])


def example_prediction() -> float:
    """One worked prediction for the uncertainty slide, from the real artifact."""
    artifact = joblib.load(config.MODELS_DIR / config.MODEL_FILENAME)
    house = pd.DataFrame(
        [
            {
                "house_age_years": 10.0,
                "mrt_distance_m": 250.0,
                "n_convenience_stores": 6,
                "latitude": 24.975,
                "longitude": 121.540,
            }
        ]
    )
    return float(artifact["point"].predict(house)[0])


def add_title_slide(prs: Presentation, title: str, subtitle: str) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[0])
    slide.shapes.title.text = title
    slide.placeholders[1].text = subtitle


def add_bullets(prs: Presentation, title: str, bullets: list) -> None:
    """bullets: list of str, or (text, indent_level) tuples."""
    slide = prs.slides.add_slide(prs.slide_layouts[1])
    slide.shapes.title.text = title
    frame = slide.placeholders[1].text_frame
    frame.clear()
    for index, item in enumerate(bullets):
        text, level = item if isinstance(item, tuple) else (item, 0)
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = text
        paragraph.level = level
        paragraph.font.size = Pt(18 if level == 0 else 15)
        paragraph.font.color.rgb = INK if level == 0 else MUTED
    return slide


def add_table_slide(
    prs: Presentation, title: str, headers: list, rows: list, note: str = ""
) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = title

    shape = slide.shapes.add_table(
        len(rows) + 1, len(headers), Inches(0.6), Inches(1.6), Inches(9.0), Inches(0.4)
    )
    table = shape.table
    for column, heading in enumerate(headers):
        cell = table.cell(0, column)
        cell.text = heading
        cell.text_frame.paragraphs[0].font.size = Pt(13)
        cell.text_frame.paragraphs[0].font.bold = True
    for row_index, row in enumerate(rows, start=1):
        for column, value in enumerate(row):
            cell = table.cell(row_index, column)
            cell.text = str(value)
            cell.text_frame.paragraphs[0].font.size = Pt(12)

    if note:
        box = slide.shapes.add_textbox(Inches(0.6), Inches(6.2), Inches(9.0), Inches(0.9))
        paragraph = box.text_frame.paragraphs[0]
        paragraph.text = note
        paragraph.font.size = Pt(13)
        paragraph.font.color.rgb = ACCENT


def build(metadata: dict, evaluation: dict) -> Presentation:
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)

    baseline = cv_rmse(metadata, "mean_baseline")
    raw_linear = cv_rmse(metadata, "linear_regression")
    engineered = cv_rmse(metadata, "linear_engineered")
    ridge = cv_rmse(metadata, "ridge_engineered")
    forest = cv_rmse(metadata, "random_forest")
    boosting = cv_rmse(metadata, "gradient_boosting")
    bounds = metadata["residual_bounds"]
    coverage = evaluation["interval_coverage"]
    log_only = log_only_cv_rmse()
    example = example_prediction()

    # 1. Problem and outcome
    add_title_slide(
        prs,
        "Real Estate Price Estimation",
        "From EDA to a production API\n"
        f"{metadata['n_dataset_rows']} transactions, 6 raw predictors, "
        "target: price per unit area\n"
        "Deliverable: model, API, and demo UI",
    )

    # 2. Data and EDA findings
    add_bullets(
        prs,
        "What the data showed",
        [
            "MRT distance: heavy right tail, and a curved relationship with price",
            ("The effect of moving away from a station is not constant", 1),
            "House age: U-shaped, price falls then rises again",
            ("A single linear coefficient cannot represent that", 1),
            "Location: latitude, longitude and MRT distance overlap heavily",
            "One high-price outlier at 117.5, kept",
            ("So both MAE and RMSE are reported", 1),
            "Transaction date: weak relationship, and questionable production semantics",
        ],
    )

    # 3. Modeling progression
    add_table_slide(
        prs,
        "Modeling progression",
        ["Step", "CV RMSE", "Why"],
        [
            ["Mean baseline", f"{baseline:.2f}", "minimum useful bar"],
            ["Raw linear regression", f"{raw_linear:.3f}", "simple interpretable benchmark"],
            ["+ log(MRT distance)", f"{log_only:.3f}", "EDA showed a curved distance effect"],
            ["+ age squared", f"{engineered:.3f}", "EDA showed a U-shaped age effect"],
            ["Ridge", f"{ridge:.3f}", "test regularization"],
            ["Random Forest", f"{forest:.3f}", "nonlinear benchmark"],
            ["Gradient Boosting", f"{boosting:.3f}", "second nonlinear benchmark"],
        ],
        note="Each linear improvement came from a specific EDA observation, not from tuning.",
    )

    # 4. What improved and what did not
    add_bullets(
        prs,
        "What improved, and what did not",
        [
            "Kept",
            (f"log(MRT distance): {raw_linear:.3f} to {log_only:.3f}, a clear improvement", 1),
            (f"age squared: {log_only:.3f} to {engineered:.3f}, a further improvement", 1),
            "Tested and not useful",
            (f"Ridge and Lasso: {engineered:.3f} to {ridge:.3f}, little change", 1),
            ("Lasso set longitude to zero, matching the EDA overlap finding", 1),
            ("distance from centre: no improvement, removed from the pipeline", 1),
            "Failed experiments are reported, not hidden",
        ],
    )

    # 5. Final model choice
    add_table_slide(
        prs,
        "Choosing the production model",
        ["Engineered linear", "Random Forest"],
        [
            [f"CV RMSE {engineered:.3f}", f"CV RMSE {forest:.3f}, the lowest"],
            ["6 coefficients and an intercept", "300 trees"],
            ["Coefficients read directly", "Needs permutation importance"],
            ["Direct EDA interpretation", "Captures nonlinearity automatically"],
            ["Small artifact, fast retrain", "Larger model to serve"],
        ],
        note=(
            "Random Forest had the best average score. The engineered linear model was close "
            "enough that the extra complexity was not compelling for this prototype."
        ),
    )

    # 6. Final evaluation
    add_bullets(
        prs,
        "Final evaluation",
        [
            f"Cross-validated RMSE {engineered:.2f} on {metadata['n_training_rows']} training rows",
            f"Holdout RMSE {evaluation['rmse']:.2f}, "
            f"MAE {evaluation['mae']:.2f}, "
            f"R squared {evaluation['r2']:.2f} on {evaluation['n_test_rows']} rows",
            "Cross-validation is the main comparison estimate",
            ("The holdout is a final check on unseen data, used once", 1),
            "Limitations",
            (f"{metadata['n_dataset_rows']} observations, one district, 2012 to 2013", 1),
            ("Not a current valuation model", 1),
        ],
    )

    # 7. Production architecture
    add_bullets(
        prs,
        "Production architecture",
        [
            "Demo UI and API clients",
            ("FastAPI", 1),
            ("Pydantic validation, then model-support validation", 1),
            ("sklearn Pipeline: FeatureEngineer, StandardScaler, LinearRegression", 1),
            ("Prediction plus an empirical range", 1),
            "The model is loaded once at startup, not per request",
            "The same pipeline object is used in training and serving",
            "/health and /model/info expose readiness and provenance",
        ],
    )

    # 8. What serving tests discovered
    add_bullets(
        prs,
        "Boundary testing changed the product contract",
        [
            "age squared improved cross-validated performance in domain",
            "A 95-year-old property 9 km from a station exposed polynomial extrapolation",
            ("Before: an implausible prediction, with a warning attached", 1),
            ("A warning does not stop a number being used", 1),
            ("After: 422 outside_model_support, and no prediction", 1),
            "The input is not clipped, because that answers a different question",
            "A physically possible input is not necessarily one the model is qualified to predict",
        ],
    )

    # 9. Uncertainty redesign
    add_bullets(
        prs,
        "Simplifying the uncertainty range",
        [
            "Before: one linear point model plus two gradient boosting quantile models",
            ("Three estimators to explain and maintain, and about 82% coverage", 1),
            "After: out-of-fold residuals from the selected model",
            (f"5th and 95th percentiles: {bounds['lower']:.2f} and +{bounds['upper']:.2f}", 1),
            (
                f"Example: {example:.1f} gives a range of "
                f"{example + bounds['lower']:.1f} to {example + bounds['upper']:.1f}",
                1,
            ),
            f"About {coverage:.0%} coverage on the {evaluation['n_test_rows']}-row holdout",
            (
                "Exploratory, not an independently validated guarantee: the interval was "
                "revised after that holdout had been inspected",
                1,
            ),
            "Known limitation: the width is the same for every property",
        ],
    )

    # 10. Next steps
    add_bullets(
        prs,
        "What I would do next",
        [
            "Collect more recent and geographically broader transactions",
            "Monitor prediction and residual distributions in production",
            "Retrain as new transactions arrive",
            "Reassess the uncertainty range with more data",
            "Test time-aware validation for a continuously updated product",
        ],
    )

    return prs


def main() -> None:
    metadata, evaluation = load_results()
    prs = build(metadata, evaluation)
    prs.save(OUTPUT)
    print(f"wrote {OUTPUT} with {len(prs.slides)} slides")


if __name__ == "__main__":
    main()
