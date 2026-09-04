"""
RazorGuard end-to-end return-risk prediction pipeline.

Flow:
Input
  -> Input validation
  -> CatBoost prediction
  -> SHAP evidence
  -> Structured validation
  -> LLM explanation
  -> LLM validation
  -> Final result
"""

from pathlib import Path

import pandas as pd
import shap
from catboost import CatBoostClassifier

from src.validator import (
    validate_input_order,
    validate_prediction,
    validate_llm_explanation,
    get_shap_evidence,
    generate_llm_explanation,
)


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "razorguard_catboost_baseline.cbm"

MODEL_FEATURES = [
    "return_rate_90d",
    "category_return_rate",
    "is_cod",
    "order_value",
    "discount_pct",
    "delivery_distance_km",
    "previous_returns",
    "customer_age_days",
    "previous_orders",
    "product_rating",
    "orders_last_30d",
    "region_code",
    "browser_family",
    "device_age_days",
    "session_hour",
]

CAT_FEATURES = [
    "region_code",
    "browser_family",
]


# ============================================================
# Load model
# ============================================================

def load_model():
    """Load the frozen CatBoost model."""
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))

    return model


# Load once when module is imported.
final_model = load_model()


# ============================================================
# End-to-end prediction
# ============================================================

def run_end_to_end(order_row):
    """
    Run the complete RazorGuard prediction pipeline.

    Parameters
    ----------
    order_row : pandas.DataFrame
        A single-row DataFrame containing the 15 model features.

    Returns
    -------
    dict
        Prediction, SHAP evidence, explanations and validation results.
    """

    # ---------------------------------------------------------
    # 1. Validate input
    # ---------------------------------------------------------

    input_validation = validate_input_order(order_row)

    if not input_validation["valid"]:
        return {
            "status": "FAILED",
            "errors": input_validation["errors"],
        }

    # ---------------------------------------------------------
    # 2. Prepare input
    # ---------------------------------------------------------

    input_df = order_row.copy()

    # Ensure exact model feature order.
    input_df = input_df[MODEL_FEATURES]

    # CatBoost categorical columns must be strings.
    for col in CAT_FEATURES:
        input_df[col] = input_df[col].astype(str)

    # ---------------------------------------------------------
    # 3. Model prediction
    # ---------------------------------------------------------

    probability = float(
        final_model.predict_proba(input_df)[:, 1][0]
    )

    # Risk level
    if probability >= 0.50:
        risk_level = "HIGH"
    elif probability >= 0.20:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # ---------------------------------------------------------
    # 4. Generate SHAP values
    # ---------------------------------------------------------

    explainer = shap.TreeExplainer(final_model)

    shap_values = explainer.shap_values(input_df)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # ---------------------------------------------------------
    # 5. Generate structured SHAP evidence
    # ---------------------------------------------------------

    predicted_probabilities = pd.Series(
        [probability],
        index=input_df.index,
    )

    evidence_result = get_shap_evidence(
        0,
        shap_values,
        input_df,
        predicted_probabilities,
        MODEL_FEATURES,
        top_k=5,
    )

    # ---------------------------------------------------------
    # 6. Build structured evidence
    # ---------------------------------------------------------

    structured_result = {
        "model": {
            "name": "CatBoost",
            "version": "frozen_v1",
            "roc_auc": 0.6733197900795667,
            "pr_auc": 0.33405353392660825,
        },

        "prediction": {
            "probability": probability,
            "risk_level": risk_level,
        },

        "evidence": evidence_result["top_drivers"],

        "deterministic_explanation": (
            f"Predicted return risk is {probability:.1%} "
            f"({risk_level}). "
            + "; ".join(
                f"{driver['feature']} = {driver['value']} "
                f"{driver['direction'].replace('_', ' ')}"
                for driver in evidence_result["top_drivers"][:3]
            )
            + "."
        ),
    }

    # ---------------------------------------------------------
    # 7. Validate structured evidence
    # ---------------------------------------------------------

    structured_validation = validate_prediction(
        structured_result
    )

    if not structured_validation["valid"]:
        return {
            "status": "FAILED",
            "stage": "structured_validation",
            "validation": structured_validation,
        }

    # ---------------------------------------------------------
    # 8. Prepare evidence for LLM
    # ---------------------------------------------------------

    llm_evidence = {
        "risk_probability": probability,
        "risk_level": risk_level,
        "top_drivers": structured_result["evidence"][:5],
    }

    # ---------------------------------------------------------
    # 9. Generate LLM explanation
    # ---------------------------------------------------------

    explanation = generate_llm_explanation(
        llm_evidence
    )

    # ---------------------------------------------------------
    # 10. Validate LLM explanation
    # ---------------------------------------------------------

    llm_validation = validate_llm_explanation(
        explanation,
        structured_result,
    )

    # ---------------------------------------------------------
    # 11. Final result
    # ---------------------------------------------------------

    return {
        "status": (
            "SUCCESS"
            if llm_validation["valid"]
            else "FAILED"
        ),

        "prediction": structured_result["prediction"],

        "evidence": structured_result["evidence"],

        "deterministic_explanation":
            structured_result["deterministic_explanation"],

        "llm_explanation": explanation,

        "validation": {
            "structured": structured_validation,
            "llm": llm_validation,
        },
    }