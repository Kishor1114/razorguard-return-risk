import math
import json
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY is not set.")

groq_client = Groq(api_key=groq_api_key)


ALLOWED_RISK_LEVELS = {"LOW", "MEDIUM", "HIGH"}

REQUIRED_EVIDENCE_FIELDS = {
    "feature",
    "value",
    "shap_value",
    "absolute_shap",
    "direction",
}

def validate_input_order(order_row):
    """
    Validate input order before sending it to the ML model.
    """

    errors = []

    expected_features = [
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

    if not isinstance(order_row, pd.DataFrame):
        errors.append("Input must be a pandas DataFrame.")
        return {
            "valid": False,
            "errors": errors
        }

    missing_columns = [
        col for col in expected_features
        if col not in order_row.columns
    ]

    if missing_columns:
        errors.append(
            f"Missing required features: {missing_columns}"
        )

    if errors:
        return {
            "valid": False,
            "errors": errors
        }

    numeric_features = [
        "return_rate_90d",
        "category_return_rate",
        "order_value",
        "discount_pct",
        "delivery_distance_km",
        "previous_returns",
        "customer_age_days",
        "previous_orders",
        "product_rating",
        "orders_last_30d",
        "device_age_days",
        "session_hour",
    ]

    for col in numeric_features:

        value = order_row[col].iloc[0]

        if value is None or pd.isna(value):
            errors.append(
                f"Feature '{col}' cannot be missing."
            )
            continue

        try:
            float(value)
        except (TypeError, ValueError):
            errors.append(
                f"Feature '{col}' must be numeric."
            )

    is_cod_value = order_row["is_cod"].iloc[0]

    if not isinstance(is_cod_value, (bool, np.bool_)):
        errors.append(
            "Feature 'is_cod' must be boolean."
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }

def to_python_value(value):
    """Convert NumPy/Pandas values into JSON-safe Python values."""

    if pd.isna(value):
        return None

    if isinstance(value, np.bool_):
        return bool(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    return value

def validate_prediction(result):
    """
    Validate the complete RazorGuard prediction object.

    Returns:
        {
            "valid": bool,
            "errors": list[str]
        }
    """

    errors = []

    # ---------------------------------------------------------
    # 1. Top-level structure
    # ---------------------------------------------------------

    if not isinstance(result, dict):
        return {
            "valid": False,
            "errors": ["Prediction result must be a dictionary."]
        }

    required_top_level = {
        "model",
        "prediction",
        "evidence",
        "deterministic_explanation",
    }

    missing = required_top_level - result.keys()

    if missing:
        errors.append(
            f"Missing top-level fields: {sorted(missing)}"
        )

    if errors:
        return {
            "valid": False,
            "errors": errors
        }

    # ---------------------------------------------------------
    # 2. Model metadata
    # ---------------------------------------------------------

    model = result["model"]

    if not isinstance(model, dict):
        errors.append("model must be a dictionary.")
    else:
        for field in ["name", "version", "roc_auc", "pr_auc"]:
            if field not in model:
                errors.append(
                    f"Missing model field: {field}"
                )

        for metric in ["roc_auc", "pr_auc"]:
            if metric in model:
                value = model[metric]

                if not isinstance(value, (int, float)):
                    errors.append(
                        f"{metric} must be numeric."
                    )
                elif not 0 <= value <= 1:
                    errors.append(
                        f"{metric} must be between 0 and 1."
                    )

    # ---------------------------------------------------------
    # 3. Prediction
    # ---------------------------------------------------------

    prediction = result["prediction"]

    if not isinstance(prediction, dict):
        errors.append("prediction must be a dictionary.")
    else:

        if "probability" not in prediction:
            errors.append("Missing prediction.probability")

        if "risk_level" not in prediction:
            errors.append("Missing prediction.risk_level")

        if "probability" in prediction:

            probability = prediction["probability"]

            if not isinstance(probability, (int, float)):
                errors.append(
                    "Prediction probability must be numeric."
                )
            elif not math.isfinite(probability):
                errors.append(
                    "Prediction probability must be finite."
                )
            elif not 0 <= probability <= 1:
                errors.append(
                    "Prediction probability must be between 0 and 1."
                )

        if "risk_level" in prediction:

            risk_level = prediction["risk_level"]

            if risk_level not in ALLOWED_RISK_LEVELS:
                errors.append(
                    f"Invalid risk level: {risk_level}"
                )

    # ---------------------------------------------------------
    # 4. SHAP evidence
    # ---------------------------------------------------------

    evidence = result["evidence"]

    if not isinstance(evidence, list):
        errors.append("evidence must be a list.")
    elif len(evidence) == 0:
        errors.append("evidence cannot be empty.")
    else:

        for i, item in enumerate(evidence):

            if not isinstance(item, dict):
                errors.append(
                    f"Evidence item {i} must be a dictionary."
                )
                continue

            missing_fields = REQUIRED_EVIDENCE_FIELDS - item.keys()

            if missing_fields:
                errors.append(
                    f"Evidence item {i} missing fields: "
                    f"{sorted(missing_fields)}"
                )
                continue

            shap_value = item["shap_value"]
            absolute_shap = item["absolute_shap"]
            direction = item["direction"]

            # SHAP values must be numeric
            if not isinstance(shap_value, (int, float)):
                errors.append(
                    f"Evidence item {i}: shap_value must be numeric."
                )
                continue

            if not isinstance(absolute_shap, (int, float)):
                errors.append(
                    f"Evidence item {i}: absolute_shap must be numeric."
                )
                continue

            # absolute_shap must equal abs(shap_value)
            if not math.isclose(
                absolute_shap,
                abs(shap_value),
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                errors.append(
                    f"Evidence item {i}: "
                    "absolute_shap does not match abs(shap_value)."
                )

            # Direction must match SHAP sign
            if shap_value > 0:
                expected_direction = "increases_risk"
            elif shap_value < 0:
                expected_direction = "decreases_risk"
            else:
                expected_direction = "neutral"

            if direction != expected_direction:
                errors.append(
                    f"Evidence item {i}: direction "
                    f"'{direction}' does not match SHAP sign."
                )

    # ---------------------------------------------------------
    # 5. Deterministic explanation
    # ---------------------------------------------------------

    deterministic_explanation = result["deterministic_explanation"]

    if not isinstance(deterministic_explanation, str):
        errors.append(
            "deterministic_explanation must be a string."
        )
    elif not deterministic_explanation.strip():
        errors.append(
            "deterministic_explanation cannot be empty."
        )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    return {
        "valid": len(errors) == 0,
        "errors": errors,
    }

if __name__ == "__main__":
    print("RazorGuard validator module loaded successfully.")


def validate_llm_explanation(
    explanation,
    structured_evidence,
    top_k=5
):
    """
    Validate an LLM-generated explanation against
    structured SHAP evidence.

    The structured SHAP evidence is the source of truth.
    """

    errors = []

    # ---------------------------------------------------------
    # 1. Explanation must exist
    # ---------------------------------------------------------

    if not isinstance(explanation, str):
        return {
            "valid": False,
            "errors": ["LLM explanation must be a string."]
        }

    if not explanation.strip():
        return {
            "valid": False,
            "errors": ["LLM explanation cannot be empty."]
        }

    # ---------------------------------------------------------
    # 2. Validate structured evidence first
    # ---------------------------------------------------------

    structured_validation = validate_prediction(
        structured_evidence
    )

    if not structured_validation["valid"]:
        errors.append("Structured evidence is invalid.")
        errors.extend(structured_validation["errors"])

        return {
            "valid": False,
            "errors": errors
        }

    # ---------------------------------------------------------
    # 3. Extract trusted prediction information
    # ---------------------------------------------------------

    prediction = structured_evidence["prediction"]

    probability = prediction["probability"]
    risk_level = prediction["risk_level"]

    evidence = structured_evidence["evidence"][:top_k]

    trusted_features = {
        item["feature"]
        for item in evidence
    }

    # ---------------------------------------------------------
    # 4. Check unsupported / invented features
    # ---------------------------------------------------------

    unsupported_features = [
        "customer_income",
        "income",
        "salary",
        "credit_score",
        "customer_lifetime_value",
    ]

    explanation_lower = explanation.lower()

    for feature in unsupported_features:
        if feature in explanation_lower:
            errors.append(
                f"LLM explanation may contain an "
                f"unsupported feature: '{feature}'."
            )

    # ---------------------------------------------------------
    # 5. Check risk probability
    # ---------------------------------------------------------

    probability_percent = f"{probability * 100:.1f}%"
    probability_percent_2 = f"{probability * 100:.2f}%"

    if (
        probability_percent not in explanation
        and probability_percent_2 not in explanation
    ):
        errors.append(
            "LLM explanation does not contain the "
            "expected risk probability."
        )

    # ---------------------------------------------------------
    # 6. Check risk level
    # ---------------------------------------------------------

    if risk_level not in explanation.upper():
        errors.append(
            f"LLM explanation does not contain the "
            f"risk level '{risk_level}'."
        )

    # ---------------------------------------------------------
    # 7. Check trusted features
    # ---------------------------------------------------------

    for item in evidence:

        feature = item["feature"]

        if feature not in explanation:
            errors.append(
                f"LLM explanation is missing trusted "
                f"feature '{feature}'."
            )

    # ---------------------------------------------------------
    # 8. Check SHAP direction
    # ---------------------------------------------------------

    positive_terms = [
        "increases risk",
        "increase risk",
        "raises risk",
        "raise risk",
        "higher risk",
        "increased risk",
        "contributes to increasing risk",
    ]

    negative_terms = [
        "decreases risk",
        "decrease risk",
        "reduces risk",
        "reduce risk",
        "lowers risk",
        "lower risk",
        "decreased risk",
        "contributes to decreasing risk",
    ]

    for item in evidence:

        feature = item["feature"]
        direction = item["direction"]

        feature_position = explanation.find(feature)

        # Feature isn't present.
        # Section 7 already reports that.
        if feature_position == -1:
            continue

        context = explanation[
            max(0, feature_position - 150):
            feature_position + 350
        ].lower()

        has_positive = any(
            term in context
            for term in positive_terms
        )

        has_negative = any(
            term in context
            for term in negative_terms
        )

        if direction == "increases_risk" and not has_positive:
            errors.append(
                f"LLM explanation may contradict SHAP "
                f"direction for '{feature}'."
            )

        elif direction == "decreases_risk" and not has_negative:
            errors.append(
                f"LLM explanation may contradict SHAP "
                f"direction for '{feature}'."
            )

    # ---------------------------------------------------------
    # 9. Final validation result
    # ---------------------------------------------------------

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }

def get_shap_evidence(
    sample_position,
    shap_values,
    X_data,
    predicted_probabilities,
    feature_names,
    top_k=5
):
    """
    Convert raw SHAP output into structured, auditable evidence.
    """

    shap_row = np.asarray(
        shap_values[sample_position]
    ).flatten()

    probability = float(
        predicted_probabilities.iloc[sample_position]
        if hasattr(predicted_probabilities, "iloc")
        else predicted_probabilities[sample_position]
    )

    evidence = []

    for feature, shap_value in zip(
        feature_names,
        shap_row
    ):

        value = X_data.iloc[
            sample_position
        ][feature]

        if shap_value > 0:
            direction = "increases_risk"
        elif shap_value < 0:
            direction = "decreases_risk"
        else:
            direction = "neutral"

        evidence.append({
            "feature": feature,
            "value": to_python_value(value),
            "shap_value": float(shap_value),
            "absolute_shap": float(abs(shap_value)),
            "direction": direction
        })

    evidence = sorted(
        evidence,
        key=lambda x: x["absolute_shap"],
        reverse=True
    )

    return {
        "risk_probability": probability,
        "top_drivers": evidence[:top_k]
    }

def generate_llm_explanation(evidence):
    """
    Generate a human-readable explanation from structured SHAP evidence.

    The LLM does NOT receive raw order data.
    It only receives the model's already-computed evidence.
    """

    prompt = f"""
You are an AI risk explanation assistant.

Your job is ONLY to explain an existing ML prediction.
You must NOT recalculate risk, invent evidence, or change the model's decision.

Use ONLY the structured evidence provided below.

Risk probability:
{evidence["risk_probability"]:.4f}

Risk level:
{evidence["risk_level"]}

Top SHAP drivers:
{json.dumps(evidence["top_drivers"], indent=2, default=str)}

STRICT DIRECTION RULES:
- If a driver's direction is "increases_risk", you MUST explicitly say that
  the feature "increases risk".
- If a driver's direction is "decreases_risk", you MUST explicitly say that
  the feature "decreases risk".
- NEVER reverse or reinterpret the direction from the SHAP value.
- The "direction" field is authoritative.
- Do not describe a feature as increasing risk if its direction is
  "decreases_risk", and vice versa.

Requirements:
1. State the predicted return-risk percentage.
2. State the risk level.
3. Mention the top 5 supplied drivers.
4. For EVERY driver, preserve its supplied direction.
5. Use the supplied feature names and values.
6. You may mention SHAP values, but they must not change the stated direction.
7. Do not invent customer behavior, product information, or business context.
8. Do not recommend an action.
9. Do not claim causation. Use "contributes to" or "is associated with".
10. Do not mention unsupported features.
11. Keep the explanation under 120 words.

Return only the explanation.
"""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You explain ML predictions faithfully. "
                    "Never contradict structured evidence."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }
