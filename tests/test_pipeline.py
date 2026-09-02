import pandas as pd

from src.pipeline import run_end_to_end


def make_normal_order():
    return pd.DataFrame([{
        "return_rate_90d": 0.166667,
        "category_return_rate": 0.163885,
        "is_cod": True,
        "order_value": 1121.56,
        "discount_pct": 0.1599,
        "delivery_distance_km": 297.3,
        "previous_returns": 1,
        "customer_age_days": 528,
        "previous_orders": 12,
        "product_rating": 4.22,
        "orders_last_30d": 4,
        "region_code": "east",
        "browser_family": "chrome",
        "device_age_days": 969,
        "session_hour": 2,
    }])


def fake_llm_explanation(evidence):
    """
    Deterministic LLM replacement for automated tests.

    This avoids making pytest depend on the external Groq API.
    """
    drivers = evidence["top_drivers"]

    lines = [
        f"Predicted return-risk: "
        f"{evidence['risk_probability']:.2%}. "
        f"Risk level: {evidence['risk_level']}.",
        "Top drivers:"
    ]

    for driver in drivers:
        direction_text = driver["direction"].replace("_", " ")

        lines.append(
            f"- {driver['feature']} = {driver['value']}, "
            f"which {direction_text} "
            f"(SHAP {driver['shap_value']:+.3f})."
        )

    return "\n".join(lines)


def test_normal_order(monkeypatch):

    monkeypatch.setattr(
        "src.pipeline.generate_llm_explanation",
        fake_llm_explanation
    )

    result = run_end_to_end(make_normal_order())

    assert result["status"] == "SUCCESS"

    assert "prediction" in result

    assert 0 <= result["prediction"]["probability"] <= 1

    assert result["prediction"]["risk_level"] in {
        "LOW",
        "MEDIUM",
        "HIGH",
    }


def test_structured_validation(monkeypatch):

    monkeypatch.setattr(
        "src.pipeline.generate_llm_explanation",
        fake_llm_explanation
    )

    result = run_end_to_end(make_normal_order())

    assert result["validation"]["structured"]["valid"] is True


def test_llm_validation(monkeypatch):

    monkeypatch.setattr(
        "src.pipeline.generate_llm_explanation",
        fake_llm_explanation
    )

    result = run_end_to_end(make_normal_order())

    assert result["validation"]["llm"]["valid"] is True