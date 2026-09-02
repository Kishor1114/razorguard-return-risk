import pandas as pd

from src.validator import validate_input_order


def make_valid_order():
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


def test_valid_order():
    result = validate_input_order(make_valid_order())

    assert result["valid"] is True
    assert result["errors"] == []


def test_wrong_numeric_type():
    order = make_valid_order()
    order["order_value"] = order["order_value"].astype(object)
    order.loc[0, "order_value"] = "not_a_number"

    result = validate_input_order(order)

    assert result["valid"] is False
    assert "Feature 'order_value' must be numeric." in result["errors"]


def test_missing_required_column():
    order = make_valid_order()
    order = order.drop(columns=["order_value"])

    result = validate_input_order(order)

    assert result["valid"] is False
    assert "Missing required features: ['order_value']" in result["errors"]


def test_missing_numeric_value():
    order = make_valid_order()
    order.loc[0, "order_value"] = None

    result = validate_input_order(order)

    assert result["valid"] is False
    assert "Feature 'order_value' cannot be missing." in result["errors"]


def test_invalid_boolean():
    order = make_valid_order()
    order["is_cod"] = order["is_cod"].astype(object)
    order.loc[0, "is_cod"] = "yes"

    result = validate_input_order(order)

    assert result["valid"] is False
    assert "Feature 'is_cod' must be boolean." in result["errors"]


def test_multiple_invalid_fields():
    order = make_valid_order()

    order["order_value"] = order["order_value"].astype(object)
    order["is_cod"] = order["is_cod"].astype(object)

    order.loc[0, "order_value"] = "bad"
    order.loc[0, "is_cod"] = "yes"

    result = validate_input_order(order)

    assert result["valid"] is False
    assert "Feature 'order_value' must be numeric." in result["errors"]
    assert "Feature 'is_cod' must be boolean." in result["errors"]