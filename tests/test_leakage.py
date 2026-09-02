"""
RazorGuard - Day 1: Leakage and reproducibility tests.

IMPORTANT: the leakage tests do NOT just re-call build_dataset() and compare to
itself - that would only prove the generator is internally consistent, not that
it's actually leakage-safe. Instead, for a random sample of orders, this file
independently recomputes each historical feature directly from the raw order log
using pandas filtering (timestamp < target order's timestamp), and asserts the
recomputed value matches what the generator produced. A bug in the generator's
incremental bookkeeping would show up here even if the generator "agrees with
itself".

Run: pytest tests/test_leakage.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_generator import (
    generate_full_dataset,
    CATEGORY_PRIOR_ALPHA,
    CATEGORY_PRIOR_BETA,
    CUSTOMER_PRIOR_RETURN_RATE,
    SEED,
)

N_LEAKAGE_SAMPLE = 300  # orders to independently re-verify (full 10k also checked structurally)


@pytest.fixture(scope="module")
def dataset():
    return generate_full_dataset(SEED)


# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------

def test_reproducibility_same_seed():
    df1 = generate_full_dataset(SEED)
    df2 = generate_full_dataset(SEED)
    pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))


def test_different_seed_gives_different_data():
    df1 = generate_full_dataset(SEED)
    df2 = generate_full_dataset(SEED + 1)
    assert not df1["returned"].equals(df2["returned"])


# ----------------------------------------------------------------------------
# Structural leakage invariant: every customer's history entries used to build
# feature i must have timestamp strictly less than order i's own timestamp.
# We reconstruct this directly from the raw log's own timestamp ordering.
# ----------------------------------------------------------------------------

def test_dataset_processing_order_matches_timestamp_order(dataset):
    # The generator processes orders sorted by (timestamp, order_id). Confirm the
    # output preserves that order, which is the invariant the whole leakage
    # guarantee depends on.
    sorted_check = dataset.sort_values(["timestamp", "order_id"]).reset_index(drop=True)
    pd.testing.assert_series_equal(
        dataset["order_id"].reset_index(drop=True), sorted_check["order_id"]
    )


def test_customer_history_features_independently_recomputed(dataset):
    rng = np.random.default_rng(123)
    # Only sample orders that have at least one prior order for their customer,
    # otherwise the check is trivially "no history" on both sides.
    candidates = dataset[dataset["previous_orders"] > 0]
    sample = candidates.sample(n=min(N_LEAKAGE_SAMPLE, len(candidates)), random_state=123)

    for _, row in sample.iterrows():
        cid = row["customer_id"]
        current_ts = row["timestamp"]

        # Independent recomputation: filter the FULL raw dataset for this customer's
        # orders strictly before the current order's timestamp.
        cust_history = dataset[
            (dataset["customer_id"] == cid) & (dataset["timestamp"] < current_ts)
        ]
        # Leakage guard: none of these may be >= current_ts, and the current order
        # itself must not appear in its own history.
        assert (cust_history["timestamp"] < current_ts).all()
        assert row["order_id"] not in cust_history["order_id"].values

        expected_previous_orders = len(cust_history)
        expected_previous_returns = int(cust_history["returned"].sum())

        window_90 = cust_history[cust_history["timestamp"] >= current_ts - pd.Timedelta(days=90)]
        expected_return_rate_90d = (
            float(window_90["returned"].mean()) if len(window_90) > 0 else CUSTOMER_PRIOR_RETURN_RATE
        )

        window_30 = cust_history[cust_history["timestamp"] >= current_ts - pd.Timedelta(days=30)]
        expected_orders_last_30d = len(window_30)

        assert row["previous_orders"] == expected_previous_orders
        assert row["previous_returns"] == expected_previous_returns
        assert row["orders_last_30d"] == expected_orders_last_30d
        assert np.isclose(row["return_rate_90d"], expected_return_rate_90d, atol=1e-9)


def test_category_return_rate_independently_recomputed(dataset):
    sample = dataset.sample(n=min(N_LEAKAGE_SAMPLE, len(dataset)), random_state=7)

    for _, row in sample.iterrows():
        cat = row["category"]
        current_ts = row["timestamp"]

        cat_history = dataset[
            (dataset["category"] == cat) & (dataset["timestamp"] < current_ts)
        ]
        assert row["order_id"] not in cat_history["order_id"].values

        cat_orders = len(cat_history)
        cat_returns = int(cat_history["returned"].sum())
        expected_rate = (cat_returns + CATEGORY_PRIOR_ALPHA) / (
            cat_orders + CATEGORY_PRIOR_ALPHA + CATEGORY_PRIOR_BETA
        )
        assert np.isclose(row["category_return_rate"], expected_rate, atol=1e-9)


def test_no_order_ever_uses_its_own_outcome(dataset):
    # First order for every customer must show zero history, by definition.
    first_orders = dataset.sort_values("timestamp").groupby("customer_id").head(1)
    assert (first_orders["previous_orders"] == 0).all()
    assert (first_orders["previous_returns"] == 0).all()
    assert (first_orders["orders_last_30d"] == 0).all()
    assert np.allclose(first_orders["return_rate_90d"], CUSTOMER_PRIOR_RETURN_RATE)


# ----------------------------------------------------------------------------
# Basic sanity / non-triviality checks
# ----------------------------------------------------------------------------

def test_target_is_binary(dataset):
    assert set(dataset["returned"].unique()) <= {0, 1}


def test_return_rate_not_forced_but_plausible(dataset):
    rate = dataset["returned"].mean()
    assert 0.08 <= rate <= 0.30


def test_order_ids_unique(dataset):
    assert dataset["order_id"].is_unique


def test_no_feature_is_a_perfect_predictor(dataset):
    # Sanity check that this isn't a trivial deterministic-rule dataset: no single
    # numeric feature should perfectly separate the classes.
    numeric_cols = [
        "return_rate_90d", "category_return_rate", "order_value",
        "discount_pct", "delivery_distance_km", "previous_returns",
    ]
    for col in numeric_cols:
        corr = dataset[col].corr(dataset["returned"])
        assert abs(corr) < 0.9, f"{col} suspiciously close to a deterministic label rule (corr={corr:.3f})"


def test_temporal_split_boundaries(dataset):
    assert (dataset.loc[dataset["split"] == "train", "timestamp"].dt.month <= 5).all()
    assert (dataset.loc[dataset["split"] == "validation", "timestamp"].dt.month == 6).all()
    assert (dataset.loc[dataset["split"] == "test", "timestamp"].dt.month >= 7).all()
